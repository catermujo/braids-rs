use crate::fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use crate::model::{
    ChunkQuery, CombineNode, CombineOp, FastNoiseChange, FastNoiseGraphSpec, FastNoiseKernel,
    FastNoisePlanner, GraphDimension, NodeSpec, PositionSlots, PositionSource, Sample2DNode,
    Sample3DNode, Warp2DNode,
};
use crate::runtime::{KernelPayload, SamplePayload, make_cpu_backend, summarize_samples};
use crate::{FastNoiseCpuBackend, scenarios};
use braids::{BackendConfig, BraidExecutor, BufferSlot, PlannerBackend, Stack};
use std::sync::Arc;

fn sample_noise(seed: i32, noise_type: NoiseType, frequency: f32) -> FastNoiseLite {
    let mut noise = FastNoiseLite::with_seed(seed);
    noise.set_noise_type(Some(noise_type));
    noise.set_frequency(Some(frequency));
    noise.set_fractal_type(Some(FractalType::FBm));
    noise.set_fractal_octaves(Some(3));
    noise
}

fn sample_noise_node(seed: i32, noise_type: NoiseType, frequency: f32) -> Arc<FastNoiseLite> {
    Arc::new(sample_noise(seed, noise_type, frequency))
}

fn make_stack(
    spec: FastNoiseGraphSpec,
) -> braids::BraidResult<Stack<FastNoisePlanner, FastNoiseCpuBackend>> {
    let executor = Arc::new(BraidExecutor::new(4));
    let backend = executor.register_backend(
        Arc::new(make_cpu_backend()),
        BackendConfig { lane_count: 4 },
    );
    Stack::create(executor, Arc::new(FastNoisePlanner), backend, spec)
}

fn run_one(
    stack: &Stack<FastNoisePlanner, FastNoiseCpuBackend>,
    query: ChunkQuery,
) -> braids::BraidResult<crate::ChunkSummary> {
    let job = stack.dispatch(vec![query])?;
    let mut values = stack.collect(job)?;
    Ok(values.remove(0))
}

#[test]
fn update_remove_and_set_final_field_recompile() {
    let spec = FastNoiseGraphSpec {
        dimension: GraphDimension::D2,
        final_field: "left".to_owned(),
        nodes: vec![
            NodeSpec::Sample2D(Sample2DNode {
                id: "left".to_owned(),
                source: PositionSource::Base,
                noise: sample_noise_node(11, NoiseType::Perlin, 0.01),
            }),
            NodeSpec::Sample2D(Sample2DNode {
                id: "right".to_owned(),
                source: PositionSource::Base,
                noise: sample_noise_node(19, NoiseType::Value, 0.02),
            }),
        ],
    };
    let stack = make_stack(spec).expect("stack");
    let query = ChunkQuery::Grid2D {
        width: 8,
        height: 6,
        origin: [0.0, 0.0],
        step: [1.0, 1.0],
    };
    let left = run_one(&stack, query.clone()).expect("left");

    let mut updated = sample_noise(23, NoiseType::OpenSimplex2, 0.03);
    updated.set_fractal_gain(Some(0.7));
    stack
        .update(&[
            FastNoiseChange::UpsertNode(NodeSpec::Sample2D(Sample2DNode {
                id: "right".to_owned(),
                source: PositionSource::Base,
                noise: Arc::new(updated),
            })),
            FastNoiseChange::SetFinalField {
                id: "right".to_owned(),
            },
        ])
        .expect("update");
    let right = run_one(&stack, query.clone()).expect("right");
    assert_ne!(left.checksum, right.checksum);

    stack
        .update(&[FastNoiseChange::RemoveNode {
            id: "left".to_owned(),
        }])
        .expect("remove unused node");
    let after_remove = run_one(&stack, query).expect("after remove");
    assert_eq!(right.samples, after_remove.samples);
    assert_eq!(right.checksum, after_remove.checksum);
}

#[test]
fn packet_sizing_and_offsets_follow_query_shapes() {
    let planner = FastNoisePlanner;
    let spec = FastNoiseGraphSpec {
        dimension: GraphDimension::D2,
        final_field: "sample".to_owned(),
        nodes: vec![NodeSpec::Sample2D(Sample2DNode {
            id: "sample".to_owned(),
            source: PositionSource::Base,
            noise: sample_noise_node(31, NoiseType::Perlin, 0.02),
        })],
    };
    let state = planner.init_state(&spec).expect("state");
    let plan = planner
        .compile(&state, &mut braids::PlannerScratch::default())
        .expect("plan");
    let mut packet = braids::JobPacket::default();
    planner
        .encode_batch(
            &plan,
            &[
                ChunkQuery::Grid2D {
                    width: 4,
                    height: 3,
                    origin: [0.0, 0.0],
                    step: [1.0, 1.0],
                },
                ChunkQuery::Grid2D {
                    width: 2,
                    height: 2,
                    origin: [10.0, 5.0],
                    step: [0.5, 0.5],
                },
            ],
            &mut packet,
            &mut braids::BatchScratch::default(),
        )
        .expect("encode");

    assert_eq!(packet.query_count, 2);
    assert_eq!(
        packet
            .slice::<u32>(crate::model::SLOT_QUERY_META)
            .expect("meta"),
        &[4, 3, 1, 2, 2, 1]
    );
    assert_eq!(
        packet
            .slice::<u32>(crate::model::SLOT_QUERY_OFFSETS)
            .expect("offsets"),
        &[0, 12, 16]
    );
}

#[test]
fn sample3d_payload_roundtrip_preserves_fields() {
    let mut noise = sample_noise(37, NoiseType::OpenSimplex2S, 0.125);
    noise.set_fractal_lacunarity(Some(2.5));
    noise.set_fractal_gain(Some(0.45));
    let noise = Arc::new(noise);

    let payload = SamplePayload::<3> {
        source: PositionSlots {
            coords: [BufferSlot(41), BufferSlot(42), BufferSlot(43)],
        },
        output: BufferSlot(99),
        noise: noise.clone(),
    };

    let mut scratch = braids::PlannerScratch::default();
    let kernel = payload.encode(&mut scratch).expect("encode");
    let decoded =
        SamplePayload::<3>::decode(FastNoiseKernel::Sample3d, &kernel.payload).expect("decode");

    assert_eq!(decoded.source.coords, payload.source.coords);
    assert_eq!(decoded.output, payload.output);
    assert_eq!(decoded.noise.seed, noise.seed);
    assert_eq!(decoded.noise.frequency, noise.frequency);
    assert_eq!(decoded.noise.noise_type, noise.noise_type);
    assert_eq!(decoded.noise.fractal_type, noise.fractal_type);
    assert_eq!(decoded.noise.octaves, noise.octaves);
    assert_eq!(decoded.noise.lacunarity, noise.lacunarity);
    assert_eq!(decoded.noise.gain, noise.gain);
}

#[test]
fn sample2d_and_warp2d_match_direct_fastnoise() {
    let sample = Arc::new(sample_noise(41, NoiseType::Perlin, 0.015));
    let warp = {
        let mut noise = FastNoiseLite::with_seed(43);
        noise.set_frequency(Some(0.02));
        noise.set_domain_warp_amp(Some(4.0));
        noise.set_fractal_type(Some(FractalType::DomainWarpProgressive));
        Arc::new(noise)
    };
    let spec = FastNoiseGraphSpec {
        dimension: GraphDimension::D2,
        final_field: "sample".to_owned(),
        nodes: vec![
            NodeSpec::Warp2D(Warp2DNode {
                id: "warp".to_owned(),
                source: PositionSource::Base,
                noise: warp.clone(),
            }),
            NodeSpec::Sample2D(Sample2DNode {
                id: "sample".to_owned(),
                source: PositionSource::Node("warp".to_owned()),
                noise: sample.clone(),
            }),
        ],
    };
    let stack = make_stack(spec).expect("stack");
    let query = ChunkQuery::Grid2D {
        width: 7,
        height: 5,
        origin: [2.0, -3.0],
        step: [0.75, 1.25],
    };
    let summary = run_one(&stack, query.clone()).expect("summary");

    let mut direct = Vec::new();
    if let ChunkQuery::Grid2D {
        width,
        height,
        origin,
        step,
    } = query
    {
        for y in 0..height {
            for x in 0..width {
                let px = origin[0] + x as f32 * step[0];
                let py = origin[1] + y as f32 * step[1];
                let (wx, wy) = warp.domain_warp_2d(px, py);
                direct.push(sample.get_noise_2d(wx, wy));
            }
        }
    }
    assert_eq!(summary, summarize_samples(direct.as_slice()));
}

#[test]
fn combine_and_ygradient_match_expected_math() {
    let sample_a = Arc::new(sample_noise(53, NoiseType::Value, 0.025));
    let sample_b = Arc::new(sample_noise(59, NoiseType::Perlin, 0.031));
    let spec = FastNoiseGraphSpec {
        dimension: GraphDimension::D3,
        final_field: "final".to_owned(),
        nodes: vec![
            NodeSpec::Sample3D(Sample3DNode {
                id: "a".to_owned(),
                source: PositionSource::Base,
                noise: sample_a.clone(),
            }),
            NodeSpec::Sample3D(Sample3DNode {
                id: "b".to_owned(),
                source: PositionSource::Base,
                noise: sample_b.clone(),
            }),
            NodeSpec::Combine(CombineNode {
                id: "sum".to_owned(),
                inputs: vec!["a".to_owned(), "b".to_owned()],
                op: CombineOp::Add,
                params: Vec::new(),
            }),
            NodeSpec::Combine(CombineNode {
                id: "clamped".to_owned(),
                inputs: vec!["sum".to_owned()],
                op: CombineOp::Clamp,
                params: vec![-0.8, 0.8],
            }),
            NodeSpec::Combine(CombineNode {
                id: "final".to_owned(),
                inputs: vec!["clamped".to_owned()],
                op: CombineOp::YGradient,
                params: vec![-4.0, 4.0, 0.5, -0.5],
            }),
        ],
    };
    let stack = make_stack(spec).expect("stack");
    let query = ChunkQuery::Grid3D {
        width: 4,
        height: 5,
        depth: 3,
        origin: [1.0, -4.0, 2.0],
        step: [0.5, 2.0, 0.75],
    };
    let summary = run_one(&stack, query.clone()).expect("summary");

    let mut direct = Vec::new();
    if let ChunkQuery::Grid3D {
        width,
        height,
        depth,
        origin,
        step,
    } = query
    {
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let px = origin[0] + x as f32 * step[0];
                    let py = origin[1] + y as f32 * step[1];
                    let pz = origin[2] + z as f32 * step[2];
                    let mut value =
                        sample_a.get_noise_3d(px, py, pz) + sample_b.get_noise_3d(px, py, pz);
                    value = value.clamp(-0.8, 0.8);
                    let t = ((py + 4.0) / 8.0).clamp(0.0, 1.0);
                    value += 0.5 + ((-0.5 - 0.5) * t);
                    direct.push(value);
                }
            }
        }
    }
    assert_eq!(summary, summarize_samples(direct.as_slice()));
}

#[test]
fn terrain_and_voxel_scenarios_run_through_stack() {
    let terrain = make_stack(scenarios::terrain_height_2d()).expect("terrain");
    let voxel = make_stack(scenarios::voxel_density_3d()).expect("voxel");

    let terrain_summary = run_one(
        &terrain,
        ChunkQuery::Grid2D {
            width: 16,
            height: 16,
            origin: [64.0, 32.0],
            step: [1.0, 1.0],
        },
    )
    .expect("terrain summary");
    let voxel_summary = run_one(
        &voxel,
        ChunkQuery::Grid3D {
            width: 8,
            height: 10,
            depth: 8,
            origin: [0.0, -16.0, 0.0],
            step: [1.0, 1.0, 1.0],
        },
    )
    .expect("voxel summary");

    assert_eq!(terrain_summary.samples, 256);
    assert_eq!(voxel_summary.samples, 640);
}

#[test]
fn voxel_scenario_matches_direct_fastnoise() {
    let spec = scenarios::voxel_density_3d();
    let stack = make_stack(spec.clone()).expect("stack");
    let query = ChunkQuery::Grid3D {
        width: 7,
        height: 9,
        depth: 5,
        origin: [11.0, -16.0, -7.0],
        step: [1.25, 0.5, 0.75],
    };
    let summary = run_one(&stack, query.clone()).expect("summary");

    let warp = spec
        .nodes
        .iter()
        .find_map(|node| match node {
            NodeSpec::Warp3D(node) if node.id == scenarios::VOXEL_WARP_NODE => {
                Some(node.noise.clone())
            }
            _ => None,
        })
        .expect("warp");
    let base = spec
        .nodes
        .iter()
        .find_map(|node| match node {
            NodeSpec::Sample3D(node) if node.id == scenarios::VOXEL_BASE_NODE => {
                Some(node.noise.clone())
            }
            _ => None,
        })
        .expect("base");
    let cave = spec
        .nodes
        .iter()
        .find_map(|node| match node {
            NodeSpec::Sample3D(node) if node.id == scenarios::VOXEL_CAVE_NODE => {
                Some(node.noise.clone())
            }
            _ => None,
        })
        .expect("cave");

    let mut direct = Vec::new();
    if let ChunkQuery::Grid3D {
        width,
        height,
        depth,
        origin,
        step,
    } = query
    {
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let px = origin[0] + x as f32 * step[0];
                    let py = origin[1] + y as f32 * step[1];
                    let pz = origin[2] + z as f32 * step[2];
                    let base_value = base.get_noise_3d(px, py, pz);
                    let (wx, wy, wz) = warp.domain_warp_3d(px, py, pz);
                    let cave_value = cave.get_noise_3d(wx, wy, wz);
                    let mut value = base_value;
                    value -= cave_value;
                    let t = ((py + 32.0) / 64.0).clamp(0.0, 1.0);
                    let gradient = 1.0 + ((-1.0 - 1.0) * t);
                    value += gradient;
                    direct.push(value);
                }
            }
        }
    }

    assert_eq!(summary, summarize_samples(direct.as_slice()));
}

#[test]
fn dependency_chain_updates_downstream_without_cross_stack_corruption() {
    let biome = make_stack(scenarios::biome_control_2d()).expect("biome");
    let terrain = make_stack(scenarios::terrain_height_2d()).expect("terrain");
    let voxel = make_stack(scenarios::voxel_density_3d()).expect("voxel");

    let biome_query = ChunkQuery::Grid2D {
        width: 12,
        height: 12,
        origin: [10.0, 20.0],
        step: [1.0, 1.0],
    };
    let terrain_query = ChunkQuery::Grid2D {
        width: 12,
        height: 12,
        origin: [10.0, 20.0],
        step: [1.0, 1.0],
    };
    let voxel_query = ChunkQuery::Grid3D {
        width: 8,
        height: 12,
        depth: 8,
        origin: [10.0, -16.0, 20.0],
        step: [1.0, 1.0, 1.0],
    };

    let biome_summary = run_one(&biome, biome_query.clone()).expect("biome summary");
    terrain
        .update(&scenarios::terrain_patch_from_biome(&biome_summary))
        .expect("terrain patch");
    let terrain_summary = run_one(&terrain, terrain_query.clone()).expect("terrain summary");
    voxel
        .update(&scenarios::voxel_patch_from_terrain(&terrain_summary))
        .expect("voxel patch");
    let voxel_summary = run_one(&voxel, voxel_query).expect("voxel summary");

    let biome_again = run_one(&biome, biome_query).expect("biome again");
    assert_eq!(biome_summary.checksum, biome_again.checksum);
    assert_ne!(terrain_summary.checksum, 0);
    assert_ne!(voxel_summary.checksum, 0);
}
