use crate::fastnoise_lite::{DomainWarpType, FastNoiseLite, FractalType, NoiseType};
use crate::model::{
    ChunkSummary, CombineNode, CombineOp, FastNoiseChange, FastNoiseGraphSpec, GraphDimension,
    NodeSpec, PositionSource, Sample2DNode, Sample3DNode, Warp2DNode, Warp3DNode,
};
use std::sync::Arc;

pub const BIOME_WARP_NODE: &str = "biome_warp";
pub const BIOME_MOISTURE_NODE: &str = "biome_moisture";
pub const BIOME_TEMPERATURE_NODE: &str = "biome_temperature";
pub const BIOME_FINAL_NODE: &str = "biome_final";

pub const TERRAIN_WARP_NODE: &str = "terrain_warp";
pub const TERRAIN_CONTINENT_NODE: &str = "terrain_continent";
pub const TERRAIN_EROSION_NODE: &str = "terrain_erosion";
pub const TERRAIN_PEAKS_NODE: &str = "terrain_peaks";
pub const TERRAIN_DETAIL_NODE: &str = "terrain_detail";
pub const TERRAIN_FINAL_NODE: &str = "terrain_final";

pub const VOXEL_WARP_NODE: &str = "voxel_warp";
pub const VOXEL_BASE_NODE: &str = "voxel_base";
pub const VOXEL_CAVE_NODE: &str = "voxel_cave";
pub const VOXEL_SHAPE_NODE: &str = "voxel_shape";
pub const VOXEL_FINAL_NODE: &str = "voxel_final";

pub fn biome_control_2d() -> FastNoiseGraphSpec {
    FastNoiseGraphSpec {
        dimension: GraphDimension::D2,
        final_field: BIOME_FINAL_NODE.to_owned(),
        nodes: vec![
            NodeSpec::Warp2D(Warp2DNode {
                id: BIOME_WARP_NODE.to_owned(),
                source: PositionSource::Base,
                noise: warp_noise(701, 0.0025, 18.0),
            }),
            NodeSpec::Sample2D(Sample2DNode {
                id: BIOME_MOISTURE_NODE.to_owned(),
                source: PositionSource::Node(BIOME_WARP_NODE.to_owned()),
                noise: Arc::new(sample_noise(
                    711,
                    NoiseType::Perlin,
                    FractalType::FBm,
                    0.0022,
                    4,
                )),
            }),
            NodeSpec::Sample2D(Sample2DNode {
                id: BIOME_TEMPERATURE_NODE.to_owned(),
                source: PositionSource::Node(BIOME_WARP_NODE.to_owned()),
                noise: Arc::new(sample_noise(
                    719,
                    NoiseType::OpenSimplex2,
                    FractalType::FBm,
                    0.0018,
                    4,
                )),
            }),
            NodeSpec::Combine(CombineNode {
                id: BIOME_FINAL_NODE.to_owned(),
                inputs: vec![
                    BIOME_MOISTURE_NODE.to_owned(),
                    BIOME_TEMPERATURE_NODE.to_owned(),
                ],
                op: CombineOp::Add,
                params: Vec::new(),
            }),
        ],
    }
}

pub fn terrain_height_2d() -> FastNoiseGraphSpec {
    FastNoiseGraphSpec {
        dimension: GraphDimension::D2,
        final_field: TERRAIN_FINAL_NODE.to_owned(),
        nodes: vec![
            NodeSpec::Warp2D(Warp2DNode {
                id: TERRAIN_WARP_NODE.to_owned(),
                source: PositionSource::Base,
                noise: warp_noise(1001, 0.0015, 24.0),
            }),
            NodeSpec::Sample2D(Sample2DNode {
                id: TERRAIN_CONTINENT_NODE.to_owned(),
                source: PositionSource::Node(TERRAIN_WARP_NODE.to_owned()),
                noise: Arc::new(sample_noise(
                    1011,
                    NoiseType::OpenSimplex2,
                    FractalType::FBm,
                    0.0012,
                    5,
                )),
            }),
            NodeSpec::Sample2D(Sample2DNode {
                id: TERRAIN_EROSION_NODE.to_owned(),
                source: PositionSource::Node(TERRAIN_WARP_NODE.to_owned()),
                noise: Arc::new(sample_noise(
                    1019,
                    NoiseType::Perlin,
                    FractalType::FBm,
                    0.0032,
                    4,
                )),
            }),
            NodeSpec::Sample2D(Sample2DNode {
                id: TERRAIN_PEAKS_NODE.to_owned(),
                source: PositionSource::Node(TERRAIN_WARP_NODE.to_owned()),
                noise: Arc::new(sample_noise(
                    1027,
                    NoiseType::OpenSimplex2S,
                    FractalType::Ridged,
                    0.0056,
                    4,
                )),
            }),
            NodeSpec::Sample2D(Sample2DNode {
                id: TERRAIN_DETAIL_NODE.to_owned(),
                source: PositionSource::Node(TERRAIN_WARP_NODE.to_owned()),
                noise: terrain_detail_noise(1033),
            }),
            NodeSpec::Combine(CombineNode {
                id: "terrain_base".to_owned(),
                inputs: vec![
                    TERRAIN_CONTINENT_NODE.to_owned(),
                    TERRAIN_PEAKS_NODE.to_owned(),
                ],
                op: CombineOp::Add,
                params: Vec::new(),
            }),
            NodeSpec::Combine(CombineNode {
                id: "terrain_shaped".to_owned(),
                inputs: vec!["terrain_base".to_owned(), TERRAIN_EROSION_NODE.to_owned()],
                op: CombineOp::Sub,
                params: Vec::new(),
            }),
            NodeSpec::Combine(CombineNode {
                id: TERRAIN_FINAL_NODE.to_owned(),
                inputs: vec!["terrain_shaped".to_owned(), TERRAIN_DETAIL_NODE.to_owned()],
                op: CombineOp::Add,
                params: Vec::new(),
            }),
        ],
    }
}

pub fn voxel_density_3d() -> FastNoiseGraphSpec {
    FastNoiseGraphSpec {
        dimension: GraphDimension::D3,
        final_field: VOXEL_FINAL_NODE.to_owned(),
        nodes: vec![
            NodeSpec::Warp3D(Warp3DNode {
                id: VOXEL_WARP_NODE.to_owned(),
                source: PositionSource::Base,
                noise: warp_noise(2001, 0.011, 6.0),
            }),
            NodeSpec::Sample3D(Sample3DNode {
                id: VOXEL_BASE_NODE.to_owned(),
                source: PositionSource::Base,
                noise: Arc::new(sample_noise(
                    2011,
                    NoiseType::OpenSimplex2,
                    FractalType::FBm,
                    0.018,
                    5,
                )),
            }),
            NodeSpec::Sample3D(Sample3DNode {
                id: VOXEL_CAVE_NODE.to_owned(),
                source: PositionSource::Node(VOXEL_WARP_NODE.to_owned()),
                noise: Arc::new(sample_noise(
                    2019,
                    NoiseType::Cellular,
                    FractalType::Ridged,
                    0.03,
                    3,
                )),
            }),
            NodeSpec::Combine(CombineNode {
                id: VOXEL_SHAPE_NODE.to_owned(),
                inputs: vec![VOXEL_BASE_NODE.to_owned(), VOXEL_CAVE_NODE.to_owned()],
                op: CombineOp::Sub,
                params: Vec::new(),
            }),
            NodeSpec::Combine(CombineNode {
                id: VOXEL_FINAL_NODE.to_owned(),
                inputs: vec![VOXEL_SHAPE_NODE.to_owned()],
                op: CombineOp::YGradient,
                params: vec![-32.0, 32.0, 1.0, -1.0],
            }),
        ],
    }
}

pub fn terrain_patch_from_biome(summary: &ChunkSummary) -> Vec<FastNoiseChange> {
    let mut noise = FastNoiseLite::with_seed(1033);
    noise.set_noise_type(Some(NoiseType::ValueCubic));
    noise.set_fractal_type(Some(FractalType::FBm));
    noise.set_fractal_octaves(Some(3));
    noise.set_fractal_lacunarity(Some(2.0));
    noise.set_fractal_gain(Some(0.45));
    noise.set_frequency(Some(0.015 + summary.taps[0].abs() * 0.02));
    noise.set_fractal_gain(Some((0.35 + summary.taps[3].abs() * 0.35).min(0.95)));
    vec![FastNoiseChange::UpsertNode(NodeSpec::Sample2D(
        Sample2DNode {
            id: TERRAIN_DETAIL_NODE.to_owned(),
            source: PositionSource::Node(TERRAIN_WARP_NODE.to_owned()),
            noise: Arc::new(noise),
        },
    ))]
}

pub fn voxel_patch_from_terrain(summary: &ChunkSummary) -> Vec<FastNoiseChange> {
    let mut noise = sample_noise(2011, NoiseType::OpenSimplex2, FractalType::FBm, 0.018, 5);
    noise.set_frequency(Some(0.014 + summary.mean.abs() * 0.01));
    noise.set_fractal_gain(Some((0.4 + summary.taps[6].abs() * 0.2).min(0.95)));
    vec![FastNoiseChange::UpsertNode(NodeSpec::Sample3D(
        Sample3DNode {
            id: VOXEL_BASE_NODE.to_owned(),
            source: PositionSource::Base,
            noise: Arc::new(noise),
        },
    ))]
}

fn sample_noise(
    seed: i32,
    noise_type: NoiseType,
    fractal_type: FractalType,
    frequency: f32,
    octaves: i32,
) -> FastNoiseLite {
    let mut noise = FastNoiseLite::with_seed(seed);
    noise.set_noise_type(Some(noise_type));
    noise.set_frequency(Some(frequency));
    noise.set_fractal_type(Some(fractal_type));
    noise.set_fractal_octaves(Some(octaves));
    noise.set_fractal_lacunarity(Some(2.0));
    noise.set_fractal_gain(Some(0.5));
    noise.set_fractal_weighted_strength(Some(0.15));
    noise
}

fn terrain_detail_noise(seed: i32) -> Arc<FastNoiseLite> {
    let mut noise = sample_noise(seed, NoiseType::ValueCubic, FractalType::FBm, 0.022, 3);
    noise.set_fractal_gain(Some(0.45));
    Arc::new(noise)
}

fn warp_noise(seed: i32, frequency: f32, amplitude: f32) -> Arc<FastNoiseLite> {
    let mut noise = FastNoiseLite::with_seed(seed);
    noise.set_frequency(Some(frequency));
    noise.set_domain_warp_type(Some(DomainWarpType::OpenSimplex2));
    noise.set_domain_warp_amp(Some(amplitude));
    noise.set_fractal_type(Some(FractalType::DomainWarpProgressive));
    noise.set_fractal_octaves(Some(3));
    noise.set_fractal_gain(Some(0.5));
    Arc::new(noise)
}
