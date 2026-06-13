//! Criterion benchmarks for fair FastNoise worldgen stack comparisons.

use braids::{BackendConfig, BraidExecutor, InlineContext, Stack};
use braids_fastnoise::{
    ChunkQuery, ChunkSummary, FastNoiseCpuBackend, FastNoiseGraphSpec, FastNoiseLite,
    FastNoisePlanner, NodeSpec, make_cpu_backend, scenarios,
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

type NoiseStack = Stack<FastNoisePlanner, FastNoiseCpuBackend>;

#[derive(Default)]
struct DirectScratch {
    base_x: Vec<f32>,
    base_y: Vec<f32>,
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    d: Vec<f32>,
}

#[derive(Clone)]
struct TerrainRecipe {
    warp: Arc<FastNoiseLite>,
    continent: Arc<FastNoiseLite>,
    erosion: Arc<FastNoiseLite>,
    peaks: Arc<FastNoiseLite>,
    detail: Arc<FastNoiseLite>,
}

fn bench_fastnoise_worldgen(c: &mut Criterion) {
    let lanes = available_lanes();
    let chunk_count = lanes * 4;
    let queries = make_terrain_queries(chunk_count);
    verify_checksums(queries.as_slice(), lanes);

    let mut fair_group = c.benchmark_group("fastnoise_worldgen_fair");
    fair_group.measurement_time(Duration::from_secs(2));
    fair_group.throughput(Throughput::Elements(chunk_count as u64));

    fair_group.bench_function(
        BenchmarkId::new("terrain_inline_serial", format!("chunks_{chunk_count}")),
        |b| {
            let stack = make_noise_stack(1, 1);
            let mut inline = InlineContext::default();
            b.iter(|| {
                let checksum = must(
                    run_inline_batch(&stack, queries.as_slice(), &mut inline),
                    "terrain inline serial",
                );
                black_box(checksum);
            });
        },
    );

    fair_group.bench_function(
        BenchmarkId::new("terrain_async_serial", format!("chunks_{chunk_count}")),
        |b| {
            let stack = make_noise_stack(1, 1);
            b.iter(|| {
                let checksum = must(
                    run_async_batch(&stack, queries.as_slice()),
                    "terrain async serial",
                );
                black_box(checksum);
            });
        },
    );

    fair_group.bench_function(
        BenchmarkId::new(
            "terrain_async_parallel",
            format!("chunks_{chunk_count}_lanes_{lanes}"),
        ),
        |b| {
            let stack = make_noise_stack(lanes, lanes);
            b.iter(|| {
                let checksum = must(
                    run_async_batch(&stack, queries.as_slice()),
                    "terrain async parallel",
                );
                black_box(checksum);
            });
        },
    );

    fair_group.finish();

    let mut showcase_group = c.benchmark_group("fastnoise_worldgen_showcase");
    showcase_group.measurement_time(Duration::from_secs(2));
    showcase_group.throughput(Throughput::Elements(chunk_count as u64));

    showcase_group.bench_function(
        BenchmarkId::new("terrain_direct_fastnoise", format!("chunks_{chunk_count}")),
        |b| {
            let recipe = terrain_recipe();
            let mut scratch = DirectScratch::default();
            b.iter(|| {
                let checksum = run_direct_batch(&recipe, queries.as_slice(), &mut scratch);
                black_box(checksum);
            });
        },
    );

    showcase_group.bench_function(
        BenchmarkId::new(
            "terrain_braid_parallel_showcase",
            format!("chunks_{chunk_count}_lanes_{lanes}"),
        ),
        |b| {
            let stack = make_noise_stack(lanes, lanes);
            b.iter(|| {
                let checksum = must(
                    run_async_batch(&stack, queries.as_slice()),
                    "terrain braids parallel showcase",
                );
                black_box(checksum);
            });
        },
    );

    showcase_group.finish();
}

criterion_group!(fastnoise_worldgen_benches, bench_fastnoise_worldgen);
criterion_main!(fastnoise_worldgen_benches);

fn available_lanes() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(4)
        .clamp(2, 8)
}

fn make_terrain_queries(chunk_count: usize) -> Vec<ChunkQuery> {
    (0..chunk_count as u32).map(terrain_query).collect()
}

fn make_noise_stack(worker_count: usize, lane_count: usize) -> NoiseStack {
    let executor = Arc::new(BraidExecutor::new(worker_count));
    let backend =
        executor.register_backend(Arc::new(make_cpu_backend()), BackendConfig { lane_count });
    must(
        Stack::create(
            executor,
            Arc::new(FastNoisePlanner),
            backend,
            scenarios::terrain_height_2d(),
        ),
        "terrain stack setup",
    )
}

fn verify_checksums(queries: &[ChunkQuery], lanes: usize) {
    let recipe = terrain_recipe();
    let mut scratch = DirectScratch::default();
    let direct = run_direct_batch(&recipe, queries, &mut scratch);

    let inline_stack = make_noise_stack(1, 1);
    let mut inline = InlineContext::default();
    let inline_checksum = must(
        run_inline_batch(&inline_stack, queries, &mut inline),
        "terrain inline checksum",
    );

    let async_serial_stack = make_noise_stack(1, 1);
    let async_serial = must(
        run_async_batch(&async_serial_stack, queries),
        "terrain async serial checksum",
    );

    let async_parallel_stack = make_noise_stack(lanes, lanes);
    let async_parallel = must(
        run_async_batch(&async_parallel_stack, queries),
        "terrain async parallel checksum",
    );

    assert_eq!(direct, inline_checksum);
    assert_eq!(direct, async_serial);
    assert_eq!(direct, async_parallel);
}

fn run_direct_batch(
    recipe: &TerrainRecipe,
    queries: &[ChunkQuery],
    scratch: &mut DirectScratch,
) -> u64 {
    let mut checksum = 0u64;
    for query in queries {
        let summary = render_terrain_direct(recipe, query, scratch);
        checksum = checksum.wrapping_add(summary_digest(&summary));
        black_box(summary);
    }
    checksum
}

fn run_inline_batch(
    stack: &NoiseStack,
    queries: &[ChunkQuery],
    inline: &mut InlineContext,
) -> Result<u64, braids::BraidError> {
    let mut checksum = 0u64;
    for query in queries {
        let summary = stack.resolve_one_inline_ref_with(query, inline)?;
        checksum = checksum.wrapping_add(summary_digest(&summary));
        black_box(summary);
    }
    Ok(checksum)
}

fn run_async_batch(stack: &NoiseStack, queries: &[ChunkQuery]) -> Result<u64, braids::BraidError> {
    let mut jobs = Vec::with_capacity(queries.len());
    for query in queries {
        jobs.push(stack.dispatch(vec![query.clone()])?);
    }

    let mut checksum = 0u64;
    for job in jobs {
        let mut summaries = stack.collect(job)?;
        let summary = summaries.remove(0);
        checksum = checksum.wrapping_add(summary_digest(&summary));
        black_box(summary);
    }
    Ok(checksum)
}

fn terrain_query(seed: u32) -> ChunkQuery {
    ChunkQuery::Grid2D {
        width: 128,
        height: 128,
        origin: [seed as f32 * 37.0, seed as f32 * 19.0],
        step: [1.0, 1.0],
    }
}

fn terrain_recipe() -> TerrainRecipe {
    let spec = scenarios::terrain_height_2d();
    TerrainRecipe {
        warp: find_warp2d_noise(&spec, scenarios::TERRAIN_WARP_NODE),
        continent: find_sample2d_noise(&spec, scenarios::TERRAIN_CONTINENT_NODE),
        erosion: find_sample2d_noise(&spec, scenarios::TERRAIN_EROSION_NODE),
        peaks: find_sample2d_noise(&spec, scenarios::TERRAIN_PEAKS_NODE),
        detail: find_sample2d_noise(&spec, scenarios::TERRAIN_DETAIL_NODE),
    }
}

fn find_warp2d_noise(spec: &FastNoiseGraphSpec, id: &str) -> Arc<FastNoiseLite> {
    spec.nodes
        .iter()
        .find_map(|node| match node {
            NodeSpec::Warp2D(node) if node.id == id => Some(node.noise.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("warp2d node missing: {id}"))
}

fn find_sample2d_noise(spec: &FastNoiseGraphSpec, id: &str) -> Arc<FastNoiseLite> {
    spec.nodes
        .iter()
        .find_map(|node| match node {
            NodeSpec::Sample2D(node) if node.id == id => Some(node.noise.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("sample2d node missing: {id}"))
}

fn render_terrain_direct(
    recipe: &TerrainRecipe,
    query: &ChunkQuery,
    scratch: &mut DirectScratch,
) -> ChunkSummary {
    let ChunkQuery::Grid2D {
        width,
        height,
        origin,
        step,
    } = query
    else {
        panic!("terrain direct needs grid2d")
    };

    let len = width * height;
    resize_direct_2d(scratch, len);

    for y in 0..*height {
        for x in 0..*width {
            let index = (y * width) + x;
            scratch.base_x[index] = origin[0] + x as f32 * step[0];
            scratch.base_y[index] = origin[1] + y as f32 * step[1];
        }
    }
    for index in 0..len {
        let (wx, wy) = recipe
            .warp
            .domain_warp_2d(scratch.base_x[index], scratch.base_y[index]);
        scratch.pos_x[index] = wx;
        scratch.pos_y[index] = wy;
    }
    for index in 0..len {
        scratch.a[index] = recipe
            .continent
            .get_noise_2d(scratch.pos_x[index], scratch.pos_y[index]);
    }
    for index in 0..len {
        scratch.b[index] = recipe
            .erosion
            .get_noise_2d(scratch.pos_x[index], scratch.pos_y[index]);
    }
    for index in 0..len {
        scratch.c[index] = recipe
            .peaks
            .get_noise_2d(scratch.pos_x[index], scratch.pos_y[index]);
    }
    for index in 0..len {
        scratch.d[index] = recipe
            .detail
            .get_noise_2d(scratch.pos_x[index], scratch.pos_y[index]);
    }
    for index in 0..len {
        scratch.a[index] += scratch.c[index];
    }
    for index in 0..len {
        scratch.a[index] -= scratch.b[index];
    }
    for index in 0..len {
        scratch.a[index] += scratch.d[index];
    }

    summarize_values(&scratch.a[..len])
}

fn resize_direct_2d(scratch: &mut DirectScratch, len: usize) {
    resize_values(&mut scratch.base_x, len);
    resize_values(&mut scratch.base_y, len);
    resize_values(&mut scratch.pos_x, len);
    resize_values(&mut scratch.pos_y, len);
    resize_values(&mut scratch.a, len);
    resize_values(&mut scratch.b, len);
    resize_values(&mut scratch.c, len);
    resize_values(&mut scratch.d, len);
}

fn resize_values(values: &mut Vec<f32>, len: usize) {
    if values.len() < len {
        values.resize(len, 0.0);
    }
}

fn summarize_values(values: &[f32]) -> ChunkSummary {
    if values.is_empty() {
        return ChunkSummary {
            samples: 0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            checksum: 0,
            taps: [0.0; 8],
        };
    }

    let mut min = values[0];
    let mut max = values[0];
    let mut sum = 0.0f64;
    let mut checksum = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        min = min.min(*value);
        max = max.max(*value);
        sum += f64::from(*value);
        checksum ^= u64::from(value.to_bits()) + 0x9e37_79b9;
        checksum = checksum.wrapping_mul(0x1000_0000_01b3);
    }

    let mut taps = [0.0; 8];
    let tap_count = taps.len();
    for (index, tap) in taps.iter_mut().enumerate() {
        let sample_index = if values.len() == 1 {
            0
        } else {
            index * (values.len() - 1) / (tap_count - 1)
        };
        *tap = values[sample_index];
    }

    ChunkSummary {
        samples: values.len(),
        min,
        max,
        mean: (sum / values.len() as f64) as f32,
        checksum,
        taps,
    }
}

fn summary_digest(summary: &ChunkSummary) -> u64 {
    summary.checksum ^ ((summary.samples as u64) << 16) ^ u64::from(summary.mean.to_bits())
}

fn must<T, E: std::fmt::Display>(result: Result<T, E>, context: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("{context}: {error}"),
    }
}
