//! Direct-serial vs braid-parallel showcase for terrain chunk generation.

use braids::{BackendConfig, BraidExecutor, Stack};
use braids_fastnoise::{
    ChunkQuery, ChunkSummary, FastNoiseGraphSpec, FastNoiseLite, FastNoisePlanner, NodeSpec,
    make_cpu_backend, scenarios,
};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lanes = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(4)
        .clamp(2, 8);
    let chunk_count = lanes * 4;
    let queries = (0..chunk_count)
        .map(|seed| terrain_query(seed as u32))
        .collect::<Vec<_>>();

    let recipe = terrain_recipe();
    let mut scratch = Vec::new();
    let direct_start = Instant::now();
    let mut direct_checksum = 0u64;
    for query in &queries {
        let summary = render_terrain_direct(&recipe, query, &mut scratch);
        direct_checksum = direct_checksum.wrapping_add(summary_digest(&summary));
        black_box(summary);
    }
    let direct_elapsed = direct_start.elapsed();

    let executor = Arc::new(BraidExecutor::new(lanes));
    let backend = executor.register_backend(
        Arc::new(make_cpu_backend()),
        BackendConfig { lane_count: lanes },
    );
    let stack = Stack::create(
        Arc::clone(&executor),
        Arc::new(FastNoisePlanner),
        backend,
        scenarios::terrain_height_2d(),
    )?;

    let braid_start = Instant::now();
    let mut jobs = Vec::with_capacity(queries.len());
    for query in &queries {
        jobs.push(stack.dispatch(vec![query.clone()])?);
    }
    let mut braid_checksum = 0u64;
    for job in jobs {
        let mut summaries = stack.collect(job)?;
        let summary = summaries.remove(0);
        braid_checksum = braid_checksum.wrapping_add(summary_digest(&summary));
        black_box(summary);
    }
    let braid_elapsed = braid_start.elapsed();

    if direct_checksum != braid_checksum {
        return Err("checksum mismatch between direct and braid runs".into());
    }

    println!(
        "terrain lanes showcase: chunks={} lanes={} direct_serial_ms={:.3} braid_parallel_ms={:.3} speedup_x={:.2} checksum={}",
        chunk_count,
        lanes,
        direct_elapsed.as_secs_f64() * 1000.0,
        braid_elapsed.as_secs_f64() * 1000.0,
        direct_elapsed.as_secs_f64() / braid_elapsed.as_secs_f64(),
        direct_checksum
    );

    Ok(())
}

#[derive(Clone)]
struct TerrainRecipe {
    warp: Arc<FastNoiseLite>,
    continent: Arc<FastNoiseLite>,
    erosion: Arc<FastNoiseLite>,
    peaks: Arc<FastNoiseLite>,
    detail: Arc<FastNoiseLite>,
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

fn terrain_query(seed: u32) -> ChunkQuery {
    ChunkQuery::Grid2D {
        width: 128,
        height: 128,
        origin: [seed as f32 * 37.0, seed as f32 * 19.0],
        step: [1.0, 1.0],
    }
}

fn render_terrain_direct(
    recipe: &TerrainRecipe,
    query: &ChunkQuery,
    values: &mut Vec<f32>,
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
    if values.len() < len {
        values.resize(len, 0.0);
    }

    let mut index = 0usize;
    for y in 0..*height {
        for x in 0..*width {
            let px = origin[0] + x as f32 * step[0];
            let py = origin[1] + y as f32 * step[1];
            let (wx, wy) = recipe.warp.domain_warp_2d(px, py);
            let continent = recipe.continent.get_noise_2d(wx, wy);
            let erosion = recipe.erosion.get_noise_2d(wx, wy);
            let peaks = recipe.peaks.get_noise_2d(wx, wy);
            let detail = recipe.detail.get_noise_2d(wx, wy);
            let mut value = continent;
            value += peaks;
            value -= erosion;
            value += detail;
            values[index] = value;
            index += 1;
        }
    }

    summarize_values(&values[..len])
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
