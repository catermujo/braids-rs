use braids::{BackendConfig, BraidExecutor, InlineContext, Stack};
use braids_fastnoise::{
    ChunkQuery, ChunkSummary, FastNoiseCpuBackend, FastNoiseGraphSpec, FastNoiseLite,
    FastNoisePlanner, NodeSpec, make_cpu_backend, scenarios,
};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

type NoiseStack = Stack<FastNoisePlanner, FastNoiseCpuBackend>;

#[derive(Clone)]
struct TerrainRecipe {
    warp: Arc<FastNoiseLite>,
    continent: Arc<FastNoiseLite>,
    erosion: Arc<FastNoiseLite>,
    peaks: Arc<FastNoiseLite>,
    detail: Arc<FastNoiseLite>,
}

#[derive(Clone, Copy)]
enum Mode {
    Direct,
    Inline,
    Serial,
    Parallel,
}

impl Mode {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "direct" => Some(Self::Direct),
            "inline" => Some(Self::Inline),
            "serial" => Some(Self::Serial),
            "parallel" => Some(Self::Parallel),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Inline => "inline",
            Self::Serial => "serial",
            Self::Parallel => "parallel",
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mode = args
        .next()
        .as_deref()
        .and_then(Mode::parse)
        .unwrap_or(Mode::Parallel);
    let passes = args
        .next()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(60)
        .max(1);

    let lanes = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(4)
        .clamp(2, 8);
    let chunk_count = lanes * 4;
    let queries = make_queries(chunk_count);

    let start = Instant::now();
    let checksum = match mode {
        Mode::Direct => run_direct(queries.as_slice(), passes),
        Mode::Inline => run_inline(queries.as_slice(), passes)?,
        Mode::Serial => run_async(queries.as_slice(), passes, 1)?,
        Mode::Parallel => run_async(queries.as_slice(), passes, lanes)?,
    };
    let elapsed = start.elapsed();

    println!(
        "profile_terrain mode={} passes={} chunks={} lanes={} elapsed_ms={:.3} checksum={}",
        mode.as_str(),
        passes,
        chunk_count,
        lanes,
        elapsed.as_secs_f64() * 1000.0,
        checksum
    );

    Ok(())
}

fn make_queries(chunk_count: usize) -> Vec<ChunkQuery> {
    (0..chunk_count as u32).map(terrain_query).collect()
}

fn run_direct(queries: &[ChunkQuery], passes: usize) -> u64 {
    let recipe = terrain_recipe();
    let mut values = Vec::new();
    let mut checksum = 0u64;
    for _ in 0..passes {
        for query in queries {
            let summary = render_terrain_direct(&recipe, query, &mut values);
            checksum = checksum.wrapping_add(summary_digest(&summary));
            black_box(summary);
        }
    }
    checksum
}

fn run_inline(queries: &[ChunkQuery], passes: usize) -> Result<u64, braids::BraidError> {
    let stack = make_stack(1, 1)?;
    let mut inline = InlineContext::default();
    let mut checksum = 0u64;
    for _ in 0..passes {
        for query in queries {
            let summary = stack.resolve_one_inline_ref_with(query, &mut inline)?;
            checksum = checksum.wrapping_add(summary_digest(&summary));
            black_box(summary);
        }
    }
    Ok(checksum)
}

fn run_async(
    queries: &[ChunkQuery],
    passes: usize,
    lanes: usize,
) -> Result<u64, braids::BraidError> {
    let stack = make_stack(lanes, lanes)?;
    let mut checksum = 0u64;
    for _ in 0..passes {
        let mut jobs = Vec::with_capacity(queries.len());
        for query in queries {
            jobs.push(stack.dispatch(vec![query.clone()])?);
        }
        for job in jobs {
            let mut summaries = stack.collect(job)?;
            let summary = summaries.remove(0);
            checksum = checksum.wrapping_add(summary_digest(&summary));
            black_box(summary);
        }
    }
    Ok(checksum)
}

fn make_stack(worker_count: usize, lane_count: usize) -> Result<NoiseStack, braids::BraidError> {
    let executor = Arc::new(BraidExecutor::new(worker_count));
    let backend =
        executor.register_backend(Arc::new(make_cpu_backend()), BackendConfig { lane_count });
    Stack::create(
        executor,
        Arc::new(FastNoisePlanner),
        backend,
        scenarios::terrain_height_2d(),
    )
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
