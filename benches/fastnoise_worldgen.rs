//! Real FastNoise worldgen benchmark suite for serial-overhead and gameplay-feel measurements.

use braids::{
    BackendConfig, BackendHandle, BatchScratch, BraidExecutor, BraidResult, BufferSlot, CancelFlag,
    CompiledPlan, ComputeBackend, ComputeScratch, JobPacket, PipelineShape, PlannerBackend,
    PlannerScratch, Stack,
};
use braids_fastnoise::{
    ChunkQuery, ChunkSummary, FastNoiseChange, FastNoiseCpuBackend, FastNoiseGraphSpec,
    FastNoiseLite, FastNoisePlanner, NodeSpec, make_cpu_backend, scenarios,
};
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

const NULL_QUERY_SLOT: BufferSlot = BufferSlot(0);

fn main() -> BraidResult<()> {
    let config = BenchConfig::from_args();
    println!(
        "iterations={} serial=true workers=1 lanes=1 terrain=128x128 voxel=32x64x32",
        config.iterations
    );

    let reports = vec![
        bench_terrain_direct_serial(&config)?,
        bench_terrain_pipeline_serial(&config)?,
        bench_terrain_braid_serial(&config)?,
        bench_terrain_encode_only_serial(&config)?,
        bench_terrain_compute_only_serial(&config)?,
        bench_terrain_decode_only_serial(&config)?,
        bench_empty_roundtrip_serial(&config)?,
        bench_voxel_direct_serial(&config)?,
        bench_voxel_braid_serial(&config)?,
        bench_mixed_direct_serial(&config)?,
        bench_mixed_braid_serial(&config)?,
        bench_terrain_update_direct_serial(&config)?,
        bench_terrain_update_braid_serial(&config)?,
        bench_dependency_chain_direct_serial(&config)?,
        bench_dependency_chain_braid_serial(&config)?,
    ];

    for report in &reports {
        print_report(report);
    }

    print_comparison(
        &reports,
        "terrain_serial_overhead",
        "terrain_direct_serial",
        "terrain_braid_serial",
    );
    print_comparison(
        &reports,
        "terrain_stack_vs_pipeline",
        "terrain_pipeline_serial",
        "terrain_braid_serial",
    );
    print_comparison(
        &reports,
        "voxel_serial_overhead",
        "voxel_direct_serial",
        "voxel_braid_serial",
    );
    print_comparison(
        &reports,
        "mixed_serial_overhead",
        "mixed_direct_serial",
        "mixed_braid_serial",
    );
    print_comparison(
        &reports,
        "terrain_update_serial_overhead",
        "terrain_update_direct_serial",
        "terrain_update_braid_serial",
    );
    print_comparison(
        &reports,
        "dependency_serial_overhead",
        "dependency_chain_direct_serial",
        "dependency_chain_braid_serial",
    );

    Ok(())
}

#[derive(Clone, Copy)]
struct BenchConfig {
    iterations: usize,
}

impl BenchConfig {
    fn from_args() -> Self {
        let mut args = std::env::args().skip(1);
        Self {
            iterations: args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(120)
                .max(1),
        }
    }
}

struct BenchReport {
    name: &'static str,
    elapsed: Duration,
    iterations: usize,
    jobs: usize,
    queries: usize,
    checksum: u64,
    frame_stats: FrameStats,
}

#[derive(Clone, Copy)]
struct FrameStats {
    mean: Duration,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    max: Duration,
    over_8ms: usize,
    over_16ms: usize,
    over_33ms: usize,
}

fn print_report(report: &BenchReport) {
    println!(
        "{:32} total={:?} ns/iter={:.2} ns/job={:.2} ns/query={:.2} checksum={} frame_ms(mean/p50/p95/p99/max)={:.3}/{:.3}/{:.3}/{:.3}/{:.3} over8={} over16={} over33={}",
        report.name,
        report.elapsed,
        ns_per(report.elapsed, report.iterations),
        ns_per(report.elapsed, report.jobs),
        ns_per(report.elapsed, report.queries),
        report.checksum,
        duration_ms(report.frame_stats.mean),
        duration_ms(report.frame_stats.p50),
        duration_ms(report.frame_stats.p95),
        duration_ms(report.frame_stats.p99),
        duration_ms(report.frame_stats.max),
        report.frame_stats.over_8ms,
        report.frame_stats.over_16ms,
        report.frame_stats.over_33ms,
    );
}

fn print_comparison(
    reports: &[BenchReport],
    label: &'static str,
    baseline_name: &'static str,
    measured_name: &'static str,
) {
    let Some(baseline) = reports.iter().find(|report| report.name == baseline_name) else {
        return;
    };
    let Some(measured) = reports.iter().find(|report| report.name == measured_name) else {
        return;
    };
    println!(
        "{:32} query_x={:.3} frame_p95_x={:.3} frame_max_x={:.3} over16={}/{}",
        label,
        ratio(
            ns_per(measured.elapsed, measured.queries),
            ns_per(baseline.elapsed, baseline.queries),
        ),
        duration_ratio(measured.frame_stats.p95, baseline.frame_stats.p95),
        duration_ratio(measured.frame_stats.max, baseline.frame_stats.max),
        measured.frame_stats.over_16ms,
        baseline.frame_stats.over_16ms,
    );
}

fn ns_per(elapsed: Duration, units: usize) -> f64 {
    if units == 0 {
        return 0.0;
    }
    elapsed.as_nanos() as f64 / units as f64
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn duration_ratio(lhs: Duration, rhs: Duration) -> f64 {
    ratio(lhs.as_nanos() as f64, rhs.as_nanos() as f64)
}

fn ratio(lhs: f64, rhs: f64) -> f64 {
    if rhs == 0.0 {
        return 0.0;
    }
    lhs / rhs
}

fn summarize_frames(frames: &[Duration]) -> FrameStats {
    if frames.is_empty() {
        return FrameStats {
            mean: Duration::ZERO,
            p50: Duration::ZERO,
            p95: Duration::ZERO,
            p99: Duration::ZERO,
            max: Duration::ZERO,
            over_8ms: 0,
            over_16ms: 0,
            over_33ms: 0,
        };
    }

    let mut nanos = Vec::with_capacity(frames.len());
    let mut total = 0u128;
    let mut over_8ms = 0usize;
    let mut over_16ms = 0usize;
    let mut over_33ms = 0usize;
    for frame in frames {
        let frame_nanos = frame.as_nanos();
        total += frame_nanos;
        nanos.push(frame_nanos);
        if *frame > Duration::from_millis(8) {
            over_8ms += 1;
        }
        if *frame > Duration::from_millis(16) {
            over_16ms += 1;
        }
        if *frame > Duration::from_millis(33) {
            over_33ms += 1;
        }
    }
    nanos.sort_unstable();

    FrameStats {
        mean: nanos_to_duration(total / frames.len() as u128),
        p50: percentile_duration(nanos.as_slice(), 50),
        p95: percentile_duration(nanos.as_slice(), 95),
        p99: percentile_duration(nanos.as_slice(), 99),
        max: nanos_to_duration(*nanos.last().unwrap_or(&0)),
        over_8ms,
        over_16ms,
        over_33ms,
    }
}

fn percentile_duration(sorted_nanos: &[u128], percentile: usize) -> Duration {
    if sorted_nanos.is_empty() {
        return Duration::ZERO;
    }
    let last = sorted_nanos.len() - 1;
    let index = (last * percentile).div_ceil(100);
    nanos_to_duration(sorted_nanos[index])
}

fn nanos_to_duration(nanos: u128) -> Duration {
    let nanos = nanos.min(u64::MAX as u128) as u64;
    Duration::from_nanos(nanos)
}

fn build_report(
    name: &'static str,
    elapsed: Duration,
    iterations: usize,
    jobs: usize,
    queries: usize,
    checksum: u64,
    frame_times: &[Duration],
) -> BenchReport {
    BenchReport {
        name,
        elapsed,
        iterations,
        jobs,
        queries,
        checksum,
        frame_stats: summarize_frames(frame_times),
    }
}

type NoiseStack = Stack<FastNoisePlanner, FastNoiseCpuBackend>;
type NullStack = Stack<NullPlanner, NullBackend>;

struct SerialRuntime {
    executor: Arc<BraidExecutor>,
    planner: Arc<FastNoisePlanner>,
    backend: BackendHandle<FastNoiseCpuBackend>,
}

#[derive(Default)]
struct NullPlanner;

struct NullBackend;

#[derive(Default)]
struct DirectScratch {
    base_x: Vec<f32>,
    base_y: Vec<f32>,
    base_z: Vec<f32>,
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    pos_z: Vec<f32>,
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

#[derive(Clone)]
struct BiomeRecipe {
    warp: Arc<FastNoiseLite>,
    moisture: Arc<FastNoiseLite>,
    temperature: Arc<FastNoiseLite>,
}

#[derive(Clone)]
struct VoxelRecipe {
    warp: Arc<FastNoiseLite>,
    base: Arc<FastNoiseLite>,
    cave: Arc<FastNoiseLite>,
}

fn bench_terrain_direct_serial(config: &BenchConfig) -> BraidResult<BenchReport> {
    let recipe = terrain_recipe();
    let mut scratch = DirectScratch::default();
    black_box(render_terrain_direct(
        &recipe,
        &terrain_query(0),
        &mut scratch,
    ));

    let start = Instant::now();
    let mut frame_times = Vec::with_capacity(config.iterations);
    let mut checksum = 0u64;
    for iteration in 0..config.iterations {
        let frame_start = Instant::now();
        let summary =
            render_terrain_direct(&recipe, &terrain_query(iteration as u32), &mut scratch);
        checksum = checksum.wrapping_add(summary_digest(&summary));
        black_box(summary);
        frame_times.push(frame_start.elapsed());
    }

    Ok(build_report(
        "terrain_direct_serial",
        start.elapsed(),
        config.iterations,
        config.iterations,
        config.iterations,
        checksum,
        frame_times.as_slice(),
    ))
}

fn bench_terrain_braid_serial(config: &BenchConfig) -> BraidResult<BenchReport> {
    let runtime = make_serial_runtime();
    let stack = make_stack(
        &runtime.executor,
        &runtime.planner,
        &runtime.backend,
        scenarios::terrain_height_2d(),
    )?;
    warm_one(&stack, terrain_query(0))?;

    let start = Instant::now();
    let mut frame_times = Vec::with_capacity(config.iterations);
    let mut checksum = 0u64;
    for iteration in 0..config.iterations {
        let frame_start = Instant::now();
        let summary = warm_one(&stack, terrain_query(iteration as u32))?;
        checksum = checksum.wrapping_add(summary_digest(&summary));
        frame_times.push(frame_start.elapsed());
    }

    Ok(build_report(
        "terrain_braid_serial",
        start.elapsed(),
        config.iterations,
        config.iterations,
        config.iterations,
        checksum,
        frame_times.as_slice(),
    ))
}

fn bench_terrain_pipeline_serial(config: &BenchConfig) -> BraidResult<BenchReport> {
    let planner = FastNoisePlanner;
    let state = planner.init_state(&scenarios::terrain_height_2d())?;
    let mut planner_scratch = PlannerScratch::default();
    let plan = planner.compile(&state, &mut planner_scratch)?;
    let backend = make_cpu_backend();
    let mut compute_scratch = ComputeScratch::default();
    let prepared = backend.prepare(&plan, None, &mut compute_scratch)?;
    let cancel = CancelFlag::default();
    let mut batch_scratch = BatchScratch::default();
    let mut packet = JobPacket::default();

    warm_terrain_pipeline_once(
        &planner,
        &backend,
        &plan,
        &prepared,
        &cancel,
        &mut batch_scratch,
        &mut packet,
    )?;

    let start = Instant::now();
    let mut frame_times = Vec::with_capacity(config.iterations);
    let mut checksum = 0u64;
    for iteration in 0..config.iterations {
        let frame_start = Instant::now();
        packet.clear_for_reuse();
        let query = terrain_query(iteration as u32);
        planner.encode_batch(
            &plan,
            std::slice::from_ref(&query),
            &mut packet,
            &mut batch_scratch,
        )?;
        for (stage_index, stage) in plan.pipeline.stages.iter().enumerate() {
            backend.run_stage(&prepared, stage_index, stage, &mut packet, &cancel)?;
        }
        let mut summaries = planner.decode_batch(&plan, &packet)?;
        let summary = summaries.remove(0);
        checksum = checksum.wrapping_add(summary_digest(&summary));
        frame_times.push(frame_start.elapsed());
    }

    Ok(build_report(
        "terrain_pipeline_serial",
        start.elapsed(),
        config.iterations,
        config.iterations,
        config.iterations,
        checksum,
        frame_times.as_slice(),
    ))
}

fn bench_terrain_encode_only_serial(config: &BenchConfig) -> BraidResult<BenchReport> {
    let planner = FastNoisePlanner;
    let state = planner.init_state(&scenarios::terrain_height_2d())?;
    let mut planner_scratch = PlannerScratch::default();
    let plan = planner.compile(&state, &mut planner_scratch)?;
    let mut batch_scratch = BatchScratch::default();
    let mut packet = JobPacket::default();

    let start = Instant::now();
    let mut frame_times = Vec::with_capacity(config.iterations);
    let mut checksum = 0u64;
    for iteration in 0..config.iterations {
        let frame_start = Instant::now();
        packet.clear_for_reuse();
        let query = terrain_query(iteration as u32);
        planner.encode_batch(
            &plan,
            std::slice::from_ref(&query),
            &mut packet,
            &mut batch_scratch,
        )?;
        checksum = checksum.wrapping_add(packet.query_count as u64);
        black_box(&packet);
        frame_times.push(frame_start.elapsed());
    }

    Ok(build_report(
        "terrain_encode_only_serial",
        start.elapsed(),
        config.iterations,
        config.iterations,
        config.iterations,
        checksum,
        frame_times.as_slice(),
    ))
}

fn bench_terrain_compute_only_serial(config: &BenchConfig) -> BraidResult<BenchReport> {
    let planner = FastNoisePlanner;
    let state = planner.init_state(&scenarios::terrain_height_2d())?;
    let mut planner_scratch = PlannerScratch::default();
    let plan = planner.compile(&state, &mut planner_scratch)?;
    let backend = make_cpu_backend();
    let mut compute_scratch = ComputeScratch::default();
    let prepared = backend.prepare(&plan, None, &mut compute_scratch)?;
    let cancel = CancelFlag::default();
    let mut batch_scratch = BatchScratch::default();
    let mut packet = JobPacket::default();
    let query = terrain_query(0);
    planner.encode_batch(
        &plan,
        std::slice::from_ref(&query),
        &mut packet,
        &mut batch_scratch,
    )?;

    let start = Instant::now();
    let mut frame_times = Vec::with_capacity(config.iterations);
    let mut checksum = 0u64;
    for _ in 0..config.iterations {
        let frame_start = Instant::now();
        for (stage_index, stage) in plan.pipeline.stages.iter().enumerate() {
            backend.run_stage(&prepared, stage_index, stage, &mut packet, &cancel)?;
        }
        checksum = checksum.wrapping_add(packet.query_count as u64);
        black_box(&packet);
        frame_times.push(frame_start.elapsed());
    }

    Ok(build_report(
        "terrain_compute_only_serial",
        start.elapsed(),
        config.iterations,
        config.iterations,
        config.iterations,
        checksum,
        frame_times.as_slice(),
    ))
}

fn bench_terrain_decode_only_serial(config: &BenchConfig) -> BraidResult<BenchReport> {
    let planner = FastNoisePlanner;
    let state = planner.init_state(&scenarios::terrain_height_2d())?;
    let mut planner_scratch = PlannerScratch::default();
    let plan = planner.compile(&state, &mut planner_scratch)?;
    let backend = make_cpu_backend();
    let mut compute_scratch = ComputeScratch::default();
    let prepared = backend.prepare(&plan, None, &mut compute_scratch)?;
    let cancel = CancelFlag::default();
    let mut batch_scratch = BatchScratch::default();
    let mut packet = JobPacket::default();
    warm_terrain_pipeline_once(
        &planner,
        &backend,
        &plan,
        &prepared,
        &cancel,
        &mut batch_scratch,
        &mut packet,
    )?;

    let start = Instant::now();
    let mut frame_times = Vec::with_capacity(config.iterations);
    let mut checksum = 0u64;
    for _ in 0..config.iterations {
        let frame_start = Instant::now();
        let mut summaries = planner.decode_batch(&plan, &packet)?;
        let summary = summaries.remove(0);
        checksum = checksum.wrapping_add(summary_digest(&summary));
        black_box(summary);
        frame_times.push(frame_start.elapsed());
    }

    Ok(build_report(
        "terrain_decode_only_serial",
        start.elapsed(),
        config.iterations,
        config.iterations,
        config.iterations,
        checksum,
        frame_times.as_slice(),
    ))
}

fn bench_empty_roundtrip_serial(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(1));
    let planner = Arc::new(NullPlanner);
    let backend = executor.register_backend(Arc::new(NullBackend), BackendConfig { lane_count: 1 });
    let stack: NullStack = Stack::create(executor, planner, backend, ())?;

    let start = Instant::now();
    let mut frame_times = Vec::with_capacity(config.iterations);
    let mut checksum = 0u64;
    for iteration in 0..config.iterations {
        let frame_start = Instant::now();
        let job = stack.dispatch(vec![iteration as u32])?;
        let mut values = stack.collect(job)?;
        checksum = checksum.wrapping_add(values.remove(0) as u64);
        frame_times.push(frame_start.elapsed());
    }

    Ok(build_report(
        "empty_roundtrip_serial",
        start.elapsed(),
        config.iterations,
        config.iterations,
        config.iterations,
        checksum,
        frame_times.as_slice(),
    ))
}

fn bench_voxel_direct_serial(config: &BenchConfig) -> BraidResult<BenchReport> {
    let recipe = voxel_recipe();
    let mut scratch = DirectScratch::default();
    black_box(render_voxel_direct(&recipe, &voxel_query(0), &mut scratch));

    let start = Instant::now();
    let mut frame_times = Vec::with_capacity(config.iterations);
    let mut checksum = 0u64;
    for iteration in 0..config.iterations {
        let frame_start = Instant::now();
        let summary = render_voxel_direct(&recipe, &voxel_query(iteration as u32), &mut scratch);
        checksum = checksum.wrapping_add(summary_digest(&summary));
        black_box(summary);
        frame_times.push(frame_start.elapsed());
    }

    Ok(build_report(
        "voxel_direct_serial",
        start.elapsed(),
        config.iterations,
        config.iterations,
        config.iterations,
        checksum,
        frame_times.as_slice(),
    ))
}

fn bench_voxel_braid_serial(config: &BenchConfig) -> BraidResult<BenchReport> {
    let runtime = make_serial_runtime();
    let stack = make_stack(
        &runtime.executor,
        &runtime.planner,
        &runtime.backend,
        scenarios::voxel_density_3d(),
    )?;
    warm_one(&stack, voxel_query(0))?;

    let start = Instant::now();
    let mut frame_times = Vec::with_capacity(config.iterations);
    let mut checksum = 0u64;
    for iteration in 0..config.iterations {
        let frame_start = Instant::now();
        let summary = warm_one(&stack, voxel_query(iteration as u32))?;
        checksum = checksum.wrapping_add(summary_digest(&summary));
        frame_times.push(frame_start.elapsed());
    }

    Ok(build_report(
        "voxel_braid_serial",
        start.elapsed(),
        config.iterations,
        config.iterations,
        config.iterations,
        checksum,
        frame_times.as_slice(),
    ))
}

fn bench_mixed_direct_serial(config: &BenchConfig) -> BraidResult<BenchReport> {
    let terrain = terrain_recipe();
    let voxel = voxel_recipe();
    let mut scratch = DirectScratch::default();
    black_box(render_terrain_direct(
        &terrain,
        &terrain_query(0),
        &mut scratch,
    ));
    black_box(render_voxel_direct(&voxel, &voxel_query(0), &mut scratch));

    let start = Instant::now();
    let mut frame_times = Vec::with_capacity(config.iterations);
    let mut checksum = 0u64;
    for iteration in 0..config.iterations {
        let frame_start = Instant::now();
        let terrain_summary =
            render_terrain_direct(&terrain, &terrain_query(iteration as u32), &mut scratch);
        let voxel_summary =
            render_voxel_direct(&voxel, &voxel_query(iteration as u32), &mut scratch);
        checksum = checksum
            .wrapping_add(summary_digest(&terrain_summary))
            .wrapping_add(summary_digest(&voxel_summary));
        black_box((terrain_summary, voxel_summary));
        frame_times.push(frame_start.elapsed());
    }

    Ok(build_report(
        "mixed_direct_serial",
        start.elapsed(),
        config.iterations,
        config.iterations * 2,
        config.iterations * 2,
        checksum,
        frame_times.as_slice(),
    ))
}

fn bench_mixed_braid_serial(config: &BenchConfig) -> BraidResult<BenchReport> {
    let runtime = make_serial_runtime();
    let terrain = make_stack(
        &runtime.executor,
        &runtime.planner,
        &runtime.backend,
        scenarios::terrain_height_2d(),
    )?;
    let voxel = make_stack(
        &runtime.executor,
        &runtime.planner,
        &runtime.backend,
        scenarios::voxel_density_3d(),
    )?;
    warm_one(&terrain, terrain_query(0))?;
    warm_one(&voxel, voxel_query(0))?;

    let start = Instant::now();
    let mut frame_times = Vec::with_capacity(config.iterations);
    let mut checksum = 0u64;
    for iteration in 0..config.iterations {
        let frame_start = Instant::now();
        let terrain_summary = warm_one(&terrain, terrain_query(iteration as u32))?;
        let voxel_summary = warm_one(&voxel, voxel_query(iteration as u32))?;
        checksum = checksum
            .wrapping_add(summary_digest(&terrain_summary))
            .wrapping_add(summary_digest(&voxel_summary));
        frame_times.push(frame_start.elapsed());
    }

    Ok(build_report(
        "mixed_braid_serial",
        start.elapsed(),
        config.iterations,
        config.iterations * 2,
        config.iterations * 2,
        checksum,
        frame_times.as_slice(),
    ))
}

fn bench_terrain_update_direct_serial(config: &BenchConfig) -> BraidResult<BenchReport> {
    let mut recipe = terrain_recipe();
    let mut scratch = DirectScratch::default();
    let mut summary = render_terrain_direct(&recipe, &terrain_query(7), &mut scratch);

    let start = Instant::now();
    let mut frame_times = Vec::with_capacity(config.iterations);
    let mut checksum = 0u64;
    for iteration in 0..config.iterations {
        let frame_start = Instant::now();
        let patch = if iteration == 0 {
            terrain_patch_from_seed(iteration as u32)
        } else {
            scenarios::terrain_patch_from_biome(&summary)
        };
        checksum = checksum.wrapping_add(patch.len() as u64);
        apply_terrain_changes(&mut recipe, patch.as_slice());
        summary =
            render_terrain_direct(&recipe, &terrain_query(iteration as u32 + 19), &mut scratch);
        checksum = checksum.wrapping_add(summary_digest(&summary));
        black_box(&summary);
        frame_times.push(frame_start.elapsed());
    }

    Ok(build_report(
        "terrain_update_direct_serial",
        start.elapsed(),
        config.iterations,
        config.iterations,
        config.iterations,
        checksum,
        frame_times.as_slice(),
    ))
}

fn bench_terrain_update_braid_serial(config: &BenchConfig) -> BraidResult<BenchReport> {
    let runtime = make_serial_runtime();
    let stack = make_stack(
        &runtime.executor,
        &runtime.planner,
        &runtime.backend,
        scenarios::terrain_height_2d(),
    )?;
    let mut summary = warm_one(&stack, terrain_query(7))?;

    let start = Instant::now();
    let mut frame_times = Vec::with_capacity(config.iterations);
    let mut checksum = 0u64;
    for iteration in 0..config.iterations {
        let frame_start = Instant::now();
        let patch = if iteration == 0 {
            terrain_patch_from_seed(iteration as u32)
        } else {
            scenarios::terrain_patch_from_biome(&summary)
        };
        checksum = checksum.wrapping_add(patch.len() as u64);
        let version = stack.update(&patch)?;
        black_box(version);
        summary = warm_one(&stack, terrain_query(iteration as u32 + 19))?;
        checksum = checksum.wrapping_add(summary_digest(&summary));
        black_box(&summary);
        frame_times.push(frame_start.elapsed());
    }

    Ok(build_report(
        "terrain_update_braid_serial",
        start.elapsed(),
        config.iterations,
        config.iterations,
        config.iterations,
        checksum,
        frame_times.as_slice(),
    ))
}

fn bench_dependency_chain_direct_serial(config: &BenchConfig) -> BraidResult<BenchReport> {
    let biome = biome_recipe();
    let mut terrain = terrain_recipe();
    let mut voxel = voxel_recipe();
    let mut scratch = DirectScratch::default();

    let start = Instant::now();
    let mut frame_times = Vec::with_capacity(config.iterations);
    let mut checksum = 0u64;
    for iteration in 0..config.iterations {
        let frame_start = Instant::now();
        let biome_summary =
            render_biome_direct(&biome, &biome_query(iteration as u32), &mut scratch);
        apply_terrain_changes(
            &mut terrain,
            scenarios::terrain_patch_from_biome(&biome_summary).as_slice(),
        );
        let terrain_summary =
            render_terrain_direct(&terrain, &terrain_query(iteration as u32), &mut scratch);
        apply_voxel_changes(
            &mut voxel,
            scenarios::voxel_patch_from_terrain(&terrain_summary).as_slice(),
        );
        let voxel_summary =
            render_voxel_direct(&voxel, &voxel_query(iteration as u32), &mut scratch);
        checksum = checksum
            .wrapping_add(summary_digest(&biome_summary))
            .wrapping_add(summary_digest(&terrain_summary))
            .wrapping_add(summary_digest(&voxel_summary));
        frame_times.push(frame_start.elapsed());
    }

    Ok(build_report(
        "dependency_chain_direct_serial",
        start.elapsed(),
        config.iterations,
        config.iterations * 3,
        config.iterations * 3,
        checksum,
        frame_times.as_slice(),
    ))
}

fn bench_dependency_chain_braid_serial(config: &BenchConfig) -> BraidResult<BenchReport> {
    let runtime = make_serial_runtime();
    let biome = make_stack(
        &runtime.executor,
        &runtime.planner,
        &runtime.backend,
        scenarios::biome_control_2d(),
    )?;
    let terrain = make_stack(
        &runtime.executor,
        &runtime.planner,
        &runtime.backend,
        scenarios::terrain_height_2d(),
    )?;
    let voxel = make_stack(
        &runtime.executor,
        &runtime.planner,
        &runtime.backend,
        scenarios::voxel_density_3d(),
    )?;
    warm_one(&biome, biome_query(0))?;
    warm_one(&terrain, terrain_query(0))?;
    warm_one(&voxel, voxel_query(0))?;

    let start = Instant::now();
    let mut frame_times = Vec::with_capacity(config.iterations);
    let mut checksum = 0u64;
    for iteration in 0..config.iterations {
        let frame_start = Instant::now();
        let biome_summary = warm_one(&biome, biome_query(iteration as u32))?;
        terrain.update(&scenarios::terrain_patch_from_biome(&biome_summary))?;
        let terrain_summary = warm_one(&terrain, terrain_query(iteration as u32))?;
        voxel.update(&scenarios::voxel_patch_from_terrain(&terrain_summary))?;
        let voxel_summary = warm_one(&voxel, voxel_query(iteration as u32))?;
        checksum = checksum
            .wrapping_add(summary_digest(&biome_summary))
            .wrapping_add(summary_digest(&terrain_summary))
            .wrapping_add(summary_digest(&voxel_summary));
        frame_times.push(frame_start.elapsed());
    }

    Ok(build_report(
        "dependency_chain_braid_serial",
        start.elapsed(),
        config.iterations,
        config.iterations * 3,
        config.iterations * 3,
        checksum,
        frame_times.as_slice(),
    ))
}

fn make_serial_runtime() -> SerialRuntime {
    let executor = Arc::new(BraidExecutor::new(1));
    let planner = Arc::new(FastNoisePlanner);
    let backend = executor.register_backend(
        Arc::new(make_cpu_backend()),
        BackendConfig { lane_count: 1 },
    );
    SerialRuntime {
        executor,
        planner,
        backend,
    }
}

fn warm_terrain_pipeline_once(
    planner: &FastNoisePlanner,
    backend: &FastNoiseCpuBackend,
    plan: &CompiledPlan<<FastNoisePlanner as PlannerBackend>::PlannerMeta>,
    prepared: &<FastNoiseCpuBackend as ComputeBackend>::Prepared,
    cancel: &CancelFlag,
    batch_scratch: &mut BatchScratch,
    packet: &mut JobPacket,
) -> BraidResult<()> {
    packet.clear_for_reuse();
    let query = terrain_query(0);
    planner.encode_batch(plan, std::slice::from_ref(&query), packet, batch_scratch)?;
    for (stage_index, stage) in plan.pipeline.stages.iter().enumerate() {
        backend.run_stage(prepared, stage_index, stage, packet, cancel)?;
    }
    let mut summaries = planner.decode_batch(plan, packet)?;
    black_box(summaries.remove(0));
    Ok(())
}

fn make_stack(
    executor: &Arc<BraidExecutor>,
    planner: &Arc<FastNoisePlanner>,
    backend: &BackendHandle<FastNoiseCpuBackend>,
    spec: FastNoiseGraphSpec,
) -> BraidResult<NoiseStack> {
    Stack::create(
        Arc::clone(executor),
        Arc::clone(planner),
        backend.clone(),
        spec,
    )
}

fn terrain_query(seed: u32) -> ChunkQuery {
    ChunkQuery::Grid2D {
        width: 128,
        height: 128,
        origin: [seed as f32 * 37.0, seed as f32 * 19.0],
        step: [1.0, 1.0],
    }
}

fn biome_query(seed: u32) -> ChunkQuery {
    ChunkQuery::Grid2D {
        width: 128,
        height: 128,
        origin: [seed as f32 * 29.0, seed as f32 * 11.0],
        step: [1.0, 1.0],
    }
}

fn voxel_query(seed: u32) -> ChunkQuery {
    ChunkQuery::Grid3D {
        width: 32,
        height: 64,
        depth: 32,
        origin: [seed as f32 * 17.0, -32.0, seed as f32 * 13.0],
        step: [1.0, 1.0, 1.0],
    }
}

fn warm_one(stack: &NoiseStack, query: ChunkQuery) -> BraidResult<ChunkSummary> {
    let job = stack.dispatch(vec![query])?;
    let mut summaries = stack.collect(job)?;
    let summary = summaries.remove(0);
    black_box(&summary);
    Ok(summary)
}

fn terrain_patch_from_seed(seed: u32) -> Vec<FastNoiseChange> {
    let spec = scenarios::terrain_height_2d();
    let detail = spec
        .nodes
        .into_iter()
        .find_map(|node| match node {
            NodeSpec::Sample2D(sample) if sample.id == scenarios::TERRAIN_DETAIL_NODE => {
                Some(sample)
            }
            _ => None,
        })
        .expect("terrain detail node missing");

    let mut detail = detail;
    let source_noise = detail.noise.as_ref();
    let mut noise = FastNoiseLite::with_seed(seed as i32 + 1033);
    noise.set_frequency(Some(0.012 + (seed % 7) as f32 * 0.0025));
    noise.set_noise_type(Some(source_noise.noise_type));
    noise.set_rotation_type_3d(Some(source_noise.rotation_type_3d));
    noise.set_fractal_type(Some(source_noise.fractal_type));
    noise.set_fractal_octaves(Some(source_noise.octaves));
    noise.set_fractal_lacunarity(Some(source_noise.lacunarity));
    noise.set_fractal_gain(Some(source_noise.gain));
    noise.set_fractal_weighted_strength(Some(source_noise.weighted_strength));
    noise.set_fractal_ping_pong_strength(Some(source_noise.ping_pong_strength));
    noise.set_cellular_distance_function(Some(source_noise.cellular_distance_function));
    noise.set_cellular_return_type(Some(source_noise.cellular_return_type));
    noise.set_cellular_jitter(Some(source_noise.cellular_jitter_modifier));
    noise.set_domain_warp_type(Some(source_noise.domain_warp_type));
    noise.set_domain_warp_amp(Some(source_noise.domain_warp_amp));
    noise.set_fractal_gain(Some((0.35 + (seed % 5) as f32 * 0.08).min(0.95)));
    detail.noise = Arc::new(noise);
    vec![FastNoiseChange::UpsertNode(NodeSpec::Sample2D(detail))]
}

fn summary_digest(summary: &ChunkSummary) -> u64 {
    summary.checksum ^ ((summary.samples as u64) << 16) ^ u64::from(summary.mean.to_bits())
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

fn biome_recipe() -> BiomeRecipe {
    let spec = scenarios::biome_control_2d();
    BiomeRecipe {
        warp: find_warp2d_noise(&spec, scenarios::BIOME_WARP_NODE),
        moisture: find_sample2d_noise(&spec, scenarios::BIOME_MOISTURE_NODE),
        temperature: find_sample2d_noise(&spec, scenarios::BIOME_TEMPERATURE_NODE),
    }
}

fn voxel_recipe() -> VoxelRecipe {
    let spec = scenarios::voxel_density_3d();
    VoxelRecipe {
        warp: find_warp3d_noise(&spec, scenarios::VOXEL_WARP_NODE),
        base: find_sample3d_noise(&spec, scenarios::VOXEL_BASE_NODE),
        cave: find_sample3d_noise(&spec, scenarios::VOXEL_CAVE_NODE),
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

fn find_warp3d_noise(spec: &FastNoiseGraphSpec, id: &str) -> Arc<FastNoiseLite> {
    spec.nodes
        .iter()
        .find_map(|node| match node {
            NodeSpec::Warp3D(node) if node.id == id => Some(node.noise.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("warp3d node missing: {id}"))
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

fn find_sample3d_noise(spec: &FastNoiseGraphSpec, id: &str) -> Arc<FastNoiseLite> {
    spec.nodes
        .iter()
        .find_map(|node| match node {
            NodeSpec::Sample3D(node) if node.id == id => Some(node.noise.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("sample3d node missing: {id}"))
}

fn apply_terrain_changes(recipe: &mut TerrainRecipe, changes: &[FastNoiseChange]) {
    for change in changes {
        if let FastNoiseChange::UpsertNode(NodeSpec::Sample2D(node)) = change
            && node.id == scenarios::TERRAIN_DETAIL_NODE
        {
            recipe.detail = node.noise.clone();
        }
    }
}

fn apply_voxel_changes(recipe: &mut VoxelRecipe, changes: &[FastNoiseChange]) {
    for change in changes {
        if let FastNoiseChange::UpsertNode(NodeSpec::Sample3D(node)) = change
            && node.id == scenarios::VOXEL_BASE_NODE
        {
            recipe.base = node.noise.clone();
        }
    }
}

fn render_biome_direct(
    recipe: &BiomeRecipe,
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
        panic!("biome direct needs grid2d")
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
            .moisture
            .get_noise_2d(scratch.pos_x[index], scratch.pos_y[index]);
    }
    for index in 0..len {
        scratch.b[index] = recipe
            .temperature
            .get_noise_2d(scratch.pos_x[index], scratch.pos_y[index]);
    }
    for index in 0..len {
        scratch.a[index] += scratch.b[index];
    }

    summarize_values(&scratch.a[..len])
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

fn render_voxel_direct(
    recipe: &VoxelRecipe,
    query: &ChunkQuery,
    scratch: &mut DirectScratch,
) -> ChunkSummary {
    let ChunkQuery::Grid3D {
        width,
        height,
        depth,
        origin,
        step,
    } = query
    else {
        panic!("voxel direct needs grid3d")
    };

    let len = width * height * depth;
    resize_direct_3d(scratch, len);

    for z in 0..*depth {
        for y in 0..*height {
            for x in 0..*width {
                let index = ((z * height + y) * width) + x;
                scratch.base_x[index] = origin[0] + x as f32 * step[0];
                scratch.base_y[index] = origin[1] + y as f32 * step[1];
                scratch.base_z[index] = origin[2] + z as f32 * step[2];
            }
        }
    }
    for index in 0..len {
        let (wx, wy, wz) = recipe.warp.domain_warp_3d(
            scratch.base_x[index],
            scratch.base_y[index],
            scratch.base_z[index],
        );
        scratch.pos_x[index] = wx;
        scratch.pos_y[index] = wy;
        scratch.pos_z[index] = wz;
    }
    for index in 0..len {
        scratch.a[index] = recipe.base.get_noise_3d(
            scratch.base_x[index],
            scratch.base_y[index],
            scratch.base_z[index],
        );
    }
    for index in 0..len {
        scratch.b[index] = recipe.cave.get_noise_3d(
            scratch.pos_x[index],
            scratch.pos_y[index],
            scratch.pos_z[index],
        );
    }
    for index in 0..len {
        scratch.a[index] -= scratch.b[index];
    }
    for index in 0..len {
        scratch.a[index] = apply_ygradient(
            scratch.a[index],
            scratch.base_y[index],
            -32.0,
            32.0,
            1.0,
            -1.0,
        );
    }

    summarize_values(&scratch.a[..len])
}

fn apply_ygradient(input: f32, y: f32, min_y: f32, max_y: f32, low: f32, high: f32) -> f32 {
    let denom = max_y - min_y;
    let t = if denom.abs() <= f32::EPSILON {
        0.0
    } else {
        ((y - min_y) / denom).clamp(0.0, 1.0)
    };
    let gradient = low + ((high - low) * t);
    let mut value = input;
    value += gradient;
    value
}

fn resize_values(values: &mut Vec<f32>, len: usize) {
    if values.len() < len {
        values.resize(len, 0.0);
    }
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

fn resize_direct_3d(scratch: &mut DirectScratch, len: usize) {
    resize_direct_2d(scratch, len);
    resize_values(&mut scratch.base_z, len);
    resize_values(&mut scratch.pos_z, len);
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

impl PlannerBackend for NullPlanner {
    type Spec = ();
    type State = ();
    type Change = ();
    type Query = u32;
    type Resolution = u32;
    type PlannerMeta = ();

    fn init_state(&self, _spec: &Self::Spec) -> BraidResult<Self::State> {
        Ok(())
    }

    fn reset_state(&self, _state: &mut Self::State, _spec: &Self::Spec) -> BraidResult<()> {
        Ok(())
    }

    fn apply(&self, _state: &mut Self::State, _changes: &[Self::Change]) -> BraidResult<()> {
        Ok(())
    }

    fn updated_state(
        &self,
        _state: &Self::State,
        _changes: &[Self::Change],
    ) -> BraidResult<Self::State> {
        Ok(())
    }

    fn compile(
        &self,
        _state: &Self::State,
        _scratch: &mut PlannerScratch,
    ) -> BraidResult<CompiledPlan<Self::PlannerMeta>> {
        let plan = CompiledPlan {
            pipeline: PipelineShape {
                buffers: Vec::new(),
                stages: Vec::new(),
            },
            static_buffers: Vec::new(),
            planner_meta: (),
        };
        plan.validate()?;
        Ok(plan)
    }

    fn encode_batch(
        &self,
        _plan: &CompiledPlan<Self::PlannerMeta>,
        queries: &[Self::Query],
        packet: &mut JobPacket,
        _scratch: &mut BatchScratch,
    ) -> BraidResult<()> {
        packet.query_count = queries.len();
        let values = packet.ensure::<u32>(NULL_QUERY_SLOT, queries.len());
        values.copy_from_slice(queries);
        Ok(())
    }

    fn decode_batch(
        &self,
        _plan: &CompiledPlan<Self::PlannerMeta>,
        packet: &JobPacket,
    ) -> BraidResult<Vec<Self::Resolution>> {
        Ok(packet.slice::<u32>(NULL_QUERY_SLOT)?.to_vec())
    }
}

impl ComputeBackend for NullBackend {
    type Prepared = ();

    fn prepare<M: Send + Sync + 'static>(
        &self,
        _plan: &CompiledPlan<M>,
        _reuse: Option<Self::Prepared>,
        _scratch: &mut ComputeScratch,
    ) -> BraidResult<Self::Prepared> {
        Ok(())
    }

    fn run_stage(
        &self,
        _prepared: &Self::Prepared,
        _stage_index: usize,
        _stage: &braids::StageSpec,
        _packet: &mut JobPacket,
        _cancel: &CancelFlag,
    ) -> BraidResult<()> {
        Ok(())
    }
}
