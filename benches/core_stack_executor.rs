//! Core stack/executor benchmark suite.

use braid::{
    BackendConfig, BatchScratch, BraidError, BraidExecutor, BraidResult, BufferBinding, BufferSlot,
    BufferSpec, CancelFlag, CompiledPlan, ComputeBackend, ComputeScratch, ElementKind,
    InlineContext, JobPacket, KernelKind, KernelSpec, PlannerBackend, PlannerScratch, Stack,
    StageSpec,
};
use std::hint::black_box;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const DATA_SLOT: BufferSlot = BufferSlot(0);
const STAGE_SCAN_KIND: KernelKind = KernelKind(0xB100);
const STAGE_FINISH_KIND: KernelKind = KernelKind(0xB200);
const COMPILE_HEAVY_ROUNDS: u32 = 60_000;
const PREPARE_HEAVY_ROUNDS: u32 = 50_000;
const SERIAL_ONE_QUERY_SEED: usize = 0xA500;
const SERIAL_ONE_QUERY_BASE: u32 = 0xA500_A500;

fn main() -> BraidResult<()> {
    let config = BenchConfig::from_args();
    println!(
        "iterations={} batch_size={} stacks={} queue_depth={} dependency_depth={} workers={} lanes={} stage1_rounds={} stage2_rounds={}",
        config.iterations,
        config.batch_size,
        config.stack_count,
        config.queue_depth,
        config.dependency_depth,
        config.workers,
        config.lane_count,
        config.stage1_rounds,
        config.stage2_rounds
    );

    let reports = vec![
        bench_serial_async_one_query(&config)?,
        bench_serial_inline_slice_one_query(&config)?,
        bench_serial_inline_one_query(&config)?,
        bench_parallel_stacks(&config)?,
        bench_serialized_backend_stacks(&config)?,
        bench_hidden_serialized_backend_stacks(&config)?,
        bench_mixed_backend_isolation_declared(&config)?,
        bench_mixed_backend_isolation_hidden(&config)?,
        bench_compile_parallel_plain(&config)?,
        bench_compile_parallel_hidden(&config)?,
        bench_mixed_compile_runtime(&config)?,
        bench_mixed_planner_gate_runtime(&config)?,
        bench_prepare_parallel_plain(&config)?,
        bench_prepare_parallel_hidden(&config)?,
        bench_prepare_parallel_limited(&config)?,
        bench_mixed_prepare_runtime_split(&config)?,
        bench_mixed_prepare_runtime_same(&config)?,
        bench_mixed_prepare_runtime_limited(&config)?,
        bench_queue_pressure(&config)?,
        bench_update_between_runs(&config)?,
        bench_dependency_chain(&config)?,
        bench_version_swap_inflight(&config)?,
        bench_create_stack_and_run(&config)?,
    ];

    for report in reports {
        print_report(&report);
    }

    Ok(())
}

#[derive(Clone, Copy)]
struct BenchConfig {
    iterations: usize,
    batch_size: usize,
    stack_count: usize,
    queue_depth: usize,
    dependency_depth: usize,
    workers: usize,
    lane_count: usize,
    stage1_rounds: u32,
    stage2_rounds: u32,
}

impl BenchConfig {
    fn from_args() -> Self {
        let mut args = std::env::args().skip(1);
        let available = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(4);
        let workers = available.clamp(2, 8);
        Self {
            iterations: args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(100)
                .max(1),
            batch_size: args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(256)
                .max(1),
            stack_count: args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8)
                .max(2),
            queue_depth: args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(16)
                .max(2),
            dependency_depth: args
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or(4)
                .max(2),
            workers,
            lane_count: 256,
            stage1_rounds: 48,
            stage2_rounds: 24,
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
    aux_label: Option<&'static str>,
    aux_elapsed: Duration,
    aux_units: usize,
}

fn print_report(report: &BenchReport) {
    let mut line = format!(
        "{:28} total={:?} ns/iter={:.2} ns/job={:.2} ns/query={:.2} checksum={}",
        report.name,
        report.elapsed,
        ns_per(report.elapsed, report.iterations),
        ns_per(report.elapsed, report.jobs),
        ns_per(report.elapsed, report.queries),
        report.checksum
    );
    if let Some(label) = report.aux_label {
        line.push_str(
            format!(
                " {}={:.2}",
                label,
                ns_per(report.aux_elapsed, report.aux_units)
            )
            .as_str(),
        );
    }
    println!("{line}");
}

fn ns_per(elapsed: Duration, units: usize) -> f64 {
    if units == 0 {
        return 0.0;
    }
    elapsed.as_nanos() as f64 / units as f64
}

struct BenchPlanner {
    hidden_compile_serialize: bool,
    hidden_runtime_serialize: bool,
    compile_rounds: u32,
    compile_gate: Mutex<()>,
}

#[derive(Clone)]
struct BenchSpec {
    bias: u32,
    multiplier: u32,
    lane_values: Vec<u32>,
    stage1_rounds: u32,
    stage2_rounds: u32,
}

struct BenchState {
    bias: u32,
    multiplier: u32,
    lane_values: Vec<u32>,
    stage1_rounds: u32,
    stage2_rounds: u32,
}

enum BenchChange {
    SetBias(u32),
    SetMultiplier(u32),
    PatchLane { index: usize, value: u32 },
}

struct BenchBackend {
    hidden_serialize: bool,
    prepare_rounds: u32,
    hidden_gate: Mutex<()>,
}

#[derive(Default)]
struct BenchPrepared {
    bias: u32,
    multiplier: u32,
    stage1_rounds: u32,
    stage2_rounds: u32,
    lane_values: Vec<u32>,
}

impl PlannerBackend for BenchPlanner {
    type Spec = BenchSpec;
    type State = BenchState;
    type Change = BenchChange;
    type Query = u32;
    type Resolution = u32;
    type PlannerMeta = ();

    fn init_state(&self, spec: &Self::Spec) -> BraidResult<Self::State> {
        Ok(BenchState {
            bias: spec.bias,
            multiplier: spec.multiplier,
            lane_values: spec.lane_values.clone(),
            stage1_rounds: spec.stage1_rounds,
            stage2_rounds: spec.stage2_rounds,
        })
    }

    fn reset_state(&self, state: &mut Self::State, spec: &Self::Spec) -> BraidResult<()> {
        state.bias = spec.bias;
        state.multiplier = spec.multiplier;
        state.stage1_rounds = spec.stage1_rounds;
        state.stage2_rounds = spec.stage2_rounds;
        state.lane_values.clear();
        state
            .lane_values
            .extend_from_slice(spec.lane_values.as_slice());
        Ok(())
    }

    fn apply(&self, state: &mut Self::State, changes: &[Self::Change]) -> BraidResult<()> {
        for change in changes {
            match change {
                BenchChange::SetBias(value) => state.bias = *value,
                BenchChange::SetMultiplier(value) => state.multiplier = *value,
                BenchChange::PatchLane { index, value } => {
                    if state.lane_values.is_empty() {
                        continue;
                    }
                    let slot = *index % state.lane_values.len();
                    state.lane_values[slot] = *value;
                }
            }
        }
        Ok(())
    }

    fn updated_state(
        &self,
        state: &Self::State,
        changes: &[Self::Change],
    ) -> BraidResult<Self::State> {
        let mut next = BenchState {
            bias: state.bias,
            multiplier: state.multiplier,
            lane_values: state.lane_values.clone(),
            stage1_rounds: state.stage1_rounds,
            stage2_rounds: state.stage2_rounds,
        };
        self.apply(&mut next, changes)?;
        Ok(next)
    }

    fn compile(
        &self,
        state: &Self::State,
        scratch: &mut PlannerScratch,
    ) -> BraidResult<CompiledPlan<Self::PlannerMeta>> {
        let _guard = if self.hidden_compile_serialize {
            Some(
                self.compile_gate
                    .lock()
                    .map_err(|_| BraidError::poisoned("bench_planner.compile_gate"))?,
            )
        } else {
            None
        };
        if self.compile_rounds > 0 {
            black_box(burn_compile(
                state.bias
                    ^ state.multiplier
                    ^ state.stage1_rounds
                    ^ state.stage2_rounds
                    ^ state.lane_values.len() as u32,
                self.compile_rounds,
            ));
        }
        let scan_payload = encode_scan_payload(state, scratch);
        let finish_payload = encode_finish_payload(state, scratch);

        let plan = CompiledPlan {
            pipeline: braid::PipelineShape {
                buffers: vec![BufferSpec {
                    slot: DATA_SLOT,
                    element_kind: ElementKind::U32,
                    layout: braid::BufferLayout::PerQueryScalar,
                }],
                stages: vec![
                    StageSpec {
                        kernels: vec![KernelSpec {
                            kind_id: STAGE_SCAN_KIND,
                            payload: scan_payload,
                            bindings: vec![BufferBinding {
                                slot: DATA_SLOT,
                                access: braid::BufferAccess::ReadWrite,
                            }],
                            dispatch: braid::DispatchHint::WholeBatch,
                        }],
                    },
                    StageSpec {
                        kernels: vec![KernelSpec {
                            kind_id: STAGE_FINISH_KIND,
                            payload: finish_payload,
                            bindings: vec![BufferBinding {
                                slot: DATA_SLOT,
                                access: braid::BufferAccess::ReadWrite,
                            }],
                            dispatch: braid::DispatchHint::WholeBatch,
                        }],
                    },
                ],
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
        let _guard = if self.hidden_runtime_serialize {
            Some(
                self.compile_gate
                    .lock()
                    .map_err(|_| BraidError::poisoned("bench_planner.compile_gate"))?,
            )
        } else {
            None
        };
        packet.query_count = queries.len();
        packet
            .ensure::<u32>(DATA_SLOT, queries.len())
            .copy_from_slice(queries);
        Ok(())
    }

    fn decode_batch(
        &self,
        _plan: &CompiledPlan<Self::PlannerMeta>,
        packet: &JobPacket,
    ) -> BraidResult<Vec<Self::Resolution>> {
        let _guard = if self.hidden_runtime_serialize {
            Some(
                self.compile_gate
                    .lock()
                    .map_err(|_| BraidError::poisoned("bench_planner.compile_gate"))?,
            )
        } else {
            None
        };
        Ok(packet.slice::<u32>(DATA_SLOT)?.to_vec())
    }
}

impl BenchPlanner {
    fn plain() -> Self {
        Self {
            hidden_compile_serialize: false,
            hidden_runtime_serialize: false,
            compile_rounds: 0,
            compile_gate: Mutex::new(()),
        }
    }

    fn compile_heavy() -> Self {
        Self {
            hidden_compile_serialize: false,
            hidden_runtime_serialize: false,
            compile_rounds: COMPILE_HEAVY_ROUNDS,
            compile_gate: Mutex::new(()),
        }
    }

    fn hidden_compile_heavy() -> Self {
        Self {
            hidden_compile_serialize: true,
            hidden_runtime_serialize: false,
            compile_rounds: COMPILE_HEAVY_ROUNDS,
            compile_gate: Mutex::new(()),
        }
    }

    fn hidden_compile_runtime_heavy() -> Self {
        Self {
            hidden_compile_serialize: true,
            hidden_runtime_serialize: true,
            compile_rounds: COMPILE_HEAVY_ROUNDS,
            compile_gate: Mutex::new(()),
        }
    }
}

impl ComputeBackend for BenchBackend {
    type Prepared = BenchPrepared;

    fn prepare<M: Send + Sync + 'static>(
        &self,
        plan: &CompiledPlan<M>,
        reuse: Option<Self::Prepared>,
        scratch: &mut ComputeScratch,
    ) -> BraidResult<Self::Prepared> {
        let _guard = if self.hidden_serialize {
            Some(
                self.hidden_gate
                    .lock()
                    .map_err(|_| BraidError::poisoned("bench_backend.hidden_gate"))?,
            )
        } else {
            None
        };
        scratch.reset();
        let mut prepared = reuse.unwrap_or_default();

        let scan_payload = &plan.pipeline.stages[0].kernels[0].payload;
        prepared.bias = read_u32(scan_payload, 0);
        prepared.stage1_rounds = read_u32(scan_payload, 4);
        let lane_count = read_u32(scan_payload, 8) as usize;
        prepared.lane_values.clear();
        prepared.lane_values.reserve(lane_count);
        for index in 0..lane_count {
            let start = 12 + index * 4;
            prepared.lane_values.push(read_u32(scan_payload, start));
        }

        let finish_payload = &plan.pipeline.stages[1].kernels[0].payload;
        prepared.multiplier = read_u32(finish_payload, 0);
        prepared.stage2_rounds = read_u32(finish_payload, 4);

        if self.prepare_rounds > 0 {
            let spin = burn_prepare(
                prepared.bias
                    ^ prepared.multiplier
                    ^ prepared.stage1_rounds
                    ^ prepared.stage2_rounds
                    ^ lane_count as u32,
                self.prepare_rounds,
            );
            scratch.u32s.push(spin);
            black_box(scratch.u32s[0]);
        }

        Ok(prepared)
    }

    fn run_stage(
        &self,
        prepared: &Self::Prepared,
        stage_index: usize,
        _stage: &StageSpec,
        packet: &mut JobPacket,
        cancel: &CancelFlag,
    ) -> BraidResult<()> {
        let _guard = if self.hidden_serialize {
            Some(
                self.hidden_gate
                    .lock()
                    .map_err(|_| BraidError::poisoned("bench_backend.hidden_gate"))?,
            )
        } else {
            None
        };
        if cancel.is_cancelled() {
            return Err(BraidError::Cancelled);
        }

        let values = packet.slice_mut::<u32>(DATA_SLOT)?;
        match stage_index {
            0 => run_scan_stage(prepared, values, cancel),
            1 => run_finish_stage(prepared, values, cancel),
            _ => Err(BraidError::from("unexpected stage index")),
        }
    }
}

impl BenchBackend {
    fn plain() -> Self {
        Self {
            hidden_serialize: false,
            prepare_rounds: 0,
            hidden_gate: Mutex::new(()),
        }
    }

    fn hidden_serialized() -> Self {
        Self {
            hidden_serialize: true,
            prepare_rounds: 0,
            hidden_gate: Mutex::new(()),
        }
    }

    fn plain_prepare_heavy() -> Self {
        Self {
            hidden_serialize: false,
            prepare_rounds: PREPARE_HEAVY_ROUNDS,
            hidden_gate: Mutex::new(()),
        }
    }

    fn hidden_prepare_heavy() -> Self {
        Self {
            hidden_serialize: true,
            prepare_rounds: PREPARE_HEAVY_ROUNDS,
            hidden_gate: Mutex::new(()),
        }
    }
}

fn encode_scan_payload(state: &BenchState, scratch: &mut PlannerScratch) -> Arc<[u8]> {
    scratch.reset();
    push_u32(&mut scratch.bytes, state.bias);
    push_u32(&mut scratch.bytes, state.stage1_rounds);
    push_u32(&mut scratch.bytes, state.lane_values.len() as u32);
    for value in &state.lane_values {
        push_u32(&mut scratch.bytes, *value);
    }
    Arc::from(scratch.bytes.clone())
}

fn encode_finish_payload(state: &BenchState, scratch: &mut PlannerScratch) -> Arc<[u8]> {
    scratch.reset();
    push_u32(&mut scratch.bytes, state.multiplier);
    push_u32(&mut scratch.bytes, state.stage2_rounds);
    Arc::from(scratch.bytes.clone())
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn run_scan_stage(
    prepared: &BenchPrepared,
    values: &mut [u32],
    cancel: &CancelFlag,
) -> BraidResult<()> {
    let lane_count = prepared.lane_values.len().max(1);
    for (index, value) in values.iter_mut().enumerate() {
        let lane = prepared.lane_values[index % lane_count];
        let mut acc = value.wrapping_add(prepared.bias ^ lane);
        for _ in 0..prepared.stage1_rounds {
            acc = mix32(acc ^ lane.rotate_left(7));
        }
        *value = acc;

        if index & 63 == 0 && cancel.is_cancelled() {
            return Err(BraidError::Cancelled);
        }
    }
    Ok(())
}

fn run_finish_stage(
    prepared: &BenchPrepared,
    values: &mut [u32],
    cancel: &CancelFlag,
) -> BraidResult<()> {
    for (index, value) in values.iter_mut().enumerate() {
        let mut acc = value.wrapping_mul(prepared.multiplier | 1);
        for round in 0..prepared.stage2_rounds {
            acc = mix32(acc ^ round.wrapping_mul(0x9E37_79B9));
        }
        *value = acc;

        if index & 63 == 0 && cancel.is_cancelled() {
            return Err(BraidError::Cancelled);
        }
    }
    Ok(())
}

fn mix32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^ (value >> 16)
}

fn burn_compile(seed: u32, rounds: u32) -> u32 {
    let mut acc = seed;
    for round in 0..rounds {
        acc = mix32(acc ^ round.rotate_left(9));
    }
    acc
}

fn burn_prepare(seed: u32, rounds: u32) -> u32 {
    let mut acc = seed;
    for round in 0..rounds {
        acc = mix32(acc ^ round.wrapping_mul(0x9E37_79B9));
    }
    acc
}

fn bench_parallel_stacks(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers));
    let planner = Arc::new(BenchPlanner::plain());
    let backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig {
            lane_count: config.workers,
        },
    );
    let stacks = make_stack_group(&executor, &planner, &backend, config, config.stack_count)?;

    warm_parallel_stacks(&stacks, config.batch_size)?;

    let start = Instant::now();
    let mut checksum = 0u64;
    for iter in 0..config.iterations {
        let mut jobs = Vec::with_capacity(stacks.len());
        for (stack_index, stack) in stacks.iter().enumerate() {
            let queries = make_queries(
                ((iter * stacks.len()) + stack_index) as u32,
                config.batch_size,
            );
            jobs.push((stack_index, stack.dispatch(queries)?));
        }

        for (stack_index, job) in jobs {
            let values = stacks[stack_index].collect(job)?;
            checksum = checksum.wrapping_add(digest(values.as_slice()));
            black_box(&values);
        }
    }

    Ok(BenchReport {
        name: "parallel_stacks",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: config.iterations * stacks.len(),
        queries: config.iterations * stacks.len() * config.batch_size,
        checksum,
        aux_label: None,
        aux_elapsed: Duration::ZERO,
        aux_units: 0,
    })
}

fn bench_serial_async_one_query(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(1));
    let planner = Arc::new(BenchPlanner::plain());
    let backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig { lane_count: 1 },
    );
    let stack = make_stack(&executor, &planner, &backend, config, SERIAL_ONE_QUERY_SEED)?;

    warm_inline_and_async(&stack)?;

    let start = Instant::now();
    let mut checksum = 0u64;
    for iter in 0..config.iterations {
        let query = [mix32(iter as u32 ^ SERIAL_ONE_QUERY_BASE)];
        let job = stack.dispatch(query.to_vec())?;
        let values = stack.collect(job)?;
        checksum = checksum.wrapping_add(digest(values.as_slice()));
        black_box(&values);
    }

    Ok(BenchReport {
        name: "serial_async_one_query",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: config.iterations,
        queries: config.iterations,
        checksum,
        aux_label: None,
        aux_elapsed: Duration::ZERO,
        aux_units: 0,
    })
}

fn bench_serial_inline_slice_one_query(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(1));
    let planner = Arc::new(BenchPlanner::plain());
    let backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig { lane_count: 1 },
    );
    let stack = make_stack(&executor, &planner, &backend, config, SERIAL_ONE_QUERY_SEED)?;

    warm_inline_and_async(&stack)?;

    let start = Instant::now();
    let mut checksum = 0u64;
    let mut inline = InlineContext::default();
    for iter in 0..config.iterations {
        let query = [mix32(iter as u32 ^ SERIAL_ONE_QUERY_BASE)];
        let values = stack.resolve_inline_with(&query, &mut inline)?;
        checksum = checksum.wrapping_add(digest(values.as_slice()));
        black_box(&values);
    }

    Ok(BenchReport {
        name: "serial_inline_slice_one_query",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: config.iterations,
        queries: config.iterations,
        checksum,
        aux_label: None,
        aux_elapsed: Duration::ZERO,
        aux_units: 0,
    })
}

fn bench_serial_inline_one_query(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(1));
    let planner = Arc::new(BenchPlanner::plain());
    let backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig { lane_count: 1 },
    );
    let stack = make_stack(&executor, &planner, &backend, config, SERIAL_ONE_QUERY_SEED)?;

    warm_inline_and_async(&stack)?;

    let start = Instant::now();
    let mut checksum = 0u64;
    let mut inline = InlineContext::default();
    for iter in 0..config.iterations {
        let value = stack
            .resolve_one_inline_with(mix32(iter as u32 ^ SERIAL_ONE_QUERY_BASE), &mut inline)?;
        checksum = checksum.wrapping_add(digest(&[value]));
        black_box(value);
    }

    Ok(BenchReport {
        name: "serial_inline_one_query",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: config.iterations,
        queries: config.iterations,
        checksum,
        aux_label: None,
        aux_elapsed: Duration::ZERO,
        aux_units: 0,
    })
}

fn bench_serialized_backend_stacks(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers));
    let planner = Arc::new(BenchPlanner::plain());
    let backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig { lane_count: 1 },
    );
    let stacks = make_stack_group(&executor, &planner, &backend, config, config.stack_count)?;

    warm_parallel_stacks(&stacks, config.batch_size)?;

    let start = Instant::now();
    let mut checksum = 0u64;
    for iter in 0..config.iterations {
        let mut jobs = Vec::with_capacity(stacks.len());
        for (stack_index, stack) in stacks.iter().enumerate() {
            let queries = make_queries(
                ((iter * stacks.len()) + stack_index) as u32,
                config.batch_size,
            );
            jobs.push((stack_index, stack.dispatch(queries)?));
        }

        for (stack_index, job) in jobs {
            let values = stacks[stack_index].collect(job)?;
            checksum = checksum.wrapping_add(digest(values.as_slice()));
            black_box(&values);
        }
    }

    Ok(BenchReport {
        name: "serialized_declared",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: config.iterations * stacks.len(),
        queries: config.iterations * stacks.len() * config.batch_size,
        checksum,
        aux_label: None,
        aux_elapsed: Duration::ZERO,
        aux_units: 0,
    })
}

fn bench_hidden_serialized_backend_stacks(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers));
    let planner = Arc::new(BenchPlanner::plain());
    let backend = executor.register_backend(
        Arc::new(BenchBackend::hidden_serialized()),
        BackendConfig {
            lane_count: config.workers,
        },
    );
    let stacks = make_stack_group(&executor, &planner, &backend, config, config.stack_count)?;

    warm_parallel_stacks(&stacks, config.batch_size)?;

    let start = Instant::now();
    let mut checksum = 0u64;
    for iter in 0..config.iterations {
        let mut jobs = Vec::with_capacity(stacks.len());
        for (stack_index, stack) in stacks.iter().enumerate() {
            let queries = make_queries(
                ((iter * stacks.len()) + stack_index) as u32,
                config.batch_size,
            );
            jobs.push((stack_index, stack.dispatch(queries)?));
        }

        for (stack_index, job) in jobs {
            let values = stacks[stack_index].collect(job)?;
            checksum = checksum.wrapping_add(digest(values.as_slice()));
            black_box(&values);
        }
    }

    Ok(BenchReport {
        name: "serialized_hidden",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: config.iterations * stacks.len(),
        queries: config.iterations * stacks.len() * config.batch_size,
        checksum,
        aux_label: None,
        aux_elapsed: Duration::ZERO,
        aux_units: 0,
    })
}

fn bench_mixed_backend_isolation_declared(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers.max(2)));
    let planner = Arc::new(BenchPlanner::plain());
    let slow_batch_size = config.batch_size * 4;
    let fast_batch_size = (config.batch_size / 8).max(16);
    let slow_backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig { lane_count: 1 },
    );
    let fast_backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig {
            lane_count: config.workers.max(2),
        },
    );

    let slow_stacks = make_stack_group(
        &executor,
        &planner,
        &slow_backend,
        config,
        config.stack_count.max(4),
    )?;
    let fast_stack = make_stack(&executor, &planner, &fast_backend, config, 9_999)?;

    warm_parallel_stacks(&slow_stacks, slow_batch_size)?;
    warm_queue_pressure(&fast_stack, fast_batch_size)?;

    let start = Instant::now();
    let mut fast_elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for iter in 0..config.iterations {
        let mut slow_jobs = Vec::with_capacity(slow_stacks.len());
        for (stack_index, stack) in slow_stacks.iter().enumerate() {
            slow_jobs.push((
                stack_index,
                stack.dispatch(make_queries(
                    ((iter * slow_stacks.len()) + stack_index) as u32,
                    slow_batch_size,
                ))?,
            ));
        }

        let fast_start = Instant::now();
        let fast_job =
            fast_stack.dispatch(make_queries((iter as u32) ^ 0xFA57, fast_batch_size))?;
        let fast_values = fast_stack.collect(fast_job)?;
        fast_elapsed += fast_start.elapsed();
        checksum = checksum.wrapping_add(digest(fast_values.as_slice()));
        black_box(&fast_values);

        for (stack_index, job) in slow_jobs {
            let slow_values = slow_stacks[stack_index].collect(job)?;
            checksum = checksum.wrapping_add(digest(slow_values.as_slice()));
            black_box(&slow_values);
        }
    }

    Ok(BenchReport {
        name: "mixed_isolation_declared",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: config.iterations * (slow_stacks.len() + 1),
        queries: config.iterations * ((slow_stacks.len() * slow_batch_size) + fast_batch_size),
        checksum,
        aux_label: Some("fast_ns/job"),
        aux_elapsed: fast_elapsed,
        aux_units: config.iterations,
    })
}

fn bench_mixed_backend_isolation_hidden(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers.max(2)));
    let planner = Arc::new(BenchPlanner::plain());
    let slow_batch_size = config.batch_size * 4;
    let fast_batch_size = (config.batch_size / 8).max(16);
    let slow_backend = executor.register_backend(
        Arc::new(BenchBackend::hidden_serialized()),
        BackendConfig {
            lane_count: config.workers.max(2),
        },
    );
    let fast_backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig {
            lane_count: config.workers.max(2),
        },
    );

    let slow_stacks = make_stack_group(
        &executor,
        &planner,
        &slow_backend,
        config,
        config.stack_count.max(4),
    )?;
    let fast_stack = make_stack(&executor, &planner, &fast_backend, config, 10_001)?;

    warm_parallel_stacks(&slow_stacks, slow_batch_size)?;
    warm_queue_pressure(&fast_stack, fast_batch_size)?;

    let start = Instant::now();
    let mut fast_elapsed = Duration::ZERO;
    let mut checksum = 0u64;
    for iter in 0..config.iterations {
        let mut slow_jobs = Vec::with_capacity(slow_stacks.len());
        for (stack_index, stack) in slow_stacks.iter().enumerate() {
            slow_jobs.push((
                stack_index,
                stack.dispatch(make_queries(
                    ((iter * slow_stacks.len()) + stack_index) as u32,
                    slow_batch_size,
                ))?,
            ));
        }

        let fast_start = Instant::now();
        let fast_job =
            fast_stack.dispatch(make_queries((iter as u32) ^ 0xFA57, fast_batch_size))?;
        let fast_values = fast_stack.collect(fast_job)?;
        fast_elapsed += fast_start.elapsed();
        checksum = checksum.wrapping_add(digest(fast_values.as_slice()));
        black_box(&fast_values);

        for (stack_index, job) in slow_jobs {
            let slow_values = slow_stacks[stack_index].collect(job)?;
            checksum = checksum.wrapping_add(digest(slow_values.as_slice()));
            black_box(&slow_values);
        }
    }

    Ok(BenchReport {
        name: "mixed_isolation_hidden",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: config.iterations * (slow_stacks.len() + 1),
        queries: config.iterations * ((slow_stacks.len() * slow_batch_size) + fast_batch_size),
        checksum,
        aux_label: Some("fast_ns/job"),
        aux_elapsed: fast_elapsed,
        aux_units: config.iterations,
    })
}

fn bench_compile_parallel_plain(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers.max(2)));
    let planner = Arc::new(BenchPlanner::compile_heavy());
    let backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig {
            lane_count: config.workers.max(2),
        },
    );
    let stacks = make_stack_group(
        &executor,
        &planner,
        &backend,
        config,
        config.stack_count.max(config.workers),
    )?;

    warm_parallel_updates(&stacks, 1, config)?;

    let start = Instant::now();
    let (checksum, updates) = run_parallel_updates(&stacks, config.iterations, config)?;

    Ok(BenchReport {
        name: "compile_parallel_plain",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: updates,
        queries: updates,
        checksum,
        aux_label: None,
        aux_elapsed: Duration::ZERO,
        aux_units: 0,
    })
}

fn bench_compile_parallel_hidden(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers.max(2)));
    let planner = Arc::new(BenchPlanner::hidden_compile_heavy());
    let backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig {
            lane_count: config.workers.max(2),
        },
    );
    let stacks = make_stack_group(
        &executor,
        &planner,
        &backend,
        config,
        config.stack_count.max(config.workers),
    )?;

    warm_parallel_updates(&stacks, 1, config)?;

    let start = Instant::now();
    let (checksum, updates) = run_parallel_updates(&stacks, config.iterations, config)?;

    Ok(BenchReport {
        name: "compile_parallel_hidden",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: updates,
        queries: updates,
        checksum,
        aux_label: None,
        aux_elapsed: Duration::ZERO,
        aux_units: 0,
    })
}

fn bench_mixed_compile_runtime(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers.max(2)));
    let compile_planner = Arc::new(BenchPlanner::compile_heavy());
    let runtime_planner = Arc::new(BenchPlanner::plain());
    let compile_backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig {
            lane_count: config.workers.max(2),
        },
    );
    let runtime_backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig {
            lane_count: config.workers.max(2),
        },
    );
    let compile_stacks = make_stack_group(
        &executor,
        &compile_planner,
        &compile_backend,
        config,
        config.stack_count.max(config.workers),
    )?;
    let fast_stack = make_stack(
        &executor,
        &runtime_planner,
        &runtime_backend,
        config,
        20_101,
    )?;
    let fast_batch_size = (config.batch_size / 8).max(16);

    warm_parallel_updates(&compile_stacks, 1, config)?;
    warm_queue_pressure(&fast_stack, fast_batch_size)?;

    let start = Instant::now();
    let (fast_elapsed, checksum) = run_mixed_update_runtime(
        &compile_stacks,
        &fast_stack,
        config.iterations,
        config,
        fast_batch_size,
    )?;

    Ok(BenchReport {
        name: "mixed_compile_runtime",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: config.iterations * (compile_stacks.len() + 1),
        queries: config.iterations * fast_batch_size,
        checksum,
        aux_label: Some("fast_ns/job"),
        aux_elapsed: fast_elapsed,
        aux_units: config.iterations,
    })
}

fn bench_mixed_planner_gate_runtime(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers.max(2)));
    let planner = Arc::new(BenchPlanner::hidden_compile_runtime_heavy());
    let compile_backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig {
            lane_count: config.workers.max(2),
        },
    );
    let runtime_backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig {
            lane_count: config.workers.max(2),
        },
    );
    let compile_stacks = make_stack_group(
        &executor,
        &planner,
        &compile_backend,
        config,
        config.stack_count.max(config.workers),
    )?;
    let fast_stack = make_stack(&executor, &planner, &runtime_backend, config, 20_131)?;
    let fast_batch_size = (config.batch_size / 8).max(16);

    warm_parallel_updates(&compile_stacks, 1, config)?;
    warm_queue_pressure(&fast_stack, fast_batch_size)?;

    let start = Instant::now();
    let (fast_elapsed, checksum) = run_mixed_update_runtime(
        &compile_stacks,
        &fast_stack,
        config.iterations,
        config,
        fast_batch_size,
    )?;

    Ok(BenchReport {
        name: "mixed_planner_gate",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: config.iterations * (compile_stacks.len() + 1),
        queries: config.iterations * fast_batch_size,
        checksum,
        aux_label: Some("fast_ns/job"),
        aux_elapsed: fast_elapsed,
        aux_units: config.iterations,
    })
}

fn bench_prepare_parallel_plain(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers.max(2)));
    let planner = Arc::new(BenchPlanner::plain());
    let backend = executor.register_backend(
        Arc::new(BenchBackend::plain_prepare_heavy()),
        BackendConfig {
            lane_count: config.workers.max(2),
        },
    );
    let stacks = make_stack_group(
        &executor,
        &planner,
        &backend,
        config,
        config.stack_count.max(config.workers),
    )?;

    warm_parallel_updates(&stacks, 1, config)?;

    let start = Instant::now();
    let (checksum, updates) = run_parallel_updates(&stacks, config.iterations, config)?;

    Ok(BenchReport {
        name: "prepare_parallel_plain",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: updates,
        queries: updates,
        checksum,
        aux_label: None,
        aux_elapsed: Duration::ZERO,
        aux_units: 0,
    })
}

fn bench_prepare_parallel_hidden(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers.max(2)));
    let planner = Arc::new(BenchPlanner::plain());
    let backend = executor.register_backend(
        Arc::new(BenchBackend::hidden_prepare_heavy()),
        BackendConfig {
            lane_count: config.workers.max(2),
        },
    );
    let stacks = make_stack_group(
        &executor,
        &planner,
        &backend,
        config,
        config.stack_count.max(config.workers),
    )?;

    warm_parallel_updates(&stacks, 1, config)?;

    let start = Instant::now();
    let (checksum, updates) = run_parallel_updates(&stacks, config.iterations, config)?;

    Ok(BenchReport {
        name: "prepare_parallel_hidden",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: updates,
        queries: updates,
        checksum,
        aux_label: None,
        aux_elapsed: Duration::ZERO,
        aux_units: 0,
    })
}

fn bench_prepare_parallel_limited(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers.max(2)));
    let planner = Arc::new(BenchPlanner::plain());
    let backend = executor.register_backend_with_prepare_lanes(
        Arc::new(BenchBackend::hidden_prepare_heavy()),
        config.workers.max(2),
        1,
    );
    let stacks = make_stack_group(
        &executor,
        &planner,
        &backend,
        config,
        config.stack_count.max(config.workers),
    )?;

    warm_parallel_updates(&stacks, 1, config)?;

    let start = Instant::now();
    let (checksum, updates) = run_parallel_updates(&stacks, config.iterations, config)?;

    Ok(BenchReport {
        name: "prepare_parallel_limited",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: updates,
        queries: updates,
        checksum,
        aux_label: None,
        aux_elapsed: Duration::ZERO,
        aux_units: 0,
    })
}

fn bench_mixed_prepare_runtime_split(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers.max(2)));
    let planner = Arc::new(BenchPlanner::plain());
    let prepare_backend = executor.register_backend(
        Arc::new(BenchBackend::hidden_prepare_heavy()),
        BackendConfig {
            lane_count: config.workers.max(2),
        },
    );
    let runtime_backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig {
            lane_count: config.workers.max(2),
        },
    );
    let prepare_stacks = make_stack_group(
        &executor,
        &planner,
        &prepare_backend,
        config,
        config.stack_count.max(config.workers),
    )?;
    let fast_stack = make_stack(&executor, &planner, &runtime_backend, config, 20_001)?;
    let fast_batch_size = (config.batch_size / 8).max(16);

    warm_parallel_updates(&prepare_stacks, 1, config)?;
    warm_queue_pressure(&fast_stack, fast_batch_size)?;

    let start = Instant::now();
    let (fast_elapsed, checksum) = run_mixed_update_runtime(
        &prepare_stacks,
        &fast_stack,
        config.iterations,
        config,
        fast_batch_size,
    )?;

    Ok(BenchReport {
        name: "mixed_prepare_split",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: config.iterations * (prepare_stacks.len() + 1),
        queries: config.iterations * fast_batch_size,
        checksum,
        aux_label: Some("fast_ns/job"),
        aux_elapsed: fast_elapsed,
        aux_units: config.iterations,
    })
}

fn bench_mixed_prepare_runtime_same(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers.max(2)));
    let planner = Arc::new(BenchPlanner::plain());
    let shared_backend = executor.register_backend(
        Arc::new(BenchBackend::hidden_prepare_heavy()),
        BackendConfig {
            lane_count: config.workers.max(2),
        },
    );
    let prepare_stacks = make_stack_group(
        &executor,
        &planner,
        &shared_backend,
        config,
        config.stack_count.max(config.workers),
    )?;
    let fast_stack = make_stack(&executor, &planner, &shared_backend, config, 20_003)?;
    let fast_batch_size = (config.batch_size / 8).max(16);

    warm_parallel_updates(&prepare_stacks, 1, config)?;
    warm_queue_pressure(&fast_stack, fast_batch_size)?;

    let start = Instant::now();
    let (fast_elapsed, checksum) = run_mixed_update_runtime(
        &prepare_stacks,
        &fast_stack,
        config.iterations,
        config,
        fast_batch_size,
    )?;

    Ok(BenchReport {
        name: "mixed_prepare_same",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: config.iterations * (prepare_stacks.len() + 1),
        queries: config.iterations * fast_batch_size,
        checksum,
        aux_label: Some("fast_ns/job"),
        aux_elapsed: fast_elapsed,
        aux_units: config.iterations,
    })
}

fn bench_mixed_prepare_runtime_limited(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers.max(2)));
    let planner = Arc::new(BenchPlanner::plain());
    let shared_backend = executor.register_backend_with_prepare_lanes(
        Arc::new(BenchBackend::hidden_prepare_heavy()),
        config.workers.max(2),
        1,
    );
    let prepare_stacks = make_stack_group(
        &executor,
        &planner,
        &shared_backend,
        config,
        config.stack_count.max(config.workers),
    )?;
    let fast_stack = make_stack(&executor, &planner, &shared_backend, config, 20_007)?;
    let fast_batch_size = (config.batch_size / 8).max(16);

    warm_parallel_updates(&prepare_stacks, 1, config)?;
    warm_queue_pressure(&fast_stack, fast_batch_size)?;

    let start = Instant::now();
    let (fast_elapsed, checksum) = run_mixed_update_runtime(
        &prepare_stacks,
        &fast_stack,
        config.iterations,
        config,
        fast_batch_size,
    )?;

    Ok(BenchReport {
        name: "mixed_prepare_limited",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: config.iterations * (prepare_stacks.len() + 1),
        queries: config.iterations * fast_batch_size,
        checksum,
        aux_label: Some("fast_ns/job"),
        aux_elapsed: fast_elapsed,
        aux_units: config.iterations,
    })
}

fn bench_queue_pressure(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers));
    let planner = Arc::new(BenchPlanner::plain());
    let backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig {
            lane_count: config.workers,
        },
    );
    let stack = make_stack(&executor, &planner, &backend, config, 1)?;

    warm_queue_pressure(&stack, config.batch_size)?;

    let start = Instant::now();
    let mut checksum = 0u64;
    for iter in 0..config.iterations {
        let mut jobs = Vec::with_capacity(config.queue_depth);
        for job_index in 0..config.queue_depth {
            let seed = ((iter * config.queue_depth) + job_index) as u32;
            jobs.push(stack.dispatch(make_queries(seed, config.batch_size))?);
        }

        for job in jobs {
            let values = stack.collect(job)?;
            checksum = checksum.wrapping_add(digest(values.as_slice()));
            black_box(&values);
        }
    }

    Ok(BenchReport {
        name: "queue_pressure",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: config.iterations * config.queue_depth,
        queries: config.iterations * config.queue_depth * config.batch_size,
        checksum,
        aux_label: None,
        aux_elapsed: Duration::ZERO,
        aux_units: 0,
    })
}

fn bench_update_between_runs(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers));
    let planner = Arc::new(BenchPlanner::plain());
    let backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig {
            lane_count: config.workers,
        },
    );
    let stack = make_stack(&executor, &planner, &backend, config, 3)?;

    warm_update_between_runs(&stack, config)?;

    let start = Instant::now();
    let mut checksum = 0u64;
    for iter in 0..config.iterations {
        let lane_index = iter % config.lane_count;
        let version = stack.update(&[
            BenchChange::SetBias(mix32(iter as u32).wrapping_add(17)),
            BenchChange::SetMultiplier(mix32((iter as u32) ^ 0xA55A_A55A) | 1),
            BenchChange::PatchLane {
                index: lane_index,
                value: mix32((iter as u32).wrapping_add(lane_index as u32)),
            },
        ])?;
        black_box(version);

        let values =
            stack.collect(stack.dispatch(make_queries(iter as u32, config.batch_size))?)?;
        checksum = checksum.wrapping_add(digest(values.as_slice()));
        black_box(&values);
    }

    Ok(BenchReport {
        name: "update_between_runs",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: config.iterations,
        queries: config.iterations * config.batch_size,
        checksum,
        aux_label: None,
        aux_elapsed: Duration::ZERO,
        aux_units: 0,
    })
}

fn bench_dependency_chain(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers));
    let planner = Arc::new(BenchPlanner::plain());
    let backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig {
            lane_count: config.workers,
        },
    );
    let stacks = make_stack_group(
        &executor,
        &planner,
        &backend,
        config,
        config.dependency_depth,
    )?;

    warm_dependency_chain(&stacks, config.batch_size)?;

    let start = Instant::now();
    let mut checksum = 0u64;
    for iter in 0..config.iterations {
        // Core has no built-in inter-stack scheduler yet. This models app-level chaining.
        let mut values = make_queries(iter as u32, config.batch_size);
        for stack in &stacks {
            let job = stack.dispatch(values)?;
            values = stack.collect(job)?;
            checksum = checksum.wrapping_add(digest(values.as_slice()));
            black_box(&values);
        }
    }

    Ok(BenchReport {
        name: "dependency_chain",
        elapsed: start.elapsed(),
        iterations: config.iterations,
        jobs: config.iterations * config.dependency_depth,
        queries: config.iterations * config.dependency_depth * config.batch_size,
        checksum,
        aux_label: None,
        aux_elapsed: Duration::ZERO,
        aux_units: 0,
    })
}

fn bench_version_swap_inflight(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers.max(2)));
    let planner = Arc::new(BenchPlanner::plain());
    let backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig {
            lane_count: config.workers.max(2),
        },
    );
    let stack = make_stack(&executor, &planner, &backend, config, 7)?;
    let iterations = (config.iterations / 2).max(1);
    let inflight_batch_size = config.batch_size * 4;

    warm_version_swap_inflight(&stack, config, inflight_batch_size)?;

    let start = Instant::now();
    let mut checksum = 0u64;
    for iter in 0..iterations {
        let old_queries = make_queries((iter as u32) ^ 0xDEAD_BEEF, inflight_batch_size);
        let new_queries = old_queries.clone();
        let old_job = stack.dispatch(old_queries)?;

        let version = stack.update(&[
            BenchChange::SetBias(mix32((iter as u32).wrapping_add(101))),
            BenchChange::PatchLane {
                index: iter % config.lane_count,
                value: mix32((iter as u32).wrapping_mul(17)),
            },
        ])?;
        black_box(version);

        let new_job = stack.dispatch(new_queries)?;
        let old_values = stack.collect(old_job)?;
        let new_values = stack.collect(new_job)?;
        checksum = checksum
            .wrapping_add(digest(old_values.as_slice()))
            .wrapping_add(digest(new_values.as_slice()));
        black_box((&old_values, &new_values));
    }

    Ok(BenchReport {
        name: "version_swap_inflight",
        elapsed: start.elapsed(),
        iterations,
        jobs: iterations * 2,
        queries: iterations * inflight_batch_size * 2,
        checksum,
        aux_label: None,
        aux_elapsed: Duration::ZERO,
        aux_units: 0,
    })
}

fn bench_create_stack_and_run(config: &BenchConfig) -> BraidResult<BenchReport> {
    let executor = Arc::new(BraidExecutor::new(config.workers));
    let planner = Arc::new(BenchPlanner::plain());
    let backend = executor.register_backend(
        Arc::new(BenchBackend::plain()),
        BackendConfig {
            lane_count: config.workers,
        },
    );
    let iterations = (config.iterations / 2).max(1);

    warm_create_stack_and_run(&executor, &planner, &backend, config)?;

    let start = Instant::now();
    let mut checksum = 0u64;
    for iter in 0..iterations {
        let stack = make_stack(&executor, &planner, &backend, config, 100 + iter)?;
        let values =
            stack.collect(stack.dispatch(make_queries(iter as u32, config.batch_size))?)?;
        checksum = checksum.wrapping_add(digest(values.as_slice()));
        black_box(&values);
    }

    Ok(BenchReport {
        name: "create_stack_and_run",
        elapsed: start.elapsed(),
        iterations,
        jobs: iterations,
        queries: iterations * config.batch_size,
        checksum,
        aux_label: None,
        aux_elapsed: Duration::ZERO,
        aux_units: 0,
    })
}

fn make_stack_group(
    executor: &Arc<BraidExecutor>,
    planner: &Arc<BenchPlanner>,
    backend: &braid::BackendHandle<BenchBackend>,
    config: &BenchConfig,
    count: usize,
) -> BraidResult<Vec<Stack<BenchPlanner, BenchBackend>>> {
    let mut stacks = Vec::with_capacity(count);
    for index in 0..count {
        stacks.push(make_stack(executor, planner, backend, config, index)?);
    }
    Ok(stacks)
}

fn make_stack(
    executor: &Arc<BraidExecutor>,
    planner: &Arc<BenchPlanner>,
    backend: &braid::BackendHandle<BenchBackend>,
    config: &BenchConfig,
    seed: usize,
) -> BraidResult<Stack<BenchPlanner, BenchBackend>> {
    Stack::create(
        Arc::clone(executor),
        Arc::clone(planner),
        backend.clone(),
        make_spec(seed as u32, config),
    )
}

fn make_spec(seed: u32, config: &BenchConfig) -> BenchSpec {
    let mut lane_values = Vec::with_capacity(config.lane_count);
    for index in 0..config.lane_count {
        lane_values.push(mix32(
            seed.wrapping_add((index as u32).wrapping_mul(0x9E37_79B9)),
        ));
    }

    BenchSpec {
        bias: mix32(seed ^ 0x1357_9BDF),
        multiplier: mix32(seed ^ 0x2468_ACE0) | 1,
        lane_values,
        stage1_rounds: config.stage1_rounds,
        stage2_rounds: config.stage2_rounds,
    }
}

fn make_queries(seed: u32, batch_size: usize) -> Vec<u32> {
    let mut queries = Vec::with_capacity(batch_size);
    for index in 0..batch_size {
        queries.push(mix32(
            seed.wrapping_add((index as u32).wrapping_mul(0x85EB_CA6B)),
        ));
    }
    queries
}

fn make_update_changes(
    iteration: usize,
    worker_index: usize,
    lane_count: usize,
) -> Vec<BenchChange> {
    let lane_index = (iteration + worker_index) % lane_count.max(1);
    vec![
        BenchChange::SetBias(mix32(
            (iteration as u32).wrapping_add(worker_index as u32 + 17),
        )),
        BenchChange::SetMultiplier(
            mix32((iteration as u32) ^ ((worker_index as u32) << 8) ^ 0xA55A_A55A) | 1,
        ),
        BenchChange::PatchLane {
            index: lane_index,
            value: mix32((iteration as u32).wrapping_add((lane_index as u32) << 1)),
        },
    ]
}

fn digest(values: &[u32]) -> u64 {
    let first = values.first().copied().unwrap_or(0) as u64;
    let middle = values.get(values.len() / 2).copied().unwrap_or(0) as u64;
    let last = values.last().copied().unwrap_or(0) as u64;
    first ^ (middle << 1) ^ (last << 2) ^ values.len() as u64
}

fn run_parallel_updates(
    stacks: &[Stack<BenchPlanner, BenchBackend>],
    iterations: usize,
    config: &BenchConfig,
) -> BraidResult<(u64, usize)> {
    let worker_count = config.workers.max(1).min(stacks.len().max(1));
    let mut assignments = vec![Vec::new(); worker_count];
    for (index, stack) in stacks.iter().cloned().enumerate() {
        assignments[index % worker_count].push(stack);
    }

    thread::scope(|scope| -> BraidResult<(u64, usize)> {
        let mut handles = Vec::with_capacity(assignments.len());
        for (worker_index, owned_stacks) in assignments.into_iter().enumerate() {
            handles.push(scope.spawn(move || -> BraidResult<(u64, usize)> {
                let mut checksum = 0u64;
                let mut updates = 0usize;
                for iteration in 0..iterations {
                    for (local_index, stack) in owned_stacks.iter().enumerate() {
                        let version = stack.update(&make_update_changes(
                            iteration + local_index,
                            worker_index,
                            config.lane_count,
                        ))?;
                        checksum = checksum.wrapping_add(version);
                        updates += 1;
                    }
                }
                Ok((checksum, updates))
            }));
        }

        let mut checksum = 0u64;
        let mut updates = 0usize;
        for handle in handles {
            let (part_checksum, part_updates) = handle.join().unwrap()?;
            checksum = checksum.wrapping_add(part_checksum);
            updates += part_updates;
        }
        Ok((checksum, updates))
    })
}

fn run_mixed_update_runtime(
    prepare_stacks: &[Stack<BenchPlanner, BenchBackend>],
    fast_stack: &Stack<BenchPlanner, BenchBackend>,
    iterations: usize,
    config: &BenchConfig,
    fast_batch_size: usize,
) -> BraidResult<(Duration, u64)> {
    let worker_count = config.workers.max(1).min(prepare_stacks.len().max(1));
    let mut assignments = vec![Vec::new(); worker_count];
    for (index, stack) in prepare_stacks.iter().cloned().enumerate() {
        assignments[index % worker_count].push(stack);
    }

    thread::scope(|scope| -> BraidResult<(Duration, u64)> {
        let barrier = Arc::new(Barrier::new(assignments.len() + 1));
        let mut handles = Vec::with_capacity(assignments.len());
        for (worker_index, owned_stacks) in assignments.into_iter().enumerate() {
            let barrier = Arc::clone(&barrier);
            handles.push(scope.spawn(move || -> BraidResult<()> {
                barrier.wait();
                for iteration in 0..iterations {
                    for (local_index, stack) in owned_stacks.iter().enumerate() {
                        stack.update(&make_update_changes(
                            iteration + local_index,
                            worker_index,
                            config.lane_count,
                        ))?;
                    }
                }
                Ok(())
            }));
        }

        barrier.wait();
        let mut fast_elapsed = Duration::ZERO;
        let mut checksum = 0u64;
        for iteration in 0..iterations {
            let fast_start = Instant::now();
            let job =
                fast_stack.dispatch(make_queries((iteration as u32) ^ 0xFA57, fast_batch_size))?;
            let fast_values = fast_stack.collect(job)?;
            fast_elapsed += fast_start.elapsed();
            checksum = checksum.wrapping_add(digest(fast_values.as_slice()));
            black_box(&fast_values);
        }

        for handle in handles {
            handle.join().unwrap()?;
        }
        Ok((fast_elapsed, checksum))
    })
}

fn warm_parallel_stacks(
    stacks: &[Stack<BenchPlanner, BenchBackend>],
    batch_size: usize,
) -> BraidResult<()> {
    let mut jobs = Vec::with_capacity(stacks.len());
    for (index, stack) in stacks.iter().enumerate() {
        jobs.push((
            index,
            stack.dispatch(make_queries(index as u32, batch_size))?,
        ));
    }
    for (index, job) in jobs {
        black_box(stacks[index].collect(job)?);
    }
    Ok(())
}

fn warm_inline_and_async(stack: &Stack<BenchPlanner, BenchBackend>) -> BraidResult<()> {
    let query = [mix32(0x5151_5151)];
    let mut inline = InlineContext::default();
    black_box(stack.resolve_inline_with(&query, &mut inline)?);
    black_box(stack.resolve_one_inline_with(query[0], &mut inline)?);
    let job = stack.dispatch(query.to_vec())?;
    black_box(stack.collect(job)?);
    Ok(())
}

fn warm_queue_pressure(
    stack: &Stack<BenchPlanner, BenchBackend>,
    batch_size: usize,
) -> BraidResult<()> {
    let job = stack.dispatch(make_queries(11, batch_size))?;
    black_box(stack.collect(job)?);
    Ok(())
}

fn warm_parallel_updates(
    stacks: &[Stack<BenchPlanner, BenchBackend>],
    iterations: usize,
    config: &BenchConfig,
) -> BraidResult<()> {
    black_box(run_parallel_updates(stacks, iterations, config)?);
    Ok(())
}

fn warm_update_between_runs(
    stack: &Stack<BenchPlanner, BenchBackend>,
    config: &BenchConfig,
) -> BraidResult<()> {
    black_box(stack.update(&[
        BenchChange::SetBias(7),
        BenchChange::SetMultiplier(9),
        BenchChange::PatchLane {
            index: 0,
            value: 11,
        },
    ])?);
    let job = stack.dispatch(make_queries(12, config.batch_size))?;
    black_box(stack.collect(job)?);
    Ok(())
}

fn warm_dependency_chain(
    stacks: &[Stack<BenchPlanner, BenchBackend>],
    batch_size: usize,
) -> BraidResult<()> {
    let mut values = make_queries(13, batch_size);
    for stack in stacks {
        let job = stack.dispatch(values)?;
        values = stack.collect(job)?;
    }
    black_box(values);
    Ok(())
}

fn warm_version_swap_inflight(
    stack: &Stack<BenchPlanner, BenchBackend>,
    config: &BenchConfig,
    inflight_batch_size: usize,
) -> BraidResult<()> {
    let old_job = stack.dispatch(make_queries(21, inflight_batch_size))?;
    black_box(stack.update(&[
        BenchChange::SetBias(23),
        BenchChange::PatchLane {
            index: 1 % config.lane_count,
            value: 29,
        },
    ])?);
    let new_job = stack.dispatch(make_queries(21, inflight_batch_size))?;
    black_box(stack.collect(old_job)?);
    black_box(stack.collect(new_job)?);
    Ok(())
}

fn warm_create_stack_and_run(
    executor: &Arc<BraidExecutor>,
    planner: &Arc<BenchPlanner>,
    backend: &braid::BackendHandle<BenchBackend>,
    config: &BenchConfig,
) -> BraidResult<()> {
    let stack = make_stack(executor, planner, backend, config, 31)?;
    let job = stack.dispatch(make_queries(32, config.batch_size))?;
    black_box(stack.collect(job)?);
    Ok(())
}
