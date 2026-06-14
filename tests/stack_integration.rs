use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::thread;
use std::time::{Duration, Instant};

use braids::{
    BackendConfig, BatchScratch, BraidError, BraidExecutor, BraidResult, BufferAccess,
    BufferBinding, BufferLayout, BufferSlot, BufferSpec, CancelFlag, CompiledPlan, ComputeScratch,
    DispatchHint, ElementKind, InlineContext, JobPacket, JobStatus, KernelKind, KernelSpec,
    PipelineShape, PlannerBackend, PlannerScratch, Stack, StageSpec,
};

const TOY_INPUT_SLOT: BufferSlot = BufferSlot(0);
const TOY_OUTPUT_SLOT: BufferSlot = BufferSlot(1);
const TOY_KIND: KernelKind = KernelKind(77);

#[derive(Default)]
struct ToyPlanner;

#[derive(Clone)]
struct ToySpec {
    bonus: u32,
    delay_ms: u64,
}

struct ToyState {
    bonus: u32,
    delay_ms: u64,
}

enum ToyChange {
    SetBonus(u32),
}

struct ToyBackend;

#[derive(Debug)]
struct ToyPrepared {
    bonus: u32,
    delay_ms: u64,
}

impl PlannerBackend for ToyPlanner {
    const PREFER_ONE_QUERY_INLINE: bool = true;

    type Spec = ToySpec;
    type State = ToyState;
    type Change = ToyChange;
    type Query = u32;
    type Resolution = u32;
    type PlannerMeta = ();

    fn init_state(&self, spec: &Self::Spec) -> BraidResult<Self::State> {
        Ok(ToyState {
            bonus: spec.bonus,
            delay_ms: spec.delay_ms,
        })
    }

    fn reset_state(&self, state: &mut Self::State, spec: &Self::Spec) -> BraidResult<()> {
        state.bonus = spec.bonus;
        state.delay_ms = spec.delay_ms;
        Ok(())
    }

    fn apply(&self, state: &mut Self::State, changes: &[Self::Change]) -> BraidResult<()> {
        for change in changes {
            match change {
                ToyChange::SetBonus(bonus) => state.bonus = *bonus,
            }
        }
        Ok(())
    }

    fn updated_state(
        &self,
        state: &Self::State,
        changes: &[Self::Change],
    ) -> BraidResult<Self::State> {
        let mut next = ToyState {
            bonus: state.bonus,
            delay_ms: state.delay_ms,
        };
        self.apply(&mut next, changes)?;
        Ok(next)
    }

    fn compile(
        &self,
        state: &Self::State,
        scratch: &mut PlannerScratch,
    ) -> BraidResult<CompiledPlan<Self::PlannerMeta>> {
        if state.bonus == u32::MAX {
            return Err(BraidError::InvalidSpec(
                "toy bonus cannot be u32::MAX".to_owned(),
            ));
        }
        scratch.reset();
        scratch
            .bytes
            .extend_from_slice(&state.delay_ms.to_le_bytes());
        scratch.bytes.extend_from_slice(&state.bonus.to_le_bytes());
        let plan = CompiledPlan {
            pipeline: PipelineShape {
                buffers: vec![
                    BufferSpec {
                        slot: TOY_INPUT_SLOT,
                        element_kind: ElementKind::U32,
                        layout: BufferLayout::PerQueryScalar,
                    },
                    BufferSpec {
                        slot: TOY_OUTPUT_SLOT,
                        element_kind: ElementKind::U32,
                        layout: BufferLayout::PerQueryScalar,
                    },
                ],
                stages: vec![StageSpec {
                    kernels: vec![KernelSpec {
                        kind_id: TOY_KIND,
                        payload: scratch.bytes.clone().into(),
                        bindings: vec![
                            BufferBinding {
                                slot: TOY_INPUT_SLOT,
                                access: BufferAccess::Read,
                            },
                            BufferBinding {
                                slot: TOY_OUTPUT_SLOT,
                                access: BufferAccess::Write,
                            },
                        ],
                        dispatch: DispatchHint::WholeBatch,
                    }],
                }],
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
        packet
            .ensure::<u32>(TOY_INPUT_SLOT, queries.len())
            .copy_from_slice(queries);
        packet.ensure::<u32>(TOY_OUTPUT_SLOT, queries.len()).fill(0);
        Ok(())
    }

    fn encode_one(
        &self,
        _plan: &CompiledPlan<Self::PlannerMeta>,
        query: &Self::Query,
        packet: &mut JobPacket,
        _scratch: &mut BatchScratch,
    ) -> BraidResult<()> {
        packet.query_count = 1;
        packet.ensure::<u32>(TOY_INPUT_SLOT, 1)[0] = *query;
        packet.ensure::<u32>(TOY_OUTPUT_SLOT, 1)[0] = 0;
        Ok(())
    }

    fn decode_batch(
        &self,
        _plan: &CompiledPlan<Self::PlannerMeta>,
        packet: &JobPacket,
    ) -> BraidResult<Vec<Self::Resolution>> {
        Ok(packet.slice::<u32>(TOY_OUTPUT_SLOT)?.to_vec())
    }

    fn decode_one(
        &self,
        _plan: &CompiledPlan<Self::PlannerMeta>,
        packet: &JobPacket,
    ) -> BraidResult<Self::Resolution> {
        packet
            .slice::<u32>(TOY_OUTPUT_SLOT)?
            .first()
            .copied()
            .ok_or_else(|| BraidError::from("missing toy output value"))
    }
}

struct CountingPlanner {
    counts: Arc<CountingPlannerCounts>,
}

#[derive(Default)]
struct CountingPlannerCounts {
    encode_batch_calls: AtomicUsize,
    encode_one_calls: AtomicUsize,
    decode_batch_calls: AtomicUsize,
    decode_one_calls: AtomicUsize,
}

struct CountingBackend {
    counts: Arc<CountingBackendCounts>,
}

#[derive(Default)]
struct CountingBackendCounts {
    run_stage_calls: AtomicUsize,
    run_one_stage_calls: AtomicUsize,
}

struct DirectPlanner;

struct DirectBackend {
    counts: Arc<CountingBackendCounts>,
}

impl PlannerBackend for CountingPlanner {
    const PREFER_ONE_QUERY_INLINE: bool = true;

    type Spec = ToySpec;
    type State = ToyState;
    type Change = ToyChange;
    type Query = u32;
    type Resolution = u32;
    type PlannerMeta = ();

    fn init_state(&self, spec: &Self::Spec) -> BraidResult<Self::State> {
        Ok(ToyState {
            bonus: spec.bonus,
            delay_ms: spec.delay_ms,
        })
    }

    fn reset_state(&self, state: &mut Self::State, spec: &Self::Spec) -> BraidResult<()> {
        state.bonus = spec.bonus;
        state.delay_ms = spec.delay_ms;
        Ok(())
    }

    fn apply(&self, state: &mut Self::State, changes: &[Self::Change]) -> BraidResult<()> {
        for change in changes {
            match change {
                ToyChange::SetBonus(bonus) => state.bonus = *bonus,
            }
        }
        Ok(())
    }

    fn updated_state(
        &self,
        state: &Self::State,
        changes: &[Self::Change],
    ) -> BraidResult<Self::State> {
        let mut next = ToyState {
            bonus: state.bonus,
            delay_ms: state.delay_ms,
        };
        self.apply(&mut next, changes)?;
        Ok(next)
    }

    fn compile(
        &self,
        state: &Self::State,
        scratch: &mut PlannerScratch,
    ) -> BraidResult<CompiledPlan<Self::PlannerMeta>> {
        ToyPlanner.compile(state, scratch)
    }

    fn encode_batch(
        &self,
        _plan: &CompiledPlan<Self::PlannerMeta>,
        queries: &[Self::Query],
        packet: &mut JobPacket,
        _scratch: &mut BatchScratch,
    ) -> BraidResult<()> {
        self.counts
            .encode_batch_calls
            .fetch_add(1, AtomicOrdering::Relaxed);
        packet.query_count = queries.len();
        packet
            .ensure::<u32>(TOY_INPUT_SLOT, queries.len())
            .copy_from_slice(queries);
        packet.ensure::<u32>(TOY_OUTPUT_SLOT, queries.len()).fill(0);
        Ok(())
    }

    fn encode_one(
        &self,
        _plan: &CompiledPlan<Self::PlannerMeta>,
        query: &Self::Query,
        packet: &mut JobPacket,
        _scratch: &mut BatchScratch,
    ) -> BraidResult<()> {
        self.counts
            .encode_one_calls
            .fetch_add(1, AtomicOrdering::Relaxed);
        packet.query_count = 1;
        packet.ensure::<u32>(TOY_INPUT_SLOT, 1)[0] = *query;
        packet.ensure::<u32>(TOY_OUTPUT_SLOT, 1)[0] = 0;
        Ok(())
    }

    fn decode_batch(
        &self,
        _plan: &CompiledPlan<Self::PlannerMeta>,
        packet: &JobPacket,
    ) -> BraidResult<Vec<Self::Resolution>> {
        self.counts
            .decode_batch_calls
            .fetch_add(1, AtomicOrdering::Relaxed);
        Ok(packet.slice::<u32>(TOY_OUTPUT_SLOT)?.to_vec())
    }

    fn decode_one(
        &self,
        _plan: &CompiledPlan<Self::PlannerMeta>,
        packet: &JobPacket,
    ) -> BraidResult<Self::Resolution> {
        self.counts
            .decode_one_calls
            .fetch_add(1, AtomicOrdering::Relaxed);
        packet
            .slice::<u32>(TOY_OUTPUT_SLOT)?
            .first()
            .copied()
            .ok_or_else(|| BraidError::from("missing toy output value"))
    }
}

impl braids::ComputeBackend for ToyBackend {
    type Prepared = ToyPrepared;

    fn prepare<M: Send + Sync + 'static>(
        &self,
        plan: &CompiledPlan<M>,
        _reuse: Option<Self::Prepared>,
        _scratch: &mut ComputeScratch,
    ) -> BraidResult<Self::Prepared> {
        let payload = &plan.pipeline.stages[0].kernels[0].payload;
        let delay_ms = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        let bonus = u32::from_le_bytes(payload[8..12].try_into().unwrap());
        Ok(ToyPrepared { bonus, delay_ms })
    }

    fn run_stage(
        &self,
        prepared: &Self::Prepared,
        _stage_index: usize,
        _stage: &StageSpec,
        packet: &mut JobPacket,
        cancel: &CancelFlag,
    ) -> BraidResult<()> {
        toy_backend_run_stage(prepared, packet, cancel)
    }
}

impl braids::ComputeBackend for CountingBackend {
    type Prepared = ToyPrepared;

    fn prepare<M: Send + Sync + 'static>(
        &self,
        plan: &CompiledPlan<M>,
        _reuse: Option<Self::Prepared>,
        scratch: &mut ComputeScratch,
    ) -> BraidResult<Self::Prepared> {
        ToyBackend.prepare(plan, None, scratch)
    }

    fn run_stage(
        &self,
        prepared: &Self::Prepared,
        stage_index: usize,
        stage: &StageSpec,
        packet: &mut JobPacket,
        cancel: &CancelFlag,
    ) -> BraidResult<()> {
        self.counts
            .run_stage_calls
            .fetch_add(1, AtomicOrdering::Relaxed);
        let _ = (stage_index, stage);
        toy_backend_run_stage(prepared, packet, cancel)
    }

    fn run_one_stage(
        &self,
        prepared: &Self::Prepared,
        stage_index: usize,
        stage: &StageSpec,
        packet: &mut JobPacket,
        cancel: &CancelFlag,
    ) -> BraidResult<()> {
        self.counts
            .run_one_stage_calls
            .fetch_add(1, AtomicOrdering::Relaxed);
        let _ = (stage_index, stage);
        toy_backend_run_stage(prepared, packet, cancel)
    }
}

impl PlannerBackend for DirectPlanner {
    const PREFER_DIRECT_ONE_QUERY_INLINE: bool = true;

    type Spec = ToySpec;
    type State = ToyState;
    type Change = ToyChange;
    type Query = u32;
    type Resolution = u32;
    type PlannerMeta = u32;

    fn init_state(&self, spec: &Self::Spec) -> BraidResult<Self::State> {
        Ok(ToyState {
            bonus: spec.bonus,
            delay_ms: spec.delay_ms,
        })
    }

    fn reset_state(&self, state: &mut Self::State, spec: &Self::Spec) -> BraidResult<()> {
        state.bonus = spec.bonus;
        state.delay_ms = spec.delay_ms;
        Ok(())
    }

    fn apply(&self, state: &mut Self::State, changes: &[Self::Change]) -> BraidResult<()> {
        for change in changes {
            match change {
                ToyChange::SetBonus(bonus) => state.bonus = *bonus,
            }
        }
        Ok(())
    }

    fn updated_state(
        &self,
        state: &Self::State,
        changes: &[Self::Change],
    ) -> BraidResult<Self::State> {
        let mut next = ToyState {
            bonus: state.bonus,
            delay_ms: state.delay_ms,
        };
        self.apply(&mut next, changes)?;
        Ok(next)
    }

    fn compile(
        &self,
        state: &Self::State,
        _scratch: &mut PlannerScratch,
    ) -> BraidResult<CompiledPlan<Self::PlannerMeta>> {
        let plan = CompiledPlan {
            pipeline: PipelineShape::default(),
            static_buffers: Vec::new(),
            planner_meta: state.bonus,
        };
        plan.validate()?;
        Ok(plan)
    }

    fn encode_batch(
        &self,
        _plan: &CompiledPlan<Self::PlannerMeta>,
        _queries: &[Self::Query],
        _packet: &mut JobPacket,
        _scratch: &mut BatchScratch,
    ) -> BraidResult<()> {
        Err(BraidError::from(
            "direct planner batch encode should not run",
        ))
    }

    fn decode_batch(
        &self,
        _plan: &CompiledPlan<Self::PlannerMeta>,
        _packet: &JobPacket,
    ) -> BraidResult<Vec<Self::Resolution>> {
        Err(BraidError::from(
            "direct planner batch decode should not run",
        ))
    }

    fn resolve_one_direct(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        query: &Self::Query,
    ) -> Option<BraidResult<Self::Resolution>> {
        Some(Ok(*query + plan.planner_meta))
    }
}

impl braids::ComputeBackend for DirectBackend {
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
        _stage: &StageSpec,
        _packet: &mut JobPacket,
        _cancel: &CancelFlag,
    ) -> BraidResult<()> {
        self.counts
            .run_stage_calls
            .fetch_add(1, AtomicOrdering::Relaxed);
        Err(BraidError::from(
            "direct planner backend stage should not run",
        ))
    }

    fn run_one_stage(
        &self,
        _prepared: &Self::Prepared,
        _stage_index: usize,
        _stage: &StageSpec,
        _packet: &mut JobPacket,
        _cancel: &CancelFlag,
    ) -> BraidResult<()> {
        self.counts
            .run_one_stage_calls
            .fetch_add(1, AtomicOrdering::Relaxed);
        Err(BraidError::from(
            "direct planner one-query backend stage should not run",
        ))
    }
}

fn toy_backend_run_stage(
    prepared: &ToyPrepared,
    packet: &mut JobPacket,
    cancel: &CancelFlag,
) -> BraidResult<()> {
    thread::sleep(Duration::from_millis(prepared.delay_ms));
    if cancel.is_cancelled() {
        return Err(BraidError::Cancelled);
    }
    let inputs = packet.slice::<u32>(TOY_INPUT_SLOT)?.to_vec();
    let bonus = prepared.bonus;
    let mut outputs = vec![0u32; inputs.len()];
    for (ix, value) in inputs.iter().enumerate() {
        outputs[ix] = value + bonus;
    }
    packet
        .slice_mut::<u32>(TOY_OUTPUT_SLOT)?
        .copy_from_slice(outputs.as_slice());
    Ok(())
}

#[test]
fn stack_update_swaps_versions_without_clobbering_old_jobs() {
    let executor = Arc::new(BraidExecutor::new(2));
    let planner = Arc::new(ToyPlanner);
    let backend = executor.register_backend(Arc::new(ToyBackend), BackendConfig { lane_count: 2 });
    let stack = Stack::create(
        Arc::clone(&executor),
        planner,
        backend,
        ToySpec {
            bonus: 1,
            delay_ms: 80,
        },
    )
    .unwrap();

    let old_job = stack.dispatch(vec![10]).unwrap();
    let old_version = stack.current_version_id().unwrap();
    let new_version = stack.update(&[ToyChange::SetBonus(100)]).unwrap();
    assert!(new_version > old_version);
    let new_job = stack.dispatch(vec![10]).unwrap();

    assert_eq!(stack.collect(old_job).unwrap(), vec![11]);
    assert_eq!(stack.collect(new_job).unwrap(), vec![110]);
}

#[test]
fn failed_update_rolls_back_state() {
    let executor = Arc::new(BraidExecutor::new(1));
    let planner = Arc::new(ToyPlanner);
    let backend = executor.register_backend(Arc::new(ToyBackend), BackendConfig { lane_count: 1 });
    let stack = Stack::create(
        executor,
        planner,
        backend,
        ToySpec {
            bonus: 1,
            delay_ms: 0,
        },
    )
    .unwrap();

    let version_before = stack.current_version_id().unwrap();
    let error = stack.update(&[ToyChange::SetBonus(u32::MAX)]).unwrap_err();
    assert!(error.to_string().contains("u32::MAX"));
    assert_eq!(stack.current_version_id().unwrap(), version_before);

    let version_after = stack.recompile().unwrap();
    assert!(version_after > version_before);
    let job = stack.dispatch(vec![10]).unwrap();
    assert_eq!(stack.collect(job).unwrap(), vec![11]);
}

#[test]
fn resolve_inline_matches_async_and_uses_latest_version() {
    let executor = Arc::new(BraidExecutor::new(1));
    let planner = Arc::new(ToyPlanner);
    let backend = executor.register_backend(Arc::new(ToyBackend), BackendConfig { lane_count: 1 });
    let stack = Stack::create(
        executor,
        planner,
        backend,
        ToySpec {
            bonus: 3,
            delay_ms: 0,
        },
    )
    .unwrap();

    let mut inline = InlineContext::default();
    assert_eq!(
        stack.resolve_inline_with(&[1, 2], &mut inline).unwrap(),
        vec![4, 5]
    );

    let job = stack.dispatch(vec![1, 2]).unwrap();
    assert_eq!(stack.collect(job).unwrap(), vec![4, 5]);

    stack.update(&[ToyChange::SetBonus(9)]).unwrap();
    assert_eq!(stack.resolve_one_inline_with(1, &mut inline).unwrap(), 10);
}

#[test]
fn single_query_inline_prefers_one_query_planner_hooks() {
    let counts = Arc::new(CountingPlannerCounts::default());
    let backend_counts = Arc::new(CountingBackendCounts::default());
    let executor = Arc::new(BraidExecutor::new(1));
    let planner = Arc::new(CountingPlanner {
        counts: Arc::clone(&counts),
    });
    let backend = executor.register_backend(
        Arc::new(CountingBackend {
            counts: Arc::clone(&backend_counts),
        }),
        BackendConfig { lane_count: 1 },
    );
    let stack = Stack::create(
        executor,
        planner,
        backend,
        ToySpec {
            bonus: 3,
            delay_ms: 0,
        },
    )
    .unwrap();

    let mut inline = InlineContext::default();
    assert_eq!(
        stack.resolve_inline_with(&[1], &mut inline).unwrap(),
        vec![4]
    );
    assert_eq!(
        stack.resolve_one_inline_ref_with(&2, &mut inline).unwrap(),
        5
    );

    assert_eq!(counts.encode_batch_calls.load(AtomicOrdering::Relaxed), 0);
    assert_eq!(counts.decode_batch_calls.load(AtomicOrdering::Relaxed), 0);
    assert_eq!(counts.encode_one_calls.load(AtomicOrdering::Relaxed), 2);
    assert_eq!(counts.decode_one_calls.load(AtomicOrdering::Relaxed), 2);
    assert_eq!(
        backend_counts.run_stage_calls.load(AtomicOrdering::Relaxed),
        0
    );
    assert_eq!(
        backend_counts
            .run_one_stage_calls
            .load(AtomicOrdering::Relaxed),
        2
    );
}

#[test]
fn single_query_inline_can_bypass_packet_and_backend() {
    let backend_counts = Arc::new(CountingBackendCounts::default());
    let executor = Arc::new(BraidExecutor::new(1));
    let planner = Arc::new(DirectPlanner);
    let backend = executor.register_backend(
        Arc::new(DirectBackend {
            counts: Arc::clone(&backend_counts),
        }),
        BackendConfig { lane_count: 1 },
    );
    let stack = Stack::create(
        executor,
        planner,
        backend,
        ToySpec {
            bonus: 7,
            delay_ms: 0,
        },
    )
    .unwrap();

    let mut inline = InlineContext::default();
    assert_eq!(
        stack.resolve_inline_with(&[1], &mut inline).unwrap(),
        vec![8]
    );
    assert_eq!(
        stack.resolve_one_inline_ref_with(&2, &mut inline).unwrap(),
        9
    );

    assert_eq!(
        backend_counts.run_stage_calls.load(AtomicOrdering::Relaxed),
        0
    );
    assert_eq!(
        backend_counts
            .run_one_stage_calls
            .load(AtomicOrdering::Relaxed),
        0
    );
}

#[test]
fn executor_runs_stacks_in_parallel_with_shared_backend_instance() {
    let executor = Arc::new(BraidExecutor::new(2));
    let planner = Arc::new(ToyPlanner);
    let backend = executor.register_backend(Arc::new(ToyBackend), BackendConfig { lane_count: 2 });

    let stack_a = Stack::create(
        Arc::clone(&executor),
        Arc::clone(&planner),
        backend.clone(),
        ToySpec {
            bonus: 10,
            delay_ms: 120,
        },
    )
    .unwrap();
    let stack_b = Stack::create(
        executor,
        planner,
        backend,
        ToySpec {
            bonus: 20,
            delay_ms: 120,
        },
    )
    .unwrap();

    let start = Instant::now();
    let job_a = stack_a.dispatch(vec![1]).unwrap();
    let job_b = stack_b.dispatch(vec![1]).unwrap();
    let out_a = stack_a.collect(job_a).unwrap();
    let out_b = stack_b.collect(job_b).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(out_a, vec![11]);
    assert_eq!(out_b, vec![21]);
    assert!(
        elapsed < Duration::from_millis(220),
        "elapsed {:?} looked serial",
        elapsed
    );
}

#[test]
fn queued_backend_work_does_not_block_other_backends() {
    let executor = Arc::new(BraidExecutor::new(2));
    let planner = Arc::new(ToyPlanner);
    let slow_backend =
        executor.register_backend(Arc::new(ToyBackend), BackendConfig { lane_count: 1 });
    let fast_backend =
        executor.register_backend(Arc::new(ToyBackend), BackendConfig { lane_count: 1 });

    let slow_a = Stack::create(
        Arc::clone(&executor),
        Arc::clone(&planner),
        slow_backend.clone(),
        ToySpec {
            bonus: 10,
            delay_ms: 120,
        },
    )
    .unwrap();
    let slow_b = Stack::create(
        Arc::clone(&executor),
        Arc::clone(&planner),
        slow_backend,
        ToySpec {
            bonus: 20,
            delay_ms: 120,
        },
    )
    .unwrap();
    let fast = Stack::create(
        executor,
        planner,
        fast_backend,
        ToySpec {
            bonus: 30,
            delay_ms: 10,
        },
    )
    .unwrap();

    let slow_job_a = slow_a.dispatch(vec![1]).unwrap();
    let slow_job_b = slow_b.dispatch(vec![1]).unwrap();
    let fast_job = fast.dispatch(vec![1]).unwrap();

    let start = Instant::now();
    let fast_out = fast.collect(fast_job).unwrap();
    let fast_elapsed = start.elapsed();

    assert_eq!(fast_out, vec![31]);
    assert!(
        fast_elapsed < Duration::from_millis(80),
        "fast backend was blocked too long: {:?}",
        fast_elapsed
    );

    assert_eq!(slow_a.collect(slow_job_a).unwrap(), vec![11]);
    assert_eq!(slow_b.collect(slow_job_b).unwrap(), vec![21]);
}

#[test]
fn shutdown_fails_waiting_backend_jobs_instead_of_hanging() {
    let executor = Arc::new(BraidExecutor::new(2));
    let planner = Arc::new(ToyPlanner);
    let backend = executor.register_backend(Arc::new(ToyBackend), BackendConfig { lane_count: 1 });

    let stack_a = Stack::create(
        Arc::clone(&executor),
        Arc::clone(&planner),
        backend.clone(),
        ToySpec {
            bonus: 10,
            delay_ms: 150,
        },
    )
    .unwrap();
    let stack_b = Stack::create(
        Arc::clone(&executor),
        planner,
        backend,
        ToySpec {
            bonus: 20,
            delay_ms: 150,
        },
    )
    .unwrap();

    let job_a = stack_a.dispatch(vec![1]).unwrap();
    let job_b = stack_b.dispatch(vec![1]).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while Instant::now() < deadline {
        if stack_a.poll(job_a) == JobStatus::Running && stack_b.poll(job_b) == JobStatus::Running {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    thread::sleep(Duration::from_millis(20));

    executor.shutdown();

    assert_eq!(stack_b.poll(job_b), JobStatus::Failed);
    assert!(matches!(
        stack_b.collect(job_b),
        Err(BraidError::ExecutorShutdown)
    ));
}
