use crate::compute::ComputeBackend;
use crate::error::BraidResult;
use crate::job::{CancelFlag, JobPacket};
use crate::pipeline::{
    BufferAccess, BufferBinding, BufferData, BufferLayout, BufferSpec, CompiledPlan, DispatchHint,
    ElementKind, KernelSpec, PipelineShape, StageSpec, StaticBuffer,
};
use crate::planner::PlannerBackend;
use crate::scratch::{BatchScratch, ComputeScratch, PlannerScratch};
use crate::{BackendConfig, BraidError, BraidExecutor, Stack};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const TOY_INPUT_SLOT: u16 = 0;
const TOY_OUTPUT_SLOT: u16 = 1;
const TOY_BONUS_SLOT: u16 = 2;
const TOY_KIND: u32 = 77;

#[derive(Default)]
struct ToyPlanner;

#[derive(Clone)]
struct ToySpec {
    bonus: u32,
    delay_ms: u64,
}

#[derive(Clone)]
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
        Ok(CompiledPlan {
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
                    BufferSpec {
                        slot: TOY_BONUS_SLOT,
                        element_kind: ElementKind::U32,
                        layout: BufferLayout::Dynamic,
                    },
                ],
                stages: vec![StageSpec {
                    kernels: vec![KernelSpec {
                        kind_id: TOY_KIND,
                        payload: Arc::from(scratch.bytes.clone()),
                        bindings: vec![
                            BufferBinding {
                                slot: TOY_INPUT_SLOT,
                                access: BufferAccess::Read,
                            },
                            BufferBinding {
                                slot: TOY_OUTPUT_SLOT,
                                access: BufferAccess::Write,
                            },
                            BufferBinding {
                                slot: TOY_BONUS_SLOT,
                                access: BufferAccess::Read,
                            },
                        ],
                        dispatch: DispatchHint::WholeBatch,
                    }],
                }],
            },
            static_buffers: vec![StaticBuffer {
                slot: TOY_BONUS_SLOT,
                data: BufferData::U32(vec![state.bonus]),
            }],
            planner_meta: (),
        })
    }

    fn encode_batch(
        &self,
        _plan: &CompiledPlan<Self::PlannerMeta>,
        queries: &[Self::Query],
        packet: &mut JobPacket,
        _scratch: &mut BatchScratch,
    ) -> BraidResult<()> {
        packet.set_query_count(queries.len());
        packet
            .ensure_u32(TOY_INPUT_SLOT, queries.len())
            .copy_from_slice(queries);
        packet.ensure_u32(TOY_OUTPUT_SLOT, queries.len()).fill(0);
        Ok(())
    }

    fn decode_batch(
        &self,
        _plan: &CompiledPlan<Self::PlannerMeta>,
        packet: &JobPacket,
    ) -> BraidResult<Vec<Self::Resolution>> {
        Ok(packet.u32(TOY_OUTPUT_SLOT)?.to_vec())
    }
}

impl ComputeBackend for ToyBackend {
    type Prepared = ToyPrepared;

    fn prepare<M: Send + Sync + 'static>(
        &self,
        plan: &CompiledPlan<M>,
        _reuse: Option<Self::Prepared>,
        _scratch: &mut ComputeScratch,
    ) -> BraidResult<Self::Prepared> {
        let payload = &plan.pipeline.stages[0].kernels[0].payload;
        let delay_ms = u64::from_le_bytes(payload[0..8].try_into().unwrap());
        Ok(ToyPrepared { bonus: 0, delay_ms })
    }

    fn run_stage(
        &self,
        prepared: &Self::Prepared,
        _stage_index: usize,
        _stage: &StageSpec,
        packet: &mut JobPacket,
        cancel: &CancelFlag,
    ) -> BraidResult<()> {
        thread::sleep(Duration::from_millis(prepared.delay_ms));
        if cancel.is_cancelled() {
            return Err(BraidError::Cancelled);
        }
        let inputs = packet.u32(TOY_INPUT_SLOT)?.to_vec();
        let bonus = packet
            .u32(TOY_BONUS_SLOT)?
            .first()
            .copied()
            .ok_or_else(|| BraidError::from("missing toy bonus buffer value"))?;
        let mut outputs = vec![0u32; inputs.len()];
        for (ix, value) in inputs.iter().enumerate() {
            outputs[ix] = value + bonus + prepared.bonus;
        }
        packet
            .u32_mut(TOY_OUTPUT_SLOT)?
            .copy_from_slice(outputs.as_slice());
        Ok(())
    }
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
    let new_version = stack.update(vec![ToyChange::SetBonus(100)]).unwrap();
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
    let error = stack
        .update(vec![ToyChange::SetBonus(u32::MAX)])
        .unwrap_err();
    assert!(error.to_string().contains("u32::MAX"));
    assert_eq!(stack.current_version_id().unwrap(), version_before);

    let version_after = stack.recompile().unwrap();
    assert!(version_after > version_before);
    let job = stack.dispatch(vec![10]).unwrap();
    assert_eq!(stack.collect(job).unwrap(), vec![11]);
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
        if stack_a.poll(job_a) == crate::JobStatus::Running
            && stack_b.poll(job_b) == crate::JobStatus::Running
        {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    thread::sleep(Duration::from_millis(20));

    executor.shutdown();

    assert_eq!(stack_b.poll(job_b), crate::JobStatus::Failed);
    assert!(matches!(
        stack_b.collect(job_b),
        Err(BraidError::ExecutorShutdown)
    ));
}
