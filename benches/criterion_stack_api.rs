use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use braids::{
    BackendConfig, BraidError, BraidExecutor, BraidResult, BufferAccess, BufferBinding,
    BufferLayout, BufferSlot, BufferSpec, CancelFlag, CompiledPlan, ComputeBackend, ComputeScratch,
    DispatchHint, ElementKind, JobPacket, KernelKind, KernelSpec, PipelineShape, PlannerBackend,
    PlannerScratch, Stack, StageSpec, VersionId,
};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

const DATA_SLOT: BufferSlot = BufferSlot(0);
const WORKER_KIND: KernelKind = KernelKind(0xC0FF_00D0);

#[derive(Clone, Copy)]
struct BenchSpec {
    seed: u32,
}

#[derive(Clone, Copy)]
struct BenchState {
    seed: u32,
}

#[derive(Default)]
struct BenchBackend;

struct BenchPlanner;

impl PlannerBackend for BenchPlanner {
    type Spec = BenchSpec;
    type State = BenchState;
    type Change = u32;
    type Query = u32;
    type Resolution = u32;
    type PlannerMeta = ();

    fn init_state(&self, spec: &Self::Spec) -> BraidResult<Self::State> {
        Ok(BenchState { seed: spec.seed })
    }

    fn reset_state(&self, state: &mut Self::State, spec: &Self::Spec) -> BraidResult<()> {
        state.seed = spec.seed;
        Ok(())
    }

    fn apply(&self, state: &mut Self::State, changes: &[Self::Change]) -> BraidResult<()> {
        if let Some(seed) = changes.first() {
            state.seed = *seed;
        }
        Ok(())
    }

    fn updated_state(
        &self,
        state: &Self::State,
        changes: &[Self::Change],
    ) -> BraidResult<Self::State> {
        let mut next = *state;
        if let Some(seed) = changes.first() {
            next.seed = *seed;
        }
        Ok(next)
    }

    fn compile(
        &self,
        state: &Self::State,
        scratch: &mut PlannerScratch,
    ) -> BraidResult<CompiledPlan<Self::PlannerMeta>> {
        scratch.reset();
        let mut payload = Vec::<u8>::new();
        payload.extend_from_slice(&state.seed.to_le_bytes());

        Ok(CompiledPlan {
            pipeline: PipelineShape {
                buffers: vec![BufferSpec {
                    slot: DATA_SLOT,
                    element_kind: ElementKind::U32,
                    layout: BufferLayout::PerQueryScalar,
                }],
                stages: vec![StageSpec {
                    kernels: vec![KernelSpec {
                        kind_id: WORKER_KIND,
                        payload: Arc::from(payload),
                        bindings: vec![BufferBinding {
                            slot: DATA_SLOT,
                            access: BufferAccess::ReadWrite,
                        }],
                        dispatch: DispatchHint::WholeBatch,
                    }],
                }],
            },
            static_buffers: Vec::new(),
            planner_meta: (),
        })
    }

    fn encode_batch(
        &self,
        _plan: &CompiledPlan<Self::PlannerMeta>,
        queries: &[Self::Query],
        packet: &mut JobPacket,
        _scratch: &mut braids::BatchScratch,
    ) -> BraidResult<()> {
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
        Ok(packet.slice::<u32>(DATA_SLOT)?.to_vec())
    }
}

impl ComputeBackend for BenchBackend {
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
        packet: &mut JobPacket,
        cancel: &CancelFlag,
    ) -> BraidResult<()> {
        if cancel.is_cancelled() {
            return Err(BraidError::Cancelled);
        }
        let values = packet.slice_mut::<u32>(DATA_SLOT)?;
        for value in values {
            *value = value.wrapping_add(17);
        }
        Ok(())
    }
}

fn make_queries(seed: u32, batch_size: usize) -> Vec<u32> {
    (0..batch_size as u32).map(|value| value ^ seed).collect()
}

fn make_stack() -> BraidResult<Stack<BenchPlanner, BenchBackend>> {
    let executor = Arc::new(BraidExecutor::new(2));
    let planner = Arc::new(BenchPlanner);
    let backend =
        executor.register_backend(Arc::new(BenchBackend), BackendConfig { lane_count: 1 });
    Stack::create(executor, planner, backend, BenchSpec { seed: 0xBEEF })
}

fn bench_stack_apis(c: &mut Criterion) {
    let mut group = c.benchmark_group("stack_api");
    group.throughput(Throughput::Elements(256));
    group.measurement_time(Duration::from_secs(2));

    group.bench_function("stack_resolve_inline_256", |b| {
        let stack = make_stack().expect("stack setup");
        let mut inline = Default::default();
        let queries = make_queries(0xA5, 256);

        b.iter(|| {
            let result = stack
                .resolve_inline_with(queries.as_slice(), &mut inline)
                .expect("resolve inline");
            black_box(result[0]);
        });
    });

    group.bench_function("stack_resolve_one_inline_256", |b| {
        let stack = make_stack().expect("stack setup");
        let mut inline = Default::default();
        let query = 42u32;

        b.iter(|| {
            let result = stack
                .resolve_one_inline_ref_with(&query, &mut inline)
                .expect("resolve one");
            black_box(result);
        });
    });

    group.bench_with_input("stack_dispatch_collect_256", &256usize, |b, &batch_size| {
        let stack = make_stack().expect("stack setup");
        b.iter_batched(
            || make_queries(0x1, batch_size),
            |queries| {
                let values = stack.dispatch_collect(&queries).expect("dispatch_collect");
                black_box(values[0]);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("stack_update_and_version", |b| {
        let stack = make_stack().expect("stack setup");

        b.iter(|| {
            let version: VersionId = stack.update(&[0x9]).expect("update");
            black_box(version);
        });
    });

    group.bench_function("stack_current_version", |b| {
        let stack = make_stack().expect("stack setup");

        b.iter_batched(
            || (),
            |_| {
                let version = stack.current_version_id().expect("version");
                black_box(version)
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(stack_api_benches, bench_stack_apis);
criterion_main!(stack_api_benches);
