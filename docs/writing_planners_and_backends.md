# Writing Planners And Backends

This guide shows the smallest useful shape for extending `braids`.

`braids` has two extension seams:

- `PlannerBackend`: owns domain data, compile, encode, decode
- `ComputeBackend`: owns prepared execution state and stage execution

Most users should start by writing a planner first and using `CpuComputeBackend` as a temporary backend helper.

## Mental Model

Planner answers:

- what does authored data look like?
- how do changes affect that data?
- what buffers and kernels should exist?
- how are queries encoded?
- how are results decoded?

Backend answers:

- how is a compiled plan prepared for execution?
- how does one stage run?
- how much backend parallelism is safe?

## Minimal Planner Skeleton

```rust
use braids::{
    BatchScratch, BraidResult, CompiledPlan, JobPacket, PlannerBackend, PlannerScratch,
};

struct MyPlanner;

struct MySpec;
struct MyState;
struct MyChange;
struct MyQuery;
struct MyResolution;
struct MyMeta;

impl PlannerBackend for MyPlanner {
    type Spec = MySpec;
    type State = MyState;
    type Change = MyChange;
    type Query = MyQuery;
    type Resolution = MyResolution;
    type PlannerMeta = MyMeta;

    fn init_state(&self, spec: &Self::Spec) -> BraidResult<Self::State> {
        todo!()
    }

    fn reset_state(&self, state: &mut Self::State, spec: &Self::Spec) -> BraidResult<()> {
        todo!()
    }

    fn apply(&self, state: &mut Self::State, changes: &[Self::Change]) -> BraidResult<()> {
        todo!()
    }

    fn updated_state(
        &self,
        state: &Self::State,
        changes: &[Self::Change],
    ) -> BraidResult<Self::State> {
        let mut next = self.init_state(&MySpec)?;
        let _ = (state, changes, &mut next);
        todo!()
    }

    fn compile(
        &self,
        state: &Self::State,
        scratch: &mut PlannerScratch,
    ) -> BraidResult<CompiledPlan<Self::PlannerMeta>> {
        let _ = (state, scratch);
        todo!()
    }

    fn encode_batch(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        queries: &[Self::Query],
        packet: &mut JobPacket,
        scratch: &mut BatchScratch,
    ) -> BraidResult<()> {
        let _ = (plan, queries, packet, scratch);
        todo!()
    }

    fn decode_batch(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        packet: &JobPacket,
    ) -> BraidResult<Vec<Self::Resolution>> {
        let _ = (plan, packet);
        todo!()
    }
}
```

## Minimal Backend Skeleton

```rust
use braids::{
    BraidResult, CancelFlag, CompiledPlan, ComputeBackend, ComputeScratch, JobPacket, StageSpec,
};

struct MyBackend;
struct MyPrepared;

impl ComputeBackend for MyBackend {
    type Prepared = MyPrepared;

    fn prepare<M: Send + Sync + 'static>(
        &self,
        plan: &CompiledPlan<M>,
        reuse: Option<Self::Prepared>,
        scratch: &mut ComputeScratch,
    ) -> BraidResult<Self::Prepared> {
        let _ = (plan, reuse, scratch);
        todo!()
    }

    fn run_stage(
        &self,
        prepared: &Self::Prepared,
        stage_index: usize,
        stage: &StageSpec,
        packet: &mut JobPacket,
        cancel: &CancelFlag,
    ) -> BraidResult<()> {
        let _ = (prepared, stage_index, stage, packet, cancel);
        todo!()
    }
}
```

## Recommended First Step

If you do not want to build a custom backend yet:

1. define planner structs
2. compile into `KernelSpec`s and packet buffers
3. use `CpuComputeBackend`
4. register one factory per kernel kind

That gives a real end-to-end stack quickly.

## Building A Plan

Most planners should build `CompiledPlan` directly with explicit literals:

1. declare buffer slots
2. add stages
3. add static buffers if needed
4. call `plan.validate()?`

This keeps validation in one place.

## Query Encoding

Use `JobPacket` as the only mutable per-job data container.

Typical encode flow:

1. assign `packet.query_count = ...`
2. size buffers with `ensure::<T>(...)`
3. fill those buffers in planner-owned layout

Keep packet layout deterministic. Decoder should not guess.

If encode needs several temporary arrays of the same primitive type, use scratch checkout helpers
instead of allocating fresh locals every call:

```rust
let mut entry_ids = scratch.spare_u32s.checkout();
let mut required_ids = scratch.spare_u32s.checkout();

// fill and use temp vectors here

scratch.spare_u32s.give_back(entry_ids);
scratch.spare_u32s.give_back(required_ids);
```

Keep the direct `scratch.u32s` / `scratch.u64s` / `scratch.f32s` fields for the one primary
scratch vector. Use checkout helpers when one planner needs several same-typed temps at once.

## Transactional Updates

If planner changes can fail during compile, prefer:

- implement `updated_state(...)`
- use `Stack::update(...)`

That way old state and old frozen version remain live if new compile fails.

## Backend Lane Advice

When registering a backend:

- use `lane_count = 1` if backend is effectively serialized
- use more lanes only if concurrent stage execution is truly safe
- consider `register_backend_with_prepare_lanes(...)` if prepare/recompile work should be capped separately

## State And Lifetimes

Planner state is `'static` in practice because stacks and queued jobs can outlive the call site that created them.

If planner needs shared external data:

- use `Arc<T>`
- do not store borrowed references in planner state

## Memory Reuse Advice

Good planner/backend behavior:

- reuse vector capacity
- reuse scratch buffers
- reuse prepared backend state when possible

`braids` is biased toward low churn, not toward aggressive shrinking after peaks.

## Smallest Real Example

See:

- [fastnoise/examples/terrain_stack.rs](../fastnoise/examples/terrain_stack.rs)
- [fastnoise/src/lib.rs](../fastnoise/src/lib.rs)
