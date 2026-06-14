# Braids Architecture

This file explains the runtime shape of `braids` without planner-specific words.

## Goals

`braids` tries to do four things:

- keep planner logic separate from execution logic
- keep compute backend logic separate from executor logic
- allow many stacks to share one executor and one backend instance
- support versioned recompile/swap without breaking in-flight jobs

It is not a scene graph, not a game framework, and not a built-in cross-stack scheduler.

## Main Parts

### `PlannerBackend`

Planner owns:

- authoring spec type
- mutable planner state
- change application rules
- compilation from state into `CompiledPlan`
- query encoding into `JobPacket`
- result decoding out of `JobPacket`

Planner is where domain meaning lives.

### `ComputeBackend`

Backend owns:

- backend-specific prepared state
- stage execution
- any device/runtime-specific caching or reuse

Backend does not own query semantics. It only runs compiled stage kernels.

### `Stack<P, C>`

A stack is one typed runtime instance:

- one planner instance
- one shared backend handle
- one mutable planner state
- one current frozen compiled version

Stacks are the unit you create, update, recompile, and dispatch against.

### `BraidExecutor`

Executor owns worker threads and shared scheduling state.

It runs jobs from many stacks in parallel.

Backends can still declare limited capacity through lane counts. Executor respects that instead of assuming all backend work can run fully in parallel.

## Lifecycle

## Create

`Stack::create(...)` does:

1. planner builds initial mutable state from `Spec`
2. planner compiles that state into a `CompiledPlan`
3. backend prepares backend-specific execution state
4. stack stores the frozen version as current

After that, stack is ready to dispatch queries.

## Dispatch

`Stack::dispatch(...)` does not run the whole job inline.

It creates a job record, snapshots the current frozen version, and schedules async work in steps:

1. encode query batch into `JobPacket`
2. run compiled stages one by one through backend lanes
3. decode results back into planner `Resolution` values

Important:

- in-flight jobs hold `Arc` to the frozen version they started with
- later recompiles do not mutate that version
- job order inside one stack is not guaranteed unless the backend lane count and workload effectively serialize it

## Inline Resolve

`Stack::resolve_inline(&queries)` and `Stack::resolve_one_inline(...)` run the same compiled planner and
backend path directly on the caller thread.

For repeated low-latency calls, reuse `InlineContext` through
`resolve_inline_with(...)` / `resolve_one_inline_with(...)` so inline path avoids pool mutexes and
reuses its packet/scratch directly.

Use this when:

- workload is tiny
- serial latency matters more than async overlap
- queueing and wakeup cost would dominate real compute

Inline resolve still:

1. snapshots the current frozen version
2. encodes queries into a `JobPacket`
3. runs every stage through the backend
4. decodes planner `Resolution` values

What it skips is async job machinery:

- no job table
- no executor queue hops
- no backend lane scheduling
- no condvar-based `collect()`

## Update And Recompile

There are three useful update paths:

- `apply(changes)`: mutate planner state only
- `replace(spec)`: reset planner state from a fresh spec
- `recompile()`: compile current state into a new frozen version
- `update(changes)`: build a new planner state, compile it, then swap it in if compile succeeds

`update(...)` is the safest path when planner wants transactional behavior. Current state stays live if new compile fails.

## Versioned Swap

When a recompile succeeds:

1. planner returns a new `CompiledPlan`
2. backend returns a new prepared state
3. stack wraps them in a new frozen version
4. stack swaps current version pointer

Old jobs keep using the old version until they finish. New jobs use the new version after swap.

This is why planner state is separate from frozen compiled versions.

## Memory Model

`braids` tries to reuse memory instead of freeing everything immediately.

Things that stay live:

- one current mutable planner state
- one current frozen compiled version
- any old frozen versions still referenced by in-flight jobs
- scratch pools
- packet pools
- backend prepared reuse pools

Things that do not accumulate forever:

- old planner states are not kept as history
- old frozen versions drop when the last in-flight job releases them

Things that may stay at high-water mark:

- planner-owned vector capacity
- pooled `JobPacket`s and scratch buffers
- pooled backend prepared state

This is deliberate. `braids` prefers reuse and low churn over aggressive shrinking.

## Concurrency Model

Executor and backend capacity are different things.

Executor:

- decides how many worker threads can run tasks at once

Backend lanes:

- decide how many backend stage executions can run at once for one shared backend instance

This matters because some backends are effectively serialized internally. Explicit lane counts keep one slow backend from poisoning unrelated work on other backends.

There is also separate prepare-lane control for recompile/prepare pressure.

## What Braid Does Not Do

`braids` core does not:

- define your planner schema
- force use of `CpuComputeBackend`
- provide a built-in heterogeneous stack registry
- provide a built-in cross-stack dependency scheduler
- guarantee low memory after workload spikes

Those choices stay with planner, backend, or app layer.

## Practical Advice

- If your planner state wants shared external data, store `Arc<T>`, not borrowed references.
- If you need transactional updates, prefer `updated_state(...)` + `update(...)`.
- If a backend is really single-lane, declare that explicitly.
- If memory spikes matter more than churn, add planner/backend-specific compaction or trim operations later.
