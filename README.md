# `braid`

`braid` is a planner-agnostic, compute-agnostic execution core.

Main shape:
- shared `BraidExecutor`
- shared backend instances
- typed `Stack<P, C>` handles
- versioned recompile/swap
- async stack-parallel job execution

`braid` core does not require the starter CPU backend. That backend exists only to get users moving fast.

## Workspace Pieces

- `braid`: core executor / stack / traits
- `braid-fastnoise`: real FastNoise worldgen adapter with vendored upstream source

## Docs

- [docs/architecture.md](./docs/architecture.md): core concepts, job flow, versioning, and memory model
- [docs/writing_planners_and_backends.md](./docs/writing_planners_and_backends.md): how to implement a planner or backend
- `cargo doc --no-deps`: API docs for `PlannerBackend`, `ComputeBackend`, `Stack`, and `BraidExecutor`
- [examples/terrain_stack.rs](./examples/terrain_stack.rs): smallest real stack example
- [examples/lanes_showcase.rs](./examples/lanes_showcase.rs): direct serial vs braid parallel showcase

## Quick Start

Tiny stack example:

```sh
cargo run -p braid --example terrain_stack
```

Parallel lanes showcase:

```sh
cargo run -p braid --example lanes_showcase --release
```

Fair serial overhead bench:

```sh
cargo bench --bench fastnoise_worldgen -- 120
```

## Why Use It

Two useful truths from the FastNoise workload:

1. `braid` core overhead is tiny next to real chunk generation compute.
2. Multi-lane speedup is simple: register backend with more lanes, dispatch more jobs.

## Core Model

`braid` keeps four roles separate:

- `PlannerBackend`: owns your authoring model, mutable planner state, compilation, and query/result codecs
- `ComputeBackend`: owns execution of compiled stages on some device or runtime
- `Stack<P, C>`: one typed compiled instance built from one planner, one backend, and one live mutable state
- `BraidExecutor`: shared async worker pool that runs jobs from many stacks

Typical flow:

1. Create one shared executor.
2. Register one or more shared backends with lane counts.
3. Create one or more typed stacks from planner + backend + spec.
4. Dispatch query batches to stacks.
5. Apply changes and recompile when data changes.

If you want deeper lifecycle details, see [docs/architecture.md](./docs/architecture.md).

## Showcase: Direct Serial vs Braid Parallel

Example: [examples/lanes_showcase.rs](./examples/lanes_showcase.rs)

Core shape:

```rust
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

for query in &queries {
    jobs.push(stack.dispatch(vec![query.clone()])?);
}
```

For tiny serial work where queueing would dominate compute, use inline resolution instead:

```rust
let summary = stack.resolve_one_inline(query)?;
let summaries = stack.resolve_inline(&queries)?;

let mut inline = braid::InlineContext::default();
let summary = stack.resolve_one_inline_with(query, &mut inline)?;
```

Measured on this machine on `2026-06-03`:

```text
terrain lanes showcase: chunks=32 lanes=8 direct_serial_ms=106.901 braid_parallel_ms=12.971 speedup_x=8.24 checksum=13099885939998993987
```

So same terrain work, same planner, same backend family:
- direct serial baseline: `106.901 ms`
- `braid` with `8` lanes: `12.971 ms`
- speedup: about `8.24x`

## Fair Serial Overhead

This bench uses:
- `1` worker
- `1` backend lane
- direct serial baseline
- serial `braid` path

So these numbers show overhead, not lane scaling.

Source: [benches/fastnoise_worldgen.rs](./benches/fastnoise_worldgen.rs)

Latest run from:

```sh
cargo bench --bench fastnoise_worldgen -- 120
```

### Direct Serial vs Braid Serial

| Case | Direct Serial | Braid Serial | Ratio |
| --- | ---: | ---: | ---: |
| Terrain query | `2.625 ms` | `2.783 ms` | `1.060x` |
| Voxel query | `19.311 ms` | `19.157 ms` | `0.992x` |
| Mixed terrain+voxel | `10.974 ms/query` | `11.043 ms/query` | `1.006x` |
| Terrain update+run | `2.606 ms` | `2.633 ms` | `1.010x` |
| Dependency chain | `7.591 ms/query` | `7.712 ms/query` | `1.016x` |

### Terrain Micro Split

| Piece | Time |
| --- | ---: |
| Encode only | `110 ns` |
| Decode only | `17.1 us` |
| Empty stack roundtrip | `8.3 us` |
| Terrain compute only | `2.496 ms` |

Read:
- almost all time is real compute
- stack/executor roundtrip is tiny
- encode/decode are small
- if something is slow, it is mostly backend/kernel work, not `braid` core machinery

## Examples

- [examples/terrain_stack.rs](./examples/terrain_stack.rs): tiny single-stack terrain run
- [examples/lanes_showcase.rs](./examples/lanes_showcase.rs): direct serial vs braid parallel

## Notes

- Numbers are machine-specific.
- `CpuComputeBackend` is only a starter helper.
- `braid` stays generic over `ComputeBackend`.
- Vendored FastNoiseLite reference and license stay in `braid/fastnoise`:
  - `README.fastnoise-lite.md`
  - `LICENSE.fastnoise-lite.md`
