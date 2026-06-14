# `braids` - parallel ProcGen made "easy"

There are three big abstractions that make this all possible:

- `PlannerBackend` holds the actual logic and spits out an execution plan
- `ComputeBackend` tackles the execution. We provide a tiny `CpuComputeBackend` to get you started. There are plans for
  GPU support too.
- `Stack` holds a self-contained pipeline. You can have any number of stacks sharing the same planners and compute
  backends.

We also provide a fastnoise-lite impl that you can use to play around.
Using braids with 8 lanes and 8 workers you can get around `9.83x` speed-up compared to just using fastnoise-lite.

> [!NOTE]
> Measured on an M4 Pro, use `cargo run -p braids --example lanes_showcase --release` to see how far you can get
> on your machine.

## Quickstart

First you need to implement the planner trait for your pcg backend, I know this can be a hefty task but you only gotta do it once
right?

Then you can set up a basic executor as follows (taken from [examples/lanes_showcase.rs](./examples/lanes_showcase.rs)).

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

> [!IMPORTANT]
> For tiny serial work where queueing would dominate compute, use inline resolution instead:
>
> ```rust
> let summary = stack.resolve_one_inline(query)?;
> let summaries = stack.resolve_inline(&queries)?;
>
> let mut inline = braids::InlineContext::default();
> let summary = stack.resolve_one_inline_with(query, &mut inline)?;
> ```

If you want deeper lifecycle details, see [docs/architecture.md](./docs/architecture.md).

## Docs

- [docs/architecture.md](./docs/architecture.md): core concepts, job flow, versioning, and memory model
- [docs/writing_planners_and_backends.md](./docs/writing_planners_and_backends.md): how to implement a planner or backend
- [examples/terrain_stack.rs](./examples/terrain_stack.rs): smallest real stack example
- [examples/lanes_showcase.rs](./examples/lanes_showcase.rs): direct serial vs braids parallel showcase

## FAQ

### Does this only work for PCG?

Nope. It's just the intended use case. You do what you want with it.
