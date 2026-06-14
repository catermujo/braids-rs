# Backlog

## Deferred

- Optional `braids::dag` / `braids::resolve` helper for named dependency resolution.
  Scope: stable handle interning, duplicate ID detection, missing reference detection, topo sort, cycle detection, reachable-from-root traversal.
  Non-goals: owning domain node semantics, output typing, dimension checks, slot allocation, payload encoding, or full graph compilation.
  Why: `fastnoise` currently hand-rolls generic DAG compile work, while real-world procgen targets stay mostly flat and should not be forced into a graph-first core API.

- Device-resident packet path for GPU backends.
  Goal: support real GPU residency without moving planner query/result semantics into the compute backend.
  Shape:
  - Keep current `JobPacket` as host-side packet only.
  - Add backend-associated `ResidentPacket` for device buffers, staging handles, and other backend-native per-job state.
  - Add backend methods to allocate resident packets, upload host packet contents, run stages on resident packets, and download selected slots back into a host packet.
  - Add planner method to declare which slots must come back to host for decode, so GPU backends do not need to download every buffer.
  - Keep `prepare` separate from per-job resident packet allocation and reuse.
  Runtime flow:
  - planner encodes compact query descriptors into host packet
  - backend uploads host packet into resident packet
  - backend runs all stages on resident packet
  - backend downloads decode-required slots into host packet
  - planner decodes host packet into final resolutions
  Reuse:
  - add per-stack pool for retired resident packets, similar to prepared-state reuse
  - let backends reuse device buffers and staging storage across jobs
  GPU guidance:
  - planners should encode compact descriptors, not eagerly expanded giant arrays
  - heavy materialization should happen in backend/device stages
  Later extension:
  - resident-to-resident stack chaining, so one stack can feed another without a host roundtrip
  Non-goals:
  - no fake `slice::<T>()` API over device memory
  - no arbitrary Rust user types as core packet storage
  - no backend ownership of planner semantic decode rules
  Why: current host-only packet model is fine for CPU, but a proper GPU path needs explicit host/device boundaries instead of pretending both memory domains behave the same way.
