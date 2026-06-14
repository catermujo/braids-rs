//! `braids` is a planner-agnostic, compute-agnostic execution core.
//!
//! ## Start here:
//!
//! - [`PlannerBackend`]: authoring model, mutable state, compile, encode, decode
//! - [`ComputeBackend`]: prepare compiled plans and run stages
//! - [`Stack`]: one typed runtime instance built from one planner, one backend, and one live state
//! - [`BraidExecutor`]: shared async worker pool used by many stacks
//!
//! Project-level docs:
//!
//! - repository guide: `README.md`
//! - architecture guide: `docs/architecture.md`
//!
//! # Example
//!
//! ```rust
//! use std::sync::Arc;
//!
//! use braids::{BackendConfig, BraidExecutor, Stack};
//! use braid_fastnoise::{scenarios, ChunkQuery, FastNoisePlanner, make_cpu_backend};
//!
//! let executor = Arc::new(BraidExecutor::new(4));
//! let backend = executor.register_backend(Arc::new(make_cpu_backend()), BackendConfig { lane_count: 4 });
//!
//! let stack = Stack::create(
//!     Arc::clone(&executor),
//!     Arc::new(FastNoisePlanner),
//!     backend,
//!     scenarios::terrain_height_2d(),
//! )
//! .expect("stack");
//!
//! let job = stack
//!     .dispatch(vec![ChunkQuery::Grid2D {
//!         width: 8,
//!         height: 8,
//!         origin: [0.0, 0.0],
//!         step: [1.0, 1.0],
//!     }])
//!     .expect("dispatch");
//! let summaries = stack.collect(job).expect("collect");
//! let summary = &summaries[0];
//!
//! assert_eq!(summary.samples, 64);
//! assert!(summary.min <= summary.max);
//! assert!(summary.mean.is_finite());
//! ```

mod buffer_pool;
mod compute;
mod cpu;
mod error;
mod executor;
mod job;
mod pipeline;
mod planner;
mod scratch;
mod slot_table;
mod stack;
mod version;

pub use compute::ComputeBackend;
pub use cpu::{CpuComputeBackend, CpuKernel, CpuKernelFactory};
pub use error::{BraidError, BraidResult};
pub use executor::{BackendConfig, BackendHandle, BraidExecutor};
pub use job::{CancelFlag, JobPacket, JobStatus};
pub use pipeline::{
    BufferAccess, BufferBinding, BufferLayout, BufferSlot, BufferSpec, CompiledPlan, DispatchHint,
    ElementKind, JobId, KernelKind, KernelSpec, PipelineShape, StageSpec, StaticBuffer,
    StaticBufferSet, VersionId,
};
pub use planner::PlannerBackend;
pub use scratch::{BatchScratch, ComputeScratch, PlannerScratch};
pub use slot_table::{SlotKey, SlotTable};
pub use stack::{InlineContext, Stack};
