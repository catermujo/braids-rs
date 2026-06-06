//! `braid` is a planner-agnostic, compute-agnostic execution core.
//!
//! Start here:
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

#[cfg(test)]
mod tests;
