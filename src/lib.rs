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
pub use executor::BraidExecutor;
pub use job::{CancelFlag, JobPacket, JobStatus};
pub use pipeline::{
    BufferAccess, BufferBinding, BufferLayout, BufferSpec, CompiledPlan, DispatchHint, ElementKind,
    JobId, KernelSpec, PipelineShape, StageSpec, StaticBuffer, StaticBufferSet, VersionId,
};
pub use planner::PlannerBackend;
pub use scratch::{BatchScratch, ComputeScratch, PlannerScratch};
pub use slot_table::{SlotKey, SlotTable};
pub use stack::Stack;

#[cfg(test)]
mod tests;
