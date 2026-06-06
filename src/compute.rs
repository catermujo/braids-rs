use crate::error::BraidResult;
use crate::job::{CancelFlag, JobPacket};
use crate::pipeline::{CompiledPlan, StageSpec};
use crate::scratch::ComputeScratch;

/// Compute-side interface for preparing and running compiled stages.
///
/// A compute backend knows nothing about planner semantics. It only understands the generic
/// pipeline shape, kernel kinds, payloads, and packet buffers emitted by a planner.
pub trait ComputeBackend: Send + Sync + 'static {
    /// Backend-specific prepared state derived from a compiled plan.
    type Prepared: Send + Sync + 'static;

    /// Prepare backend execution state for a compiled plan.
    ///
    /// `reuse` may contain a retired prepared object from an older version of the same stack.
    fn prepare<M: Send + Sync + 'static>(
        &self,
        plan: &CompiledPlan<M>,
        reuse: Option<Self::Prepared>,
        scratch: &mut ComputeScratch,
    ) -> BraidResult<Self::Prepared>;

    /// Run one compiled stage against a mutable job packet.
    ///
    /// Backends should check `cancel` cooperatively when stage work can be long-running.
    fn run_stage(
        &self,
        prepared: &Self::Prepared,
        stage_index: usize,
        stage: &StageSpec,
        packet: &mut JobPacket,
        cancel: &CancelFlag,
    ) -> BraidResult<()>;

    /// Run one compiled stage for the single-query inline path.
    ///
    /// Override this when one-query execution can skip batch-oriented packet reads, loops, or
    /// range handling. The default delegates to [`Self::run_stage`].
    fn run_one_stage(
        &self,
        prepared: &Self::Prepared,
        stage_index: usize,
        stage: &StageSpec,
        packet: &mut JobPacket,
        cancel: &CancelFlag,
    ) -> BraidResult<()> {
        self.run_stage(prepared, stage_index, stage, packet, cancel)
    }
}
