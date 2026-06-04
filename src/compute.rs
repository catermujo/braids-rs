use crate::error::BraidResult;
use crate::job::{CancelFlag, JobPacket};
use crate::pipeline::{CompiledPlan, StageSpec};
use crate::scratch::ComputeScratch;

pub trait ComputeBackend: Send + Sync + 'static {
    type Prepared: Send + Sync + 'static;

    fn prepare<M: Send + Sync + 'static>(
        &self,
        plan: &CompiledPlan<M>,
        reuse: Option<Self::Prepared>,
        scratch: &mut ComputeScratch,
    ) -> BraidResult<Self::Prepared>;

    fn run_stage(
        &self,
        prepared: &Self::Prepared,
        stage_index: usize,
        stage: &StageSpec,
        packet: &mut JobPacket,
        cancel: &CancelFlag,
    ) -> BraidResult<()>;
}
