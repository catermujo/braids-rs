use crate::error::BraidResult;
use crate::job::JobPacket;
use crate::pipeline::CompiledPlan;
use crate::scratch::{BatchScratch, PlannerScratch};

pub trait PlannerBackend: Send + Sync + 'static {
    type Spec: Send + 'static;
    type State: Send + 'static;
    type Change: Send + 'static;
    type Query: Send + 'static;
    type Resolution: Send + 'static;
    type PlannerMeta: Send + Sync + 'static;

    fn init_state(&self, spec: &Self::Spec) -> BraidResult<Self::State>;
    fn reset_state(&self, state: &mut Self::State, spec: &Self::Spec) -> BraidResult<()>;
    fn apply(&self, state: &mut Self::State, changes: &[Self::Change]) -> BraidResult<()>;
    fn compile(
        &self,
        state: &Self::State,
        scratch: &mut PlannerScratch,
    ) -> BraidResult<CompiledPlan<Self::PlannerMeta>>;
    fn encode_batch(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        queries: &[Self::Query],
        packet: &mut JobPacket,
        scratch: &mut BatchScratch,
    ) -> BraidResult<()>;
    fn decode_batch(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        packet: &JobPacket,
    ) -> BraidResult<Vec<Self::Resolution>>;
}
