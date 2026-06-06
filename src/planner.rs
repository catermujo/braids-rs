use crate::error::{BraidError, BraidResult};
use crate::job::JobPacket;
use crate::pipeline::CompiledPlan;
use crate::scratch::{BatchScratch, PlannerScratch};

/// Planner-side interface for turning domain data into an executable pipeline.
///
/// A planner owns all domain meaning:
///
/// - how authored specs become mutable planner state
/// - how state changes are applied
/// - how state compiles into a generic [`CompiledPlan`]
/// - how queries encode into a [`JobPacket`]
/// - how backend output decodes into user-facing resolutions
pub trait PlannerBackend: Send + Sync + 'static {
    /// Set true only when planner can answer one inline query directly without packets/backend.
    const PREFER_DIRECT_ONE_QUERY_INLINE: bool = false;

    /// Set true only when single-query hooks are meaningfully cheaper than batch hooks.
    const PREFER_ONE_QUERY_INLINE: bool = false;

    /// Initial authored input used to build planner state.
    type Spec: Send + 'static;
    /// Mutable planner-owned state kept by a [`crate::Stack`].
    type State: Send + 'static;
    /// Incremental change description applied to planner state.
    type Change: Send + 'static;
    /// Per-dispatch query item.
    type Query: Send + Sync + 'static;
    /// Per-query decoded output returned by [`crate::Stack::collect`].
    type Resolution: Send + 'static;
    /// Planner-specific metadata stored inside [`CompiledPlan`].
    type PlannerMeta: Send + Sync + 'static;

    /// Build initial mutable state from a spec.
    fn init_state(&self, spec: &Self::Spec) -> BraidResult<Self::State>;
    /// Reset an existing state object from a fresh spec, preferably reusing storage.
    fn reset_state(&self, state: &mut Self::State, spec: &Self::Spec) -> BraidResult<()>;
    /// Apply changes in place to the current mutable state.
    fn apply(&self, state: &mut Self::State, changes: &[Self::Change]) -> BraidResult<()>;
    /// Build the next state from the current state plus changes without mutating the old one.
    ///
    /// This supports transactional update flow: compile the new state first, then swap it in only
    /// if compile succeeds.
    fn updated_state(
        &self,
        state: &Self::State,
        changes: &[Self::Change],
    ) -> BraidResult<Self::State>;
    /// Compile planner state into a generic pipeline plus planner metadata.
    fn compile(
        &self,
        state: &Self::State,
        scratch: &mut PlannerScratch,
    ) -> BraidResult<CompiledPlan<Self::PlannerMeta>>;
    /// Encode a batch of planner queries into a reusable packet buffer.
    fn encode_batch(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        queries: &[Self::Query],
        packet: &mut JobPacket,
        scratch: &mut BatchScratch,
    ) -> BraidResult<()>;
    /// Encode one planner query into a reusable packet buffer.
    ///
    /// Override this when the single-query path can avoid batch scaffolding or extra temporary
    /// allocations. The default delegates to [`Self::encode_batch`].
    fn encode_one(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        query: &Self::Query,
        packet: &mut JobPacket,
        scratch: &mut BatchScratch,
    ) -> BraidResult<()> {
        self.encode_batch(plan, std::slice::from_ref(query), packet, scratch)
    }

    /// Decode backend output from a packet into user-facing query resolutions.
    fn decode_batch(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        packet: &JobPacket,
    ) -> BraidResult<Vec<Self::Resolution>>;
    /// Decode one backend output from a packet into one user-facing query resolution.
    ///
    /// Override this when the single-query path can avoid batch result allocation or reshaping.
    /// The default delegates to [`Self::decode_batch`].
    fn decode_one(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        packet: &JobPacket,
    ) -> BraidResult<Self::Resolution> {
        let mut values = self.decode_batch(plan, packet)?;
        match values.len() {
            1 => Ok(values.pop().expect("single decode result")),
            _ => Err(BraidError::from(
                "single-query decode returned unexpected batch size",
            )),
        }
    }

    /// Answer one inline query directly without going through packet encode/stage/decode.
    ///
    /// Override this only for latency-critical single-query paths that can bypass generic planner
    /// and backend execution entirely. The default returns `None`.
    fn resolve_one_direct(
        &self,
        _plan: &CompiledPlan<Self::PlannerMeta>,
        _query: &Self::Query,
    ) -> Option<BraidResult<Self::Resolution>> {
        None
    }
}
