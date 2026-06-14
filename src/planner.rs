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

#[cfg(test)]
mod tests {
    use super::PlannerBackend;
    use crate::error::BraidError;
    use crate::pipeline::{CompiledPlan, PipelineShape};
    use crate::scratch::{BatchScratch, PlannerScratch};
    use std::sync::Arc;

    #[derive(Default)]
    struct TestPlanner {
        emit_many: bool,
        counts: Arc<planner_counts::Counts>,
    }

    mod planner_counts {
        use std::sync::atomic::AtomicUsize;

        #[derive(Default)]
        pub struct Counts {
            pub encode_batch: AtomicUsize,
            pub encode_one: AtomicUsize,
            pub decode_batch: AtomicUsize,
            pub decode_one: AtomicUsize,
        }
    }

    impl PlannerBackend for TestPlanner {
        type Spec = ();
        type State = ();
        type Change = ();
        type Query = u32;
        type Resolution = u32;
        type PlannerMeta = ();

        fn init_state(&self, _spec: &Self::Spec) -> crate::error::BraidResult<Self::State> {
            Ok(())
        }

        fn reset_state(
            &self,
            _state: &mut Self::State,
            _spec: &Self::Spec,
        ) -> crate::error::BraidResult<()> {
            Ok(())
        }

        fn apply(
            &self,
            _state: &mut Self::State,
            _changes: &[Self::Change],
        ) -> crate::error::BraidResult<()> {
            Ok(())
        }

        fn updated_state(
            &self,
            _state: &Self::State,
            _changes: &[Self::Change],
        ) -> crate::error::BraidResult<Self::State> {
            Ok(())
        }

        fn compile(
            &self,
            _state: &Self::State,
            _scratch: &mut PlannerScratch,
        ) -> crate::error::BraidResult<CompiledPlan<Self::PlannerMeta>> {
            Ok(CompiledPlan {
                pipeline: PipelineShape {
                    buffers: Vec::new(),
                    stages: Vec::new(),
                },
                static_buffers: Vec::new(),
                planner_meta: (),
            })
        }

        fn encode_batch(
            &self,
            _plan: &CompiledPlan<Self::PlannerMeta>,
            _queries: &[Self::Query],
            _packet: &mut crate::job::JobPacket,
            _scratch: &mut BatchScratch,
        ) -> crate::error::BraidResult<()> {
            self.counts
                .encode_batch
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }

        fn encode_one(
            &self,
            plan: &CompiledPlan<Self::PlannerMeta>,
            query: &Self::Query,
            packet: &mut crate::job::JobPacket,
            scratch: &mut BatchScratch,
        ) -> crate::error::BraidResult<()> {
            self.counts
                .encode_one
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            PlannerBackend::encode_batch(self, plan, std::slice::from_ref(query), packet, scratch)
        }

        fn decode_batch(
            &self,
            _plan: &CompiledPlan<Self::PlannerMeta>,
            _packet: &crate::job::JobPacket,
        ) -> crate::error::BraidResult<Vec<Self::Resolution>> {
            self.counts
                .decode_batch
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if self.emit_many {
                Ok(vec![1, 2])
            } else {
                Ok(vec![99])
            }
        }

        fn decode_one(
            &self,
            _plan: &CompiledPlan<Self::PlannerMeta>,
            packet: &crate::job::JobPacket,
        ) -> crate::error::BraidResult<Self::Resolution> {
            self.counts
                .decode_one
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut values = Self::decode_batch(self, _plan, packet)?;
            match values.len() {
                1 => Ok(values.pop().expect("single decode result")),
                _ => Err(BraidError::from(
                    "single-query decode returned unexpected batch size",
                )),
            }
        }
    }

    #[test]
    fn default_decode_and_encode_paths_call_batch_variants() {
        let counts = planner_counts::Counts::default();
        let planner = TestPlanner {
            emit_many: false,
            counts: Arc::new(counts),
        };
        let plan = planner
            .compile(&(), &mut PlannerScratch::default())
            .expect("compile");
        let mut packet = crate::job::JobPacket::default();
        let mut scratch = BatchScratch::default();

        planner
            .encode_one(&plan, &7, &mut packet, &mut scratch)
            .expect("encode one");
        planner.decode_one(&plan, &packet).expect("decode one");

        assert_eq!(
            planner
                .counts
                .encode_batch
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            planner
                .counts
                .encode_one
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            planner
                .counts
                .decode_batch
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            planner
                .counts
                .decode_one
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn decode_one_errors_when_batch_is_not_singleton() {
        let planner = TestPlanner {
            emit_many: true,
            counts: Arc::new(planner_counts::Counts::default()),
        };
        let plan = planner
            .compile(&(), &mut PlannerScratch::default())
            .expect("compile");

        let err = planner
            .decode_one(&plan, &crate::job::JobPacket::default())
            .expect_err("expected mismatch");
        assert!(err.to_string().contains("unexpected batch size"));
    }

    #[test]
    fn resolve_one_direct_defaults_to_none() {
        let planner = TestPlanner {
            emit_many: false,
            counts: Arc::new(planner_counts::Counts::default()),
        };
        let plan = planner
            .compile(&(), &mut PlannerScratch::default())
            .expect("compile");
        assert!(planner.resolve_one_direct(&plan, &12).is_none());
    }
}
