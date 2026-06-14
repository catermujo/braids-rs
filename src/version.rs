use crate::compute::ComputeBackend;
use crate::pipeline::{CompiledPlan, VersionId};
use crate::planner::PlannerBackend;
use std::sync::{Mutex, Weak};

pub(crate) struct FrozenStackVersion<P, C>
where
    P: PlannerBackend,
    C: ComputeBackend,
{
    #[allow(dead_code, reason = "Not sure why this is here")]
    pub(crate) id: VersionId,
    pub(crate) compiled: CompiledPlan<P::PlannerMeta>,
    pub(crate) prepared: Option<C::Prepared>,
    pub(crate) prepared_pool: Weak<Mutex<Vec<C::Prepared>>>,
}

impl<P, C> Drop for FrozenStackVersion<P, C>
where
    P: PlannerBackend,
    C: ComputeBackend,
{
    fn drop(&mut self) {
        let Some(prepared) = self.prepared.take() else {
            return;
        };
        let Some(pool) = self.prepared_pool.upgrade() else {
            return;
        };
        if let Ok(mut pool) = pool.lock() {
            pool.push(prepared);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FrozenStackVersion;
    use crate::compute::ComputeBackend;
    use crate::error::BraidResult;
    use crate::pipeline::{CompiledPlan, PipelineShape};
    use crate::planner::PlannerBackend;
    use std::sync::{Arc, Mutex, Weak};

    #[derive(Clone)]
    struct NoopPlanner;

    #[derive(Clone)]
    struct NoopBackend;

    impl PlannerBackend for NoopPlanner {
        type Spec = ();
        type State = ();
        type Change = ();
        type Query = ();
        type Resolution = ();
        type PlannerMeta = ();

        fn init_state(&self, _spec: &Self::Spec) -> BraidResult<Self::State> {
            Ok(())
        }

        fn reset_state(&self, _state: &mut Self::State, _spec: &Self::Spec) -> BraidResult<()> {
            Ok(())
        }

        fn apply(&self, _state: &mut Self::State, _changes: &[Self::Change]) -> BraidResult<()> {
            Ok(())
        }

        fn updated_state(
            &self,
            _state: &Self::State,
            _changes: &[Self::Change],
        ) -> BraidResult<Self::State> {
            Ok(())
        }

        fn compile(
            &self,
            _state: &Self::State,
            _scratch: &mut crate::scratch::PlannerScratch,
        ) -> BraidResult<CompiledPlan<Self::PlannerMeta>> {
            Ok(CompiledPlan {
                pipeline: PipelineShape::default(),
                static_buffers: Vec::new(),
                planner_meta: (),
            })
        }

        fn encode_batch(
            &self,
            _plan: &CompiledPlan<Self::PlannerMeta>,
            _queries: &[Self::Query],
            _packet: &mut crate::job::JobPacket,
            _scratch: &mut crate::scratch::BatchScratch,
        ) -> BraidResult<()> {
            Ok(())
        }

        fn decode_batch(
            &self,
            _plan: &CompiledPlan<Self::PlannerMeta>,
            _packet: &crate::job::JobPacket,
        ) -> BraidResult<Vec<Self::Resolution>> {
            Ok(Vec::new())
        }
    }

    impl ComputeBackend for NoopBackend {
        type Prepared = u8;

        fn prepare<M: Send + Sync + 'static>(
            &self,
            _plan: &CompiledPlan<M>,
            _reuse: Option<Self::Prepared>,
            _scratch: &mut crate::scratch::ComputeScratch,
        ) -> BraidResult<Self::Prepared> {
            Ok(7)
        }

        fn run_stage(
            &self,
            _prepared: &Self::Prepared,
            _stage_index: usize,
            _stage: &crate::pipeline::StageSpec,
            _packet: &mut crate::job::JobPacket,
            _cancel: &crate::job::CancelFlag,
        ) -> BraidResult<()> {
            Ok(())
        }
    }

    fn compiled_plan() -> CompiledPlan<()> {
        CompiledPlan {
            pipeline: PipelineShape::default(),
            static_buffers: Vec::new(),
            planner_meta: (),
        }
    }

    #[test]
    fn prepared_is_returned_to_reusable_pool_on_drop() {
        let pool = Arc::new(Mutex::new(Vec::<u8>::new()));
        {
            let _version = FrozenStackVersion::<NoopPlanner, NoopBackend> {
                id: 1,
                compiled: compiled_plan(),
                prepared: Some(9),
                prepared_pool: Arc::downgrade(&pool),
            };
        }
        let pool = pool.lock().unwrap();
        assert_eq!(pool.as_slice(), &[9]);
    }

    #[test]
    fn drop_without_prepared_or_dead_pool_is_noop() {
        let pool = {
            let pool = Arc::new(Mutex::new(Vec::<u8>::new()));
            let weak = Arc::downgrade(&pool);
            (pool, weak)
        };

        {
            let _version_with_dead_pool = FrozenStackVersion::<NoopPlanner, NoopBackend> {
                id: 1,
                compiled: compiled_plan(),
                prepared: Some(7),
                prepared_pool: pool.1.clone(),
            };
            drop(pool.0);
        }

        {
            let _version_without_prepared = FrozenStackVersion::<NoopPlanner, NoopBackend> {
                id: 2,
                compiled: compiled_plan(),
                prepared: None,
                prepared_pool: Weak::new(),
            };
        }
    }
}
