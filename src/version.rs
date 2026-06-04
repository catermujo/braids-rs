use crate::compute::ComputeBackend;
use crate::pipeline::{CompiledPlan, VersionId};
use crate::planner::PlannerBackend;
use std::sync::{Mutex, Weak};

pub(crate) struct FrozenStackVersion<P, C>
where
    P: PlannerBackend,
    C: ComputeBackend,
{
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
