use crate::buffer_pool::ReusablePool;
use crate::compute::ComputeBackend;
use crate::error::{BraidError, BraidResult};
use crate::executor::BraidExecutor;
use crate::job::{CancelFlag, JobPacket, JobStatus};
use crate::pipeline::{JobId, VersionId};
use crate::planner::PlannerBackend;
use crate::scratch::{BatchScratch, ComputeScratch, PlannerScratch};
use crate::version::FrozenStackVersion;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

enum JobRecordState<R> {
    Queued,
    Running,
    Completed(Option<Vec<R>>),
    Failed(Option<BraidError>),
    Cancelled,
}

struct JobRecord<R> {
    state: Mutex<JobRecordState<R>>,
    wake: Condvar,
    cancel: CancelFlag,
}

impl<R> JobRecord<R> {
    fn new() -> Self {
        Self {
            state: Mutex::new(JobRecordState::Queued),
            wake: Condvar::new(),
            cancel: CancelFlag::default(),
        }
    }

    fn status(&self) -> BraidResult<JobStatus> {
        let state = self
            .state
            .lock()
            .map_err(|_| BraidError::poisoned("stack.job_state"))?;
        Ok(match &*state {
            JobRecordState::Queued => JobStatus::Queued,
            JobRecordState::Running => JobStatus::Running,
            JobRecordState::Completed(_) => JobStatus::Completed,
            JobRecordState::Failed(_) => JobStatus::Failed,
            JobRecordState::Cancelled => JobStatus::Cancelled,
        })
    }
}

struct StackInner<P, C>
where
    P: PlannerBackend,
    C: ComputeBackend,
{
    executor: Arc<BraidExecutor>,
    planner: Arc<P>,
    backend: Arc<C>,
    state: Mutex<P::State>,
    current_version: RwLock<Arc<FrozenStackVersion<P, C>>>,
    planner_scratch: Mutex<PlannerScratch>,
    batch_scratch_pool: ReusablePool<BatchScratch>,
    packet_pool: ReusablePool<JobPacket>,
    compute_scratch_pool: ReusablePool<ComputeScratch>,
    prepared_pool: Arc<Mutex<Vec<C::Prepared>>>,
    jobs: Mutex<HashMap<JobId, Arc<JobRecord<P::Resolution>>>>,
    next_job_id: AtomicU64,
    next_version_id: AtomicU64,
}

pub struct Stack<P, C>
where
    P: PlannerBackend,
    C: ComputeBackend,
{
    inner: Arc<StackInner<P, C>>,
}

impl<P, C> Clone for Stack<P, C>
where
    P: PlannerBackend,
    C: ComputeBackend,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<P, C> Stack<P, C>
where
    P: PlannerBackend,
    C: ComputeBackend,
{
    pub fn create(
        executor: Arc<BraidExecutor>,
        planner: Arc<P>,
        backend: Arc<C>,
        spec: P::Spec,
    ) -> BraidResult<Self> {
        let state = planner.init_state(&spec)?;
        let prepared_pool = Arc::new(Mutex::new(Vec::new()));
        let mut planner_scratch = PlannerScratch::default();
        let compiled = planner.compile(&state, &mut planner_scratch)?;

        let mut compute_scratch = ComputeScratch::default();
        let prepared = backend.prepare(&compiled, None, &mut compute_scratch)?;
        let version = Arc::new(FrozenStackVersion::<P, C> {
            id: 1,
            compiled,
            prepared: Some(prepared),
            prepared_pool: Arc::downgrade(&prepared_pool),
        });

        Ok(Self {
            inner: Arc::new(StackInner {
                executor,
                planner,
                backend,
                state: Mutex::new(state),
                current_version: RwLock::new(version),
                planner_scratch: Mutex::new(planner_scratch),
                batch_scratch_pool: ReusablePool::default(),
                packet_pool: ReusablePool::default(),
                compute_scratch_pool: ReusablePool::default(),
                prepared_pool,
                jobs: Mutex::new(HashMap::new()),
                next_job_id: AtomicU64::new(1),
                next_version_id: AtomicU64::new(2),
            }),
        })
    }

    pub fn apply(&self, changes: Vec<P::Change>) -> BraidResult<()> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| BraidError::poisoned("stack.state"))?;
        self.inner.planner.apply(&mut state, &changes)
    }

    pub fn replace(&self, spec: P::Spec) -> BraidResult<()> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| BraidError::poisoned("stack.state"))?;
        self.inner.planner.reset_state(&mut state, &spec)
    }

    pub fn recompile(&self) -> BraidResult<VersionId> {
        let state = self
            .inner
            .state
            .lock()
            .map_err(|_| BraidError::poisoned("stack.state"))?;
        self.inner.compile_from_state(&state)
    }

    pub fn update(&self, changes: Vec<P::Change>) -> BraidResult<VersionId> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| BraidError::poisoned("stack.state"))?;
        self.inner.planner.apply(&mut state, &changes)?;
        self.inner.compile_from_state(&state)
    }

    pub fn dispatch(&self, queries: Vec<P::Query>) -> BraidResult<JobId> {
        let version = {
            let version = self
                .inner
                .current_version
                .read()
                .map_err(|_| BraidError::poisoned("stack.current_version"))?;
            Arc::clone(&version)
        };

        let job_id = self.inner.next_job_id.fetch_add(1, Ordering::AcqRel);
        let record = Arc::new(JobRecord::new());
        {
            let mut jobs = self
                .inner
                .jobs
                .lock()
                .map_err(|_| BraidError::poisoned("stack.jobs"))?;
            jobs.insert(job_id, Arc::clone(&record));
        }

        let inner = Arc::clone(&self.inner);
        let panic_inner = Arc::clone(&inner);
        let panic_record = Arc::clone(&record);
        self.inner.executor.submit(move || {
            if catch_unwind(AssertUnwindSafe(|| inner.run_job(version, queries, record))).is_err() {
                let _ = panic_inner
                    .finish_failed(&panic_record, BraidError::from("executor task panicked"));
            }
        })?;
        Ok(job_id)
    }

    pub fn current_version_id(&self) -> BraidResult<VersionId> {
        let version = self
            .inner
            .current_version
            .read()
            .map_err(|_| BraidError::poisoned("stack.current_version"))?;
        Ok(version.id)
    }

    pub fn poll(&self, job: JobId) -> JobStatus {
        let jobs = match self.inner.jobs.lock() {
            Ok(jobs) => jobs,
            Err(_) => return JobStatus::Failed,
        };
        let Some(record) = jobs.get(&job) else {
            return JobStatus::Failed;
        };
        record.status().unwrap_or(JobStatus::Failed)
    }

    pub fn collect(&self, job: JobId) -> BraidResult<Vec<P::Resolution>> {
        let record = {
            let jobs = self
                .inner
                .jobs
                .lock()
                .map_err(|_| BraidError::poisoned("stack.jobs"))?;
            jobs.get(&job).cloned().ok_or(BraidError::UnknownJob)?
        };

        let result = {
            let mut state = record
                .state
                .lock()
                .map_err(|_| BraidError::poisoned("stack.job_state"))?;
            loop {
                match &mut *state {
                    JobRecordState::Queued | JobRecordState::Running => {
                        state = record
                            .wake
                            .wait(state)
                            .map_err(|_| BraidError::poisoned("stack.job_state"))?;
                    }
                    JobRecordState::Completed(values) => {
                        break Ok(values.take().unwrap_or_default());
                    }
                    JobRecordState::Failed(error) => {
                        break Err(error
                            .take()
                            .unwrap_or_else(|| BraidError::from("job failed")));
                    }
                    JobRecordState::Cancelled => break Err(BraidError::Cancelled),
                }
            }
        };

        if let Ok(mut jobs) = self.inner.jobs.lock() {
            jobs.remove(&job);
        }
        result
    }

    pub fn cancel(&self, job: JobId) -> bool {
        let jobs = match self.inner.jobs.lock() {
            Ok(jobs) => jobs,
            Err(_) => return false,
        };
        let Some(record) = jobs.get(&job) else {
            return false;
        };
        record.cancel.cancel();
        true
    }
}

impl<P, C> StackInner<P, C>
where
    P: PlannerBackend,
    C: ComputeBackend,
{
    fn compile_from_state(&self, state: &P::State) -> BraidResult<VersionId> {
        let mut planner_scratch = self
            .planner_scratch
            .lock()
            .map_err(|_| BraidError::poisoned("stack.planner_scratch"))?;
        planner_scratch.reset();
        let compiled = self.planner.compile(state, &mut planner_scratch)?;
        drop(planner_scratch);

        let mut compute_scratch = self
            .compute_scratch_pool
            .checkout("stack.compute_scratch")?;
        compute_scratch.reset();
        let reuse = {
            let mut prepared_pool = self
                .prepared_pool
                .lock()
                .map_err(|_| BraidError::poisoned("stack.prepared_pool"))?;
            prepared_pool.pop()
        };
        let prepared = self
            .backend
            .prepare(&compiled, reuse, &mut compute_scratch)?;
        self.compute_scratch_pool
            .give_back("stack.compute_scratch", compute_scratch)?;

        let version_id = self.next_version_id.fetch_add(1, Ordering::AcqRel);
        let frozen = Arc::new(FrozenStackVersion::<P, C> {
            id: version_id,
            compiled,
            prepared: Some(prepared),
            prepared_pool: Arc::downgrade(&self.prepared_pool),
        });
        let mut current = self
            .current_version
            .write()
            .map_err(|_| BraidError::poisoned("stack.current_version"))?;
        *current = frozen;
        Ok(version_id)
    }

    fn run_job(
        self: &Arc<Self>,
        version: Arc<FrozenStackVersion<P, C>>,
        queries: Vec<P::Query>,
        record: Arc<JobRecord<P::Resolution>>,
    ) {
        if record.cancel.is_cancelled() {
            self.finish_cancelled(&record);
            return;
        }
        if self.mark_running(&record).is_err() {
            return;
        }

        let result = self.run_job_inner(&version, &queries, &record.cancel);
        match result {
            Ok(values) => {
                let _ = self.finish_completed(&record, values);
            }
            Err(BraidError::Cancelled) => {
                self.finish_cancelled(&record);
            }
            Err(error) => {
                let _ = self.finish_failed(&record, error);
            }
        }
    }

    fn run_job_inner(
        &self,
        version: &FrozenStackVersion<P, C>,
        queries: &[P::Query],
        cancel: &CancelFlag,
    ) -> BraidResult<Vec<P::Resolution>> {
        let mut packet = self.packet_pool.checkout("stack.packet_pool")?;
        packet.clear_for_reuse();
        let mut batch_scratch = self.batch_scratch_pool.checkout("stack.batch_scratch")?;
        batch_scratch.reset();

        self.planner
            .encode_batch(&version.compiled, queries, &mut packet, &mut batch_scratch)?;

        for (stage_index, stage) in version.compiled.pipeline.stages.iter().enumerate() {
            if cancel.is_cancelled() {
                packet.clear_for_reuse();
                batch_scratch.reset();
                self.packet_pool.give_back("stack.packet_pool", packet)?;
                self.batch_scratch_pool
                    .give_back("stack.batch_scratch", batch_scratch)?;
                return Err(BraidError::Cancelled);
            }

            let prepared = version
                .prepared
                .as_ref()
                .ok_or_else(|| BraidError::from("missing prepared state"))?;
            self.backend
                .run_stage(prepared, stage_index, stage, &mut packet, cancel)?;
        }

        let decoded = self.planner.decode_batch(&version.compiled, &packet)?;
        packet.clear_for_reuse();
        batch_scratch.reset();
        self.packet_pool.give_back("stack.packet_pool", packet)?;
        self.batch_scratch_pool
            .give_back("stack.batch_scratch", batch_scratch)?;
        Ok(decoded)
    }

    fn mark_running(&self, record: &Arc<JobRecord<P::Resolution>>) -> BraidResult<()> {
        let mut state = record
            .state
            .lock()
            .map_err(|_| BraidError::poisoned("stack.job_state"))?;
        *state = JobRecordState::Running;
        record.wake.notify_all();
        Ok(())
    }

    fn finish_completed(
        &self,
        record: &Arc<JobRecord<P::Resolution>>,
        values: Vec<P::Resolution>,
    ) -> BraidResult<()> {
        let mut state = record
            .state
            .lock()
            .map_err(|_| BraidError::poisoned("stack.job_state"))?;
        *state = JobRecordState::Completed(Some(values));
        record.wake.notify_all();
        Ok(())
    }

    fn finish_failed(
        &self,
        record: &Arc<JobRecord<P::Resolution>>,
        error: BraidError,
    ) -> BraidResult<()> {
        let mut state = record
            .state
            .lock()
            .map_err(|_| BraidError::poisoned("stack.job_state"))?;
        *state = JobRecordState::Failed(Some(error));
        record.wake.notify_all();
        Ok(())
    }

    fn finish_cancelled(&self, record: &Arc<JobRecord<P::Resolution>>) {
        if let Ok(mut state) = record.state.lock() {
            *state = JobRecordState::Cancelled;
            record.wake.notify_all();
        }
    }
}
