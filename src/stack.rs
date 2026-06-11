use crate::buffer_pool::ReusablePool;
use crate::compute::ComputeBackend;
use crate::error::{BraidError, BraidResult};
use crate::executor::{BackendHandle, BraidExecutor};
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
    backend: BackendHandle<C>,
    state: Mutex<P::State>,
    current_version: RwLock<Arc<FrozenStackVersion<P, C>>>,
    planner_scratch: Mutex<PlannerScratch>,
    batch_scratch_pool: ReusablePool<BatchScratch>,
    packet_pool: ReusablePool<JobPacket>,
    inline_context_pool: ReusablePool<InlineContext>,
    compute_scratch_pool: ReusablePool<ComputeScratch>,
    prepared_pool: Arc<Mutex<Vec<C::Prepared>>>,
    jobs: Mutex<HashMap<JobId, Arc<JobRecord<P::Resolution>>>>,
    next_job_id: AtomicU64,
    next_version_id: AtomicU64,
    current_version_id: AtomicU64,
}

/// Reusable caller-owned scratch for low-latency inline execution.
#[derive(Debug, Default)]
pub struct InlineContext {
    packet: JobPacket,
    batch_scratch: BatchScratch,
    cancel: CancelFlag,
}

impl InlineContext {
    fn reset(&mut self) {
        self.packet.clear_for_reuse();
        self.batch_scratch.reset();
        self.cancel.reset();
    }
}

/// Typed runtime handle for one compiled planner/backend combination.
pub struct Stack<P, C>
where
    P: PlannerBackend,
    C: ComputeBackend,
{
    inner: Arc<StackInner<P, C>>,
}

struct JobExecution<P, C>
where
    P: PlannerBackend,
    C: ComputeBackend,
{
    inner: Arc<StackInner<P, C>>,
    version: Arc<FrozenStackVersion<P, C>>,
    queries: Vec<P::Query>,
    record: Arc<JobRecord<P::Resolution>>,
    packet: Mutex<Option<JobPacket>>,
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
    /// Create one stack from one planner, one shared backend handle, and one initial spec.
    ///
    /// This builds mutable planner state, compiles an initial frozen version, and prepares the
    /// backend before the stack becomes dispatchable.
    pub fn create(
        executor: Arc<BraidExecutor>,
        planner: Arc<P>,
        backend: BackendHandle<C>,
        spec: P::Spec,
    ) -> BraidResult<Self> {
        let state = planner.init_state(&spec)?;
        let prepared_pool = Arc::new(Mutex::new(Vec::new()));
        let mut planner_scratch = PlannerScratch::default();
        let compiled = planner.compile(&state, &mut planner_scratch)?;
        compiled.validate()?;

        let mut compute_scratch = ComputeScratch::default();
        let prepared = backend.prepare_blocking(&compiled, None, &mut compute_scratch)?;
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
                inline_context_pool: ReusablePool::default(),
                compute_scratch_pool: ReusablePool::default(),
                prepared_pool,
                jobs: Mutex::new(HashMap::new()),
                next_job_id: AtomicU64::new(1),
                next_version_id: AtomicU64::new(2),
                current_version_id: AtomicU64::new(1),
            }),
        })
    }

    /// Apply changes to the live mutable planner state without recompiling.
    pub fn apply(&self, changes: &[P::Change]) -> BraidResult<()> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| BraidError::poisoned("stack.state"))?;
        self.inner.planner.apply(&mut state, &changes)
    }

    /// Reset the live mutable planner state from a fresh spec without recompiling.
    pub fn replace(&self, spec: P::Spec) -> BraidResult<()> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| BraidError::poisoned("stack.state"))?;
        self.inner.planner.reset_state(&mut state, &spec)
    }

    /// Compile the current mutable planner state into a new frozen version and swap it in.
    pub fn recompile(&self) -> BraidResult<VersionId> {
        let state = self
            .inner
            .state
            .lock()
            .map_err(|_| BraidError::poisoned("stack.state"))?;
        self.inner.compile_from_state(&state)
    }

    /// Build a new planner state from changes, compile it, and swap it in only if compile works.
    ///
    /// This is the safest high-level update path when planner changes should behave
    /// transactionally.
    pub fn update(&self, changes: &[P::Change]) -> BraidResult<VersionId> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| BraidError::poisoned("stack.state"))?;
        let next_state = self.inner.planner.updated_state(&state, &changes)?;
        let version_id = self.inner.compile_from_state(&next_state)?;
        *state = next_state;
        Ok(version_id)
    }

    /// Dispatch one batch of queries against the current frozen version.
    ///
    /// The returned [`JobId`] is stack-local.
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

        let execution = Arc::new(JobExecution {
            inner: Arc::clone(&self.inner),
            version,
            queries,
            record,
            packet: Mutex::new(None),
        });

        if let Err(error) = execution.schedule_encode() {
            if let Ok(mut jobs) = self.inner.jobs.lock() {
                jobs.remove(&job_id);
            }
            return Err(error);
        }

        Ok(job_id)
    }

    /// Run one batch inline on the caller thread and return decoded planner results directly.
    ///
    /// This skips async job records, executor queue hops, and backend lane scheduling. It is the
    /// right path for tiny serial workloads where queueing cost would dominate real compute.
    pub fn resolve_inline(&self, queries: &[P::Query]) -> BraidResult<Vec<P::Resolution>> {
        let mut context = self
            .inner
            .inline_context_pool
            .checkout("stack.inline_context")?;
        let result = self.resolve_inline_with(queries, &mut context);
        context.reset();
        self.inner
            .inline_context_pool
            .give_back("stack.inline_context", context)?;
        result
    }

    /// Run one batch inline on the caller thread with caller-owned reusable scratch.
    pub fn resolve_inline_with(
        &self,
        queries: &[P::Query],
        context: &mut InlineContext,
    ) -> BraidResult<Vec<P::Resolution>> {
        let version = {
            let version = self
                .inner
                .current_version
                .read()
                .map_err(|_| BraidError::poisoned("stack.current_version"))?;
            Arc::clone(&version)
        };
        context.cancel.reset();
        if P::PREFER_DIRECT_ONE_QUERY_INLINE
            && let [query] = queries
        {
            #[cfg(debug_assertions)]
            {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    self.inner
                        .execute_one_direct_inline(&version, query, context)
                }));
                return match result {
                    Ok(result) => result.map(|value| vec![value]),
                    Err(_) => Err(BraidError::from("inline execution panicked")),
                };
            }

            #[cfg(not(debug_assertions))]
            {
                return self
                    .inner
                    .execute_one_direct_inline(&version, query, context)
                    .map(|value| vec![value]);
            }
        }

        if P::PREFER_ONE_QUERY_INLINE
            && let [query] = queries
        {
            #[cfg(debug_assertions)]
            {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    self.inner.execute_one_inline(&version, query, context)
                }));
                return match result {
                    Ok(result) => result.map(|value| vec![value]),
                    Err(_) => Err(BraidError::from("inline execution panicked")),
                };
            }

            #[cfg(not(debug_assertions))]
            {
                return self
                    .inner
                    .execute_one_inline(&version, query, context)
                    .map(|value| vec![value]);
            }
        }

        #[cfg(debug_assertions)]
        {
            let result = catch_unwind(AssertUnwindSafe(|| {
                self.inner.execute_inline(&version, queries, context)
            }));
            match result {
                Ok(result) => result,
                Err(_) => Err(BraidError::from("inline execution panicked")),
            }
        }

        #[cfg(not(debug_assertions))]
        {
            self.inner.execute_inline(&version, queries, context)
        }
    }

    /// Run one query inline on the caller thread and return one decoded resolution.
    pub fn resolve_one_inline(&self, query: P::Query) -> BraidResult<P::Resolution> {
        let mut context = self
            .inner
            .inline_context_pool
            .checkout("stack.inline_context")?;
        let result = self.resolve_one_inline_ref_with(&query, &mut context);
        context.reset();
        self.inner
            .inline_context_pool
            .give_back("stack.inline_context", context)?;
        result
    }

    /// Run one query inline on the caller thread with caller-owned reusable scratch.
    pub fn resolve_one_inline_with(
        &self,
        query: P::Query,
        context: &mut InlineContext,
    ) -> BraidResult<P::Resolution> {
        self.resolve_one_inline_ref_with(&query, context)
    }

    /// Run one borrowed query inline on the caller thread and return one decoded resolution.
    pub fn resolve_one_inline_ref(&self, query: &P::Query) -> BraidResult<P::Resolution> {
        let mut context = self
            .inner
            .inline_context_pool
            .checkout("stack.inline_context")?;
        let result = self.resolve_one_inline_ref_with(query, &mut context);
        context.reset();
        self.inner
            .inline_context_pool
            .give_back("stack.inline_context", context)?;
        result
    }

    /// Run one borrowed query inline on the caller thread with caller-owned reusable scratch.
    pub fn resolve_one_inline_ref_with(
        &self,
        query: &P::Query,
        context: &mut InlineContext,
    ) -> BraidResult<P::Resolution> {
        let version = {
            let version = self
                .inner
                .current_version
                .read()
                .map_err(|_| BraidError::poisoned("stack.current_version"))?;
            Arc::clone(&version)
        };
        context.cancel.reset();
        if P::PREFER_DIRECT_ONE_QUERY_INLINE {
            #[cfg(debug_assertions)]
            {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    self.inner
                        .execute_one_direct_inline(&version, query, context)
                }));
                return match result {
                    Ok(result) => result,
                    Err(_) => Err(BraidError::from("inline execution panicked")),
                };
            }

            #[cfg(not(debug_assertions))]
            {
                return self
                    .inner
                    .execute_one_direct_inline(&version, query, context);
            }
        }

        #[cfg(debug_assertions)]
        {
            let result = catch_unwind(AssertUnwindSafe(|| {
                self.inner.execute_one_inline(&version, query, context)
            }));
            match result {
                Ok(result) => result,
                Err(_) => Err(BraidError::from("inline execution panicked")),
            }
        }

        #[cfg(not(debug_assertions))]
        {
            self.inner.execute_one_inline(&version, query, context)
        }
    }

    /// Return the current frozen version id.
    pub fn current_version_id(&self) -> BraidResult<VersionId> {
        Ok(self.inner.current_version_id.load(Ordering::Acquire))
    }

    /// Poll one stack-local job for coarse status.
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

    /// Wait for a dispatched job and return decoded planner results.
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

    /// Run one batch inline and return decoded planner results immediately.
    ///
    /// This avoids job allocation, queueing, and collect synchronization for benchmarks
    /// and caller-owned workloads that do not need async behavior.
    pub fn dispatch_collect(&self, queries: &[P::Query]) -> BraidResult<Vec<P::Resolution>> {
        let version = {
            let version = self
                .inner
                .current_version
                .read()
                .map_err(|_| BraidError::poisoned("stack.current_version"))?;
            Arc::clone(&version)
        };

        let mut context = self
            .inner
            .inline_context_pool
            .checkout("stack.inline_context")?;
        context.cancel.reset();
        let result = if P::PREFER_DIRECT_ONE_QUERY_INLINE
            && let [query] = queries
        {
            #[cfg(debug_assertions)]
            {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    self.inner
                        .execute_one_direct_inline(&version, query, &mut context)
                        .map(|value| vec![value])
                }));
                match result {
                    Ok(result) => result,
                    Err(_) => Err(BraidError::from("inline execution panicked")),
                }
            }

            #[cfg(not(debug_assertions))]
            {
                self.inner
                    .execute_one_direct_inline(&version, query, &mut context)
                    .map(|value| vec![value])
            }
        } else if P::PREFER_ONE_QUERY_INLINE
            && let [query] = queries
        {
            #[cfg(debug_assertions)]
            {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    self.inner.execute_one_inline(&version, query, &mut context)
                }));
                match result {
                    Ok(result) => result.map(|value| vec![value]),
                    Err(_) => Err(BraidError::from("inline execution panicked")),
                }
            }

            #[cfg(not(debug_assertions))]
            {
                self.inner
                    .execute_one_inline(&version, query, &mut context)
                    .map(|value| vec![value])
            }
        } else {
            self.inner
                .execute_inline(&version, queries, &mut context)
        };
        context.reset();
        self.inner
            .inline_context_pool
            .give_back("stack.inline_context", context)?;
        result
    }

    /// Request cooperative cancellation for one stack-local job.
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
    fn checkout_packet(&self, version: &FrozenStackVersion<P, C>) -> BraidResult<JobPacket> {
        let mut packet = self.packet_pool.checkout("stack.packet_pool")?;
        packet.clear_for_reuse();
        for buffer in &version.compiled.static_buffers {
            packet.load_static_buffer(buffer.slot, &buffer.data);
        }
        Ok(packet)
    }

    fn recycle_packet(&self, mut packet: JobPacket) -> BraidResult<()> {
        packet.clear_for_reuse();
        self.packet_pool.give_back("stack.packet_pool", packet)
    }

    fn encode_packet(
        &self,
        version: &FrozenStackVersion<P, C>,
        queries: &[P::Query],
    ) -> BraidResult<JobPacket> {
        let mut packet = self.checkout_packet(version)?;
        let mut batch_scratch = self.batch_scratch_pool.checkout("stack.batch_scratch")?;
        batch_scratch.reset();

        let encode_result =
            self.planner
                .encode_batch(&version.compiled, queries, &mut packet, &mut batch_scratch);

        batch_scratch.reset();
        self.batch_scratch_pool
            .give_back("stack.batch_scratch", batch_scratch)?;

        if let Err(error) = encode_result {
            let _ = self.recycle_packet(packet);
            return Err(error);
        }

        if let Err(error) = validate_runtime_packet(&version.compiled, &packet) {
            let _ = self.recycle_packet(packet);
            return Err(error);
        }

        Ok(packet)
    }

    fn run_stage_once(
        &self,
        version: &FrozenStackVersion<P, C>,
        stage_index: usize,
        packet: &mut JobPacket,
        cancel: &CancelFlag,
    ) -> BraidResult<()> {
        let prepared = version
            .prepared
            .as_ref()
            .ok_or_else(|| BraidError::from("missing prepared state"))?;
        let stage = version
            .compiled
            .pipeline
            .stages
            .get(stage_index)
            .ok_or_else(|| BraidError::from("missing stage"))?;
        self.backend
            .backend
            .run_stage(prepared, stage_index, stage, packet, cancel)?;
        validate_runtime_packet(&version.compiled, packet)
    }

    fn run_all_stages(
        &self,
        version: &FrozenStackVersion<P, C>,
        packet: &mut JobPacket,
        cancel: &CancelFlag,
    ) -> BraidResult<()> {
        for stage_index in 0..version.compiled.pipeline.stages.len() {
            if cancel.is_cancelled() {
                return Err(BraidError::Cancelled);
            }
            self.run_stage_once(version, stage_index, packet, cancel)?;
        }
        Ok(())
    }

    fn run_one_all_stages(
        &self,
        version: &FrozenStackVersion<P, C>,
        packet: &mut JobPacket,
        cancel: &CancelFlag,
    ) -> BraidResult<()> {
        let prepared = version
            .prepared
            .as_ref()
            .ok_or_else(|| BraidError::from("missing prepared state"))?;
        for (stage_index, stage) in version.compiled.pipeline.stages.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err(BraidError::Cancelled);
            }
            self.backend
                .backend
                .run_one_stage(prepared, stage_index, stage, packet, cancel)?;
            validate_runtime_packet(&version.compiled, packet)?;
        }
        Ok(())
    }

    fn decode_packet(
        &self,
        version: &FrozenStackVersion<P, C>,
        packet: JobPacket,
    ) -> BraidResult<Vec<P::Resolution>> {
        let decode_result = validate_runtime_packet(&version.compiled, &packet)
            .and_then(|_| self.planner.decode_batch(&version.compiled, &packet));
        let recycle_result = self.recycle_packet(packet);
        match decode_result {
            Ok(values) => recycle_result.map(|_| values),
            Err(error) => {
                let _ = recycle_result;
                Err(error)
            }
        }
    }

    fn execute_inline(
        &self,
        version: &FrozenStackVersion<P, C>,
        queries: &[P::Query],
        context: &mut InlineContext,
    ) -> BraidResult<Vec<P::Resolution>> {
        if context.cancel.is_cancelled() {
            return Err(BraidError::Cancelled);
        }
        context.packet.clear_for_reuse();
        for buffer in &version.compiled.static_buffers {
            context.packet.load_static_buffer(buffer.slot, &buffer.data);
        }
        context.batch_scratch.reset();
        self.planner.encode_batch(
            &version.compiled,
            queries,
            &mut context.packet,
            &mut context.batch_scratch,
        )?;
        context.batch_scratch.reset();
        validate_runtime_packet(&version.compiled, &context.packet)?;
        if let Err(error) = self.run_all_stages(version, &mut context.packet, &context.cancel) {
            context.packet.clear_for_reuse();
            return Err(error);
        }
        validate_runtime_packet(&version.compiled, &context.packet)?;
        self.planner
            .decode_batch(&version.compiled, &context.packet)
    }

    fn execute_one_inline(
        &self,
        version: &FrozenStackVersion<P, C>,
        query: &P::Query,
        context: &mut InlineContext,
    ) -> BraidResult<P::Resolution> {
        if context.cancel.is_cancelled() {
            return Err(BraidError::Cancelled);
        }
        context.packet.clear_for_reuse();
        for buffer in &version.compiled.static_buffers {
            context.packet.load_static_buffer(buffer.slot, &buffer.data);
        }
        context.batch_scratch.reset();
        self.planner.encode_one(
            &version.compiled,
            query,
            &mut context.packet,
            &mut context.batch_scratch,
        )?;
        context.batch_scratch.reset();
        validate_runtime_packet(&version.compiled, &context.packet)?;
        if let Err(error) = self.run_one_all_stages(version, &mut context.packet, &context.cancel) {
            context.packet.clear_for_reuse();
            return Err(error);
        }
        validate_runtime_packet(&version.compiled, &context.packet)?;
        self.planner.decode_one(&version.compiled, &context.packet)
    }

    fn execute_one_direct_inline(
        &self,
        version: &FrozenStackVersion<P, C>,
        query: &P::Query,
        context: &mut InlineContext,
    ) -> BraidResult<P::Resolution> {
        if context.cancel.is_cancelled() {
            return Err(BraidError::Cancelled);
        }
        self.planner
            .resolve_one_direct(&version.compiled, query)
            .ok_or_else(|| BraidError::from("missing direct one-query planner path"))?
    }

    fn compile_from_state(&self, state: &P::State) -> BraidResult<VersionId> {
        let mut planner_scratch = self
            .planner_scratch
            .lock()
            .map_err(|_| BraidError::poisoned("stack.planner_scratch"))?;
        planner_scratch.reset();
        let compiled = self.planner.compile(state, &mut planner_scratch)?;
        compiled.validate()?;
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
            .prepare_blocking(&compiled, reuse, &mut compute_scratch)?;
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
        self.current_version_id.store(version_id, Ordering::Release);
        Ok(version_id)
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

impl<P, C> JobExecution<P, C>
where
    P: PlannerBackend,
    C: ComputeBackend,
{
    fn schedule_encode(self: &Arc<Self>) -> BraidResult<()> {
        let job = Arc::clone(self);
        self.inner.executor.submit(move || job.run_encode_task())
    }

    fn schedule_stage(self: &Arc<Self>, stage_index: usize) -> BraidResult<()> {
        let run_job = Arc::clone(self);
        let reject_job = Arc::clone(self);
        self.inner.backend.schedule_with_rejection(
            move || run_job.run_stage_task(stage_index),
            move |error| reject_job.reject_scheduled_task(error),
        )
    }

    fn schedule_decode(self: &Arc<Self>) -> BraidResult<()> {
        let job = Arc::clone(self);
        self.inner.executor.submit(move || job.run_decode_task())
    }

    fn run_encode_task(self: Arc<Self>) {
        self.run_caught(|job| job.encode_step());
    }

    fn run_stage_task(self: Arc<Self>, stage_index: usize) {
        self.run_caught(|job| job.stage_step(stage_index));
    }

    fn run_decode_task(self: Arc<Self>) {
        self.run_caught(|job| job.decode_step());
    }

    fn run_caught<F>(self: Arc<Self>, body: F)
    where
        F: FnOnce(&Arc<Self>) -> BraidResult<()>,
    {
        let result = catch_unwind(AssertUnwindSafe(|| body(&self)));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(BraidError::Cancelled)) => {
                self.cleanup_packet();
                self.inner.finish_cancelled(&self.record);
            }
            Ok(Err(error)) => {
                self.cleanup_packet();
                let _ = self.inner.finish_failed(&self.record, error);
            }
            Err(_) => {
                self.cleanup_packet();
                let _ = self
                    .inner
                    .finish_failed(&self.record, BraidError::from("executor task panicked"));
            }
        }
    }

    fn encode_step(self: &Arc<Self>) -> BraidResult<()> {
        if self.record.cancel.is_cancelled() {
            return Err(BraidError::Cancelled);
        }

        self.inner.mark_running(&self.record)?;
        let packet = self.inner.encode_packet(&self.version, &self.queries)?;
        self.store_packet(packet)?;
        if self.version.compiled.pipeline.stages.is_empty() {
            self.schedule_decode()
        } else {
            self.schedule_stage(0)
        }
    }

    fn stage_step(self: &Arc<Self>, stage_index: usize) -> BraidResult<()> {
        if self.record.cancel.is_cancelled() {
            return Err(BraidError::Cancelled);
        }

        {
            let mut packet = self
                .packet
                .lock()
                .map_err(|_| BraidError::poisoned("job_execution.packet"))?;
            let packet = packet
                .as_mut()
                .ok_or_else(|| BraidError::from("missing job packet"))?;
            self.inner
                .run_stage_once(&self.version, stage_index, packet, &self.record.cancel)?;
        }

        let next_stage = stage_index + 1;
        if next_stage < self.version.compiled.pipeline.stages.len() {
            self.schedule_stage(next_stage)
        } else {
            self.schedule_decode()
        }
    }

    fn decode_step(self: &Arc<Self>) -> BraidResult<()> {
        if self.record.cancel.is_cancelled() {
            return Err(BraidError::Cancelled);
        }

        let packet = self.take_packet()?;
        let decoded = self.inner.decode_packet(&self.version, packet)?;
        self.inner.finish_completed(&self.record, decoded)
    }

    fn store_packet(&self, packet: JobPacket) -> BraidResult<()> {
        let mut slot = self
            .packet
            .lock()
            .map_err(|_| BraidError::poisoned("job_execution.packet"))?;
        if slot.is_some() {
            return Err(BraidError::from("job packet already stored"));
        }
        *slot = Some(packet);
        Ok(())
    }

    fn take_packet(&self) -> BraidResult<JobPacket> {
        let mut slot = self
            .packet
            .lock()
            .map_err(|_| BraidError::poisoned("job_execution.packet"))?;
        slot.take()
            .ok_or_else(|| BraidError::from("missing job packet"))
    }

    fn cleanup_packet(&self) {
        let packet = {
            let mut slot = match self.packet.lock() {
                Ok(slot) => slot,
                Err(_) => return,
            };
            slot.take()
        };

        let Some(mut packet) = packet else {
            return;
        };
        packet.clear_for_reuse();
        let _ = self
            .inner
            .packet_pool
            .give_back("stack.packet_pool", packet);
    }

    fn reject_scheduled_task(self: Arc<Self>, error: BraidError) {
        self.cleanup_packet();
        let _ = self.inner.finish_failed(&self.record, error);
    }
}

#[inline]
fn validate_runtime_packet<M>(
    compiled: &crate::CompiledPlan<M>,
    packet: &JobPacket,
) -> BraidResult<()> {
    #[cfg(debug_assertions)]
    {
        compiled.validate_packet(packet)
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (compiled, packet);
        Ok(())
    }
}
