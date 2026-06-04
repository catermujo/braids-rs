use crate::buffer_pool::ReusablePool;
use crate::compute::ComputeBackend;
use crate::error::{BraidError, BraidResult};
use crate::executor::{BackendHandle, BraidExecutor};
use crate::job::{CancelFlag, JobPacket, JobStatus};
use crate::pipeline::{BufferLayout, BufferSpec, CompiledPlan, JobId, VersionId};
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
        validate_compiled_plan(&compiled)?;

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
        let snapshot = state.clone();
        self.inner.planner.apply(&mut state, &changes)?;
        let version_id = match self.inner.compile_from_state(&state) {
            Ok(version_id) => version_id,
            Err(error) => {
                *state = snapshot;
                return Err(error);
            }
        };
        Ok(version_id)
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
        validate_compiled_plan(&compiled)?;
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

        let mut packet = self.inner.packet_pool.checkout("stack.packet_pool")?;
        packet.clear_for_reuse();
        for buffer in &self.version.compiled.static_buffers {
            packet.load_static_buffer(buffer.slot, &buffer.data);
        }
        let mut batch_scratch = self
            .inner
            .batch_scratch_pool
            .checkout("stack.batch_scratch")?;
        batch_scratch.reset();

        let encode_result = self.inner.planner.encode_batch(
            &self.version.compiled,
            &self.queries,
            &mut packet,
            &mut batch_scratch,
        );

        batch_scratch.reset();
        self.inner
            .batch_scratch_pool
            .give_back("stack.batch_scratch", batch_scratch)?;

        if let Err(error) = encode_result {
            packet.clear_for_reuse();
            self.inner
                .packet_pool
                .give_back("stack.packet_pool", packet)?;
            return Err(error);
        }
        validate_packet_against_plan(&self.version.compiled, &packet)?;

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
            let prepared = self
                .version
                .prepared
                .as_ref()
                .ok_or_else(|| BraidError::from("missing prepared state"))?;
            let stage = self
                .version
                .compiled
                .pipeline
                .stages
                .get(stage_index)
                .ok_or_else(|| BraidError::from("missing stage"))?;
            self.inner.backend.backend.run_stage(
                prepared,
                stage_index,
                stage,
                packet,
                &self.record.cancel,
            )?;
            validate_packet_against_plan(&self.version.compiled, packet)?;
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

        let mut packet = self.take_packet()?;
        validate_packet_against_plan(&self.version.compiled, &packet)?;
        let decoded = self
            .inner
            .planner
            .decode_batch(&self.version.compiled, &packet)?;
        packet.clear_for_reuse();
        self.inner
            .packet_pool
            .give_back("stack.packet_pool", packet)?;
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

fn validate_compiled_plan<M>(plan: &CompiledPlan<M>) -> BraidResult<()> {
    let mut specs = HashMap::with_capacity(plan.pipeline.buffers.len());
    for spec in &plan.pipeline.buffers {
        if specs.insert(spec.slot, spec).is_some() {
            return Err(BraidError::InvalidSpec(format!(
                "duplicate buffer slot {} in pipeline",
                spec.slot
            )));
        }
    }

    let mut static_slots = HashMap::with_capacity(plan.static_buffers.len());
    for buffer in &plan.static_buffers {
        if static_slots.insert(buffer.slot, ()).is_some() {
            return Err(BraidError::InvalidSpec(format!(
                "duplicate static buffer slot {}",
                buffer.slot
            )));
        }
        let Some(spec) = specs.get(&buffer.slot) else {
            return Err(BraidError::InvalidSpec(format!(
                "static buffer slot {} is not declared in pipeline",
                buffer.slot
            )));
        };
        if spec.element_kind != buffer.data.kind() {
            return Err(BraidError::InvalidSpec(format!(
                "static buffer slot {} has wrong element kind",
                buffer.slot
            )));
        }
    }

    for (stage_index, stage) in plan.pipeline.stages.iter().enumerate() {
        for (kernel_index, kernel) in stage.kernels.iter().enumerate() {
            for binding in &kernel.bindings {
                if !specs.contains_key(&binding.slot) {
                    return Err(BraidError::InvalidSpec(format!(
                        "stage {} kernel {} references undeclared buffer slot {}",
                        stage_index, kernel_index, binding.slot
                    )));
                }
            }
        }
    }

    Ok(())
}

fn validate_packet_against_plan<M>(plan: &CompiledPlan<M>, packet: &JobPacket) -> BraidResult<()> {
    let mut specs = HashMap::with_capacity(plan.pipeline.buffers.len());
    for spec in &plan.pipeline.buffers {
        specs.insert(spec.slot, spec);
    }

    for (slot, kind, len) in packet.buffer_descriptors() {
        let Some(spec) = specs.get(&slot) else {
            if len == 0 {
                continue;
            }
            return Err(BraidError::InvalidSpec(format!(
                "packet contains undeclared buffer slot {}",
                slot
            )));
        };
        if spec.element_kind != kind {
            return Err(BraidError::InvalidBufferType {
                slot,
                expected: spec.element_kind,
            });
        }
        validate_buffer_layout(spec, packet.query_count(), len)?;
    }

    Ok(())
}

fn validate_buffer_layout(spec: &BufferSpec, query_count: usize, len: usize) -> BraidResult<()> {
    let expected_len = match spec.layout {
        BufferLayout::PerQueryScalar => Some(query_count),
        BufferLayout::PerQueryVector { width } => query_count.checked_mul(width),
        BufferLayout::Dynamic => return Ok(()),
    };

    let Some(expected_len) = expected_len else {
        return Err(BraidError::InvalidSpec(format!(
            "buffer slot {} length overflow for declared layout",
            spec.slot
        )));
    };

    if len != expected_len {
        return Err(BraidError::InvalidSpec(format!(
            "buffer slot {} has length {} but declared layout expects {}",
            spec.slot, len, expected_len
        )));
    }

    Ok(())
}
