use crate::compute::ComputeBackend;
use crate::error::{BraidError, BraidResult};
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread::{self, JoinHandle};

pub(crate) trait RunnableTask: Send {
    fn run(self: Box<Self>);
    fn reject(self: Box<Self>, error: BraidError);
}

pub(crate) struct TaskFn<F, R = fn(BraidError)>
where
    F: FnOnce() + Send + 'static,
    R: FnOnce(BraidError) + Send + 'static,
{
    pub(crate) func: Option<F>,
    pub(crate) reject: Option<R>,
}

impl<F> TaskFn<F, fn(BraidError)>
where
    F: FnOnce() + Send + 'static,
{
    fn new(func: F) -> Self {
        Self {
            func: Some(func),
            reject: None,
        }
    }
}

impl<F, R> TaskFn<F, R>
where
    F: FnOnce() + Send + 'static,
    R: FnOnce(BraidError) + Send + 'static,
{
    fn with_rejection(func: F, reject: R) -> Self {
        Self {
            func: Some(func),
            reject: Some(reject),
        }
    }
}

impl<F, R> RunnableTask for TaskFn<F, R>
where
    F: FnOnce() + Send + 'static,
    R: FnOnce(BraidError) + Send + 'static,
{
    fn run(mut self: Box<Self>) {
        if let Some(func) = self.func.take() {
            func();
        }
    }

    fn reject(mut self: Box<Self>, error: BraidError) {
        if let Some(reject) = self.reject.take() {
            reject(error);
        }
    }
}

#[derive(Clone, Copy, Debug)]
/// Capacity limits for one shared backend instance.
///
/// `lane_count` is how many stage executions the executor may run concurrently against that
/// backend. Set it to `1` when the backend is effectively serialized.
pub struct BackendConfig {
    /// Maximum number of stage executions allowed at once for this backend.
    pub lane_count: usize,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self { lane_count: 1 }
    }
}

struct BackendQueueState {
    lane_count: usize,
    prepare_lane_count: usize,
    lanes_in_use: usize,
    prepares_in_use: usize,
    waiting: VecDeque<Box<dyn RunnableTask>>,
}

struct BackendRuntime {
    executor: Weak<ExecutorInner>,
    state: Mutex<BackendQueueState>,
    prepare_wake: Condvar,
}

/// Handle to one shared registered backend.
///
/// Cloning the handle does not duplicate the backend. All clones still point at the same backend
/// instance and the same executor-managed lane budget.
pub struct BackendHandle<C>
where
    C: ComputeBackend,
{
    pub(crate) backend: Arc<C>,
    runtime: Arc<BackendRuntime>,
}

impl<C> Clone for BackendHandle<C>
where
    C: ComputeBackend,
{
    fn clone(&self) -> Self {
        Self {
            backend: Arc::clone(&self.backend),
            runtime: Arc::clone(&self.runtime),
        }
    }
}

struct BackendWrappedTask {
    runtime: Arc<BackendRuntime>,
    inner: Option<Box<dyn RunnableTask>>,
}

impl RunnableTask for BackendWrappedTask {
    fn run(mut self: Box<Self>) {
        if let Some(task) = self.inner.take() {
            let _ = catch_unwind(AssertUnwindSafe(|| task.run()));
        }
        self.runtime.finish_lane();
    }

    fn reject(mut self: Box<Self>, error: BraidError) {
        if let Some(task) = self.inner.take() {
            let _ = catch_unwind(AssertUnwindSafe(|| task.reject(error)));
        }
        self.runtime.finish_lane();
    }
}

#[derive(Default)]
struct ExecutorInner {
    queue: Mutex<VecDeque<Box<dyn RunnableTask>>>,
    wake: Condvar,
    shutdown: AtomicBool,
}

/// Shared async executor used by many stacks.
///
/// The executor owns worker threads and task queues. Backend-specific capacity still comes from
/// backend lane counts, not from worker count alone.
pub struct BraidExecutor {
    backends: Mutex<Vec<Weak<BackendRuntime>>>,
    inner: Arc<ExecutorInner>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl BraidExecutor {
    /// Create an executor with `worker_count` worker threads.
    pub fn new(worker_count: usize) -> Self {
        let inner = Arc::new(ExecutorInner::default());
        let mut workers = Vec::with_capacity(worker_count.max(1));
        for _ in 0..worker_count.max(1) {
            let worker_inner = Arc::clone(&inner);
            workers.push(thread::spawn(move || worker_loop(worker_inner)));
        }
        Self {
            backends: Mutex::new(Vec::new()),
            inner,
            workers: Mutex::new(workers),
        }
    }

    /// Register one shared backend with one stage-lane count.
    pub fn register_backend<C>(&self, backend: Arc<C>, config: BackendConfig) -> BackendHandle<C>
    where
        C: ComputeBackend,
    {
        self.register_backend_with_prepare_lanes(backend, config.lane_count, config.lane_count)
    }

    /// Register one shared backend with separate stage-lane and prepare-lane counts.
    ///
    /// This is useful when background prepare/recompile pressure should be capped more tightly
    /// than normal runtime stage execution.
    pub fn register_backend_with_prepare_lanes<C>(
        &self,
        backend: Arc<C>,
        lane_count: usize,
        prepare_lane_count: usize,
    ) -> BackendHandle<C>
    where
        C: ComputeBackend,
    {
        let runtime = Arc::new(BackendRuntime {
            executor: Arc::downgrade(&self.inner),
            state: Mutex::new(BackendQueueState {
                lane_count: lane_count.max(1),
                prepare_lane_count: prepare_lane_count.max(1).min(lane_count.max(1)),
                lanes_in_use: 0,
                prepares_in_use: 0,
                waiting: VecDeque::new(),
            }),
            prepare_wake: Condvar::new(),
        });
        if let Ok(mut backends) = self.backends.lock() {
            backends.push(Arc::downgrade(&runtime));
        }
        BackendHandle { backend, runtime }
    }

    /// Stop worker threads and reject any queued backend work.
    pub fn shutdown(&self) {
        if self.inner.shutdown.swap(true, Ordering::AcqRel) {
            return;
        }

        let runtimes = match self.backends.lock() {
            Ok(mut backends) => {
                backends.retain(|runtime| runtime.strong_count() > 0);
                backends
                    .iter()
                    .filter_map(Weak::upgrade)
                    .collect::<Vec<_>>()
            }
            Err(_) => Vec::new(),
        };
        for runtime in runtimes {
            runtime.shutdown();
        }

        self.inner.wake.notify_all();
        let current_id = thread::current().id();

        if let Ok(mut workers) = self.workers.lock() {
            for handle in workers.drain(..) {
                if handle.thread().id() == current_id {
                    continue;
                }
                let _ = handle.join();
            }
        }
    }

    pub(crate) fn submit<F>(&self, task: F) -> BraidResult<()>
    where
        F: FnOnce() + Send + 'static,
    {
        submit_boxed(
            &self.inner,
            Box::new(TaskFn::new(task)) as Box<dyn RunnableTask>,
        )
        .map_err(|(error, _)| error)
    }
}

impl Drop for BraidExecutor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl<C> BackendHandle<C>
where
    C: ComputeBackend,
{
    pub(crate) fn schedule_with_rejection<F, R>(&self, task: F, reject: R) -> BraidResult<()>
    where
        F: FnOnce() + Send + 'static,
        R: FnOnce(BraidError) + Send + 'static,
    {
        self.runtime
            .schedule(Box::new(TaskFn::with_rejection(task, reject)), true)
    }

    pub(crate) fn prepare_blocking<M: Send + Sync + 'static>(
        &self,
        plan: &crate::pipeline::CompiledPlan<M>,
        reuse: Option<C::Prepared>,
        scratch: &mut crate::scratch::ComputeScratch,
    ) -> BraidResult<C::Prepared> {
        let _permit = self.runtime.acquire_prepare()?;
        self.backend.prepare(plan, reuse, scratch)
    }
}

impl BackendRuntime {
    fn schedule(
        self: &Arc<Self>,
        task: Box<dyn RunnableTask>,
        reject_on_failure: bool,
    ) -> BraidResult<()> {
        let ready = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| BraidError::poisoned("backend_runtime.state"))?;
            if state.lanes_in_use < state.lane_count {
                state.lanes_in_use += 1;
                Some(task)
            } else {
                state.waiting.push_back(task);
                None
            }
        };

        if let Some(task) = ready {
            self.submit_ready(task, reject_on_failure)?;
        }
        Ok(())
    }

    fn acquire_prepare(self: &Arc<Self>) -> BraidResult<PreparePermit> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| BraidError::poisoned("backend_runtime.state"))?;
        loop {
            let Some(executor) = self.executor.upgrade() else {
                return Err(BraidError::ExecutorShutdown);
            };
            if executor.shutdown.load(Ordering::Acquire) {
                return Err(BraidError::ExecutorShutdown);
            }

            if state.waiting.is_empty()
                && state.lanes_in_use < state.lane_count
                && state.prepares_in_use < state.prepare_lane_count
            {
                state.lanes_in_use += 1;
                state.prepares_in_use += 1;
                return Ok(PreparePermit {
                    runtime: Arc::clone(self),
                    active: true,
                });
            }

            state = self
                .prepare_wake
                .wait(state)
                .map_err(|_| BraidError::poisoned("backend_runtime.state"))?;
        }
    }

    fn finish_lane(self: &Arc<Self>) {
        let next = {
            let mut state = match self.state.lock() {
                Ok(state) => state,
                Err(_) => return,
            };
            if let Some(task) = state.waiting.pop_front() {
                Some(task)
            } else {
                state.lanes_in_use = state.lanes_in_use.saturating_sub(1);
                None
            }
        };

        if let Some(task) = next {
            if self.submit_ready(task, true).is_err() {
                self.prepare_wake.notify_all();
            }
        } else {
            self.prepare_wake.notify_all();
        }
    }

    fn finish_prepare(self: &Arc<Self>) {
        let next = {
            let mut state = match self.state.lock() {
                Ok(state) => state,
                Err(_) => return,
            };
            state.prepares_in_use = state.prepares_in_use.saturating_sub(1);
            state.lanes_in_use = state.lanes_in_use.saturating_sub(1);
            if let Some(task) = state.waiting.pop_front() {
                state.lanes_in_use += 1;
                Some(task)
            } else {
                None
            }
        };

        if let Some(task) = next
            && self.submit_ready(task, true).is_err()
        {
            self.prepare_wake.notify_all();
        }
        self.prepare_wake.notify_all();
    }

    fn submit_ready(
        self: &Arc<Self>,
        task: Box<dyn RunnableTask>,
        reject_on_failure: bool,
    ) -> BraidResult<()> {
        let wrapped = Box::new(BackendWrappedTask {
            runtime: Arc::clone(self),
            inner: Some(task),
        }) as Box<dyn RunnableTask>;

        let Some(executor) = self.executor.upgrade() else {
            if reject_on_failure {
                wrapped.reject(BraidError::ExecutorShutdown);
            } else if let Ok(mut state) = self.state.lock() {
                state.lanes_in_use = state.lanes_in_use.saturating_sub(1);
            }
            return Err(BraidError::ExecutorShutdown);
        };

        if let Err((error, task)) = submit_boxed(&executor, wrapped) {
            if reject_on_failure {
                task.reject(error.clone());
            } else if let Ok(mut state) = self.state.lock() {
                state.lanes_in_use = state.lanes_in_use.saturating_sub(1);
            }
            return Err(error);
        }

        Ok(())
    }

    fn shutdown(&self) {
        let waiting = {
            let mut state = match self.state.lock() {
                Ok(state) => state,
                Err(_) => return,
            };
            state.waiting.drain(..).collect::<Vec<_>>()
        };

        for task in waiting {
            task.reject(BraidError::ExecutorShutdown);
        }
        self.prepare_wake.notify_all();
    }
}

struct PreparePermit {
    runtime: Arc<BackendRuntime>,
    active: bool,
}

impl Drop for PreparePermit {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.runtime.finish_prepare();
        self.active = false;
    }
}

fn submit_boxed(
    inner: &Arc<ExecutorInner>,
    task: Box<dyn RunnableTask>,
) -> Result<(), (BraidError, Box<dyn RunnableTask>)> {
    if inner.shutdown.load(Ordering::Acquire) {
        return Err((BraidError::ExecutorShutdown, task));
    }

    let mut queue = match inner.queue.lock() {
        Ok(queue) => queue,
        Err(_) => return Err((BraidError::poisoned("executor.queue"), task)),
    };
    queue.push_back(task);
    drop(queue);
    inner.wake.notify_one();
    Ok(())
}

fn worker_loop(inner: Arc<ExecutorInner>) {
    loop {
        let task = {
            let mut queue = match inner.queue.lock() {
                Ok(queue) => queue,
                Err(_) => return,
            };

            loop {
                if let Some(task) = queue.pop_front() {
                    break task;
                }
                if inner.shutdown.load(Ordering::Acquire) {
                    return;
                }
                queue = match inner.wake.wait(queue) {
                    Ok(queue) => queue,
                    Err(_) => return,
                };
            }
        };

        task.run();
    }
}
