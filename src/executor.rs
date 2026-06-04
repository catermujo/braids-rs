use crate::compute::ComputeBackend;
use crate::error::{BraidError, BraidResult};
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread::{self, JoinHandle};

pub(crate) trait RunnableTask: Send {
    fn run(self: Box<Self>);
}

pub(crate) struct TaskFn<F: FnOnce() + Send + 'static> {
    pub(crate) func: Option<F>,
}

impl<F: FnOnce() + Send + 'static> RunnableTask for TaskFn<F> {
    fn run(mut self: Box<Self>) {
        if let Some(func) = self.func.take() {
            func();
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BackendConfig {
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
}

#[derive(Default)]
struct ExecutorInner {
    queue: Mutex<VecDeque<Box<dyn RunnableTask>>>,
    wake: Condvar,
    shutdown: AtomicBool,
}

pub struct BraidExecutor {
    inner: Arc<ExecutorInner>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl BraidExecutor {
    pub fn new(worker_count: usize) -> Self {
        let inner = Arc::new(ExecutorInner::default());
        let mut workers = Vec::with_capacity(worker_count.max(1));
        for _ in 0..worker_count.max(1) {
            let worker_inner = Arc::clone(&inner);
            workers.push(thread::spawn(move || worker_loop(worker_inner)));
        }
        Self {
            inner,
            workers: Mutex::new(workers),
        }
    }

    pub fn register_backend<C>(&self, backend: Arc<C>, config: BackendConfig) -> BackendHandle<C>
    where
        C: ComputeBackend,
    {
        self.register_backend_with_prepare_lanes(backend, config.lane_count, config.lane_count)
    }

    pub fn register_backend_with_prepare_lanes<C>(
        &self,
        backend: Arc<C>,
        lane_count: usize,
        prepare_lane_count: usize,
    ) -> BackendHandle<C>
    where
        C: ComputeBackend,
    {
        BackendHandle {
            backend,
            runtime: Arc::new(BackendRuntime {
                executor: Arc::downgrade(&self.inner),
                state: Mutex::new(BackendQueueState {
                    lane_count: lane_count.max(1),
                    prepare_lane_count: prepare_lane_count.max(1).min(lane_count.max(1)),
                    lanes_in_use: 0,
                    prepares_in_use: 0,
                    waiting: VecDeque::new(),
                }),
                prepare_wake: Condvar::new(),
            }),
        }
    }

    pub fn shutdown(&self) {
        if self.inner.shutdown.swap(true, Ordering::AcqRel) {
            return;
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
            Box::new(TaskFn { func: Some(task) }) as Box<dyn RunnableTask>,
        )
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
    pub(crate) fn schedule<F>(&self, task: F) -> BraidResult<()>
    where
        F: FnOnce() + Send + 'static,
    {
        self.runtime.schedule(Box::new(TaskFn { func: Some(task) }))
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
    fn schedule(self: &Arc<Self>, task: Box<dyn RunnableTask>) -> BraidResult<()> {
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
            self.submit_ready(task)?;
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
            if self.submit_ready(task).is_err() {
                if let Ok(mut state) = self.state.lock() {
                    state.lanes_in_use = state.lanes_in_use.saturating_sub(1);
                }
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

        if let Some(task) = next {
            if self.submit_ready(task).is_err() {
                if let Ok(mut state) = self.state.lock() {
                    state.lanes_in_use = state.lanes_in_use.saturating_sub(1);
                }
            }
        }
        self.prepare_wake.notify_all();
    }

    fn submit_ready(self: &Arc<Self>, task: Box<dyn RunnableTask>) -> BraidResult<()> {
        let Some(executor) = self.executor.upgrade() else {
            if let Ok(mut state) = self.state.lock() {
                state.lanes_in_use = state.lanes_in_use.saturating_sub(1);
            }
            return Err(BraidError::ExecutorShutdown);
        };

        let wrapped = Box::new(BackendWrappedTask {
            runtime: Arc::clone(self),
            inner: Some(task),
        }) as Box<dyn RunnableTask>;
        if let Err(error) = submit_boxed(&executor, wrapped) {
            if let Ok(mut state) = self.state.lock() {
                state.lanes_in_use = state.lanes_in_use.saturating_sub(1);
            }
            return Err(error);
        }

        Ok(())
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

fn submit_boxed(inner: &Arc<ExecutorInner>, task: Box<dyn RunnableTask>) -> BraidResult<()> {
    if inner.shutdown.load(Ordering::Acquire) {
        return Err(BraidError::ExecutorShutdown);
    }

    let mut queue = inner
        .queue
        .lock()
        .map_err(|_| BraidError::poisoned("executor.queue"))?;
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
