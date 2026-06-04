use crate::error::{BraidError, BraidResult};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

trait RunnableTask: Send {
    fn run(self: Box<Self>);
}

struct TaskFn<F: FnOnce() + Send + 'static> {
    func: Option<F>,
}

impl<F: FnOnce() + Send + 'static> RunnableTask for TaskFn<F> {
    fn run(mut self: Box<Self>) {
        if let Some(func) = self.func.take() {
            func();
        }
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

    pub fn shutdown(&self) {
        if self.inner.shutdown.swap(true, Ordering::AcqRel) {
            return;
        }
        self.inner.wake.notify_all();

        if let Ok(mut workers) = self.workers.lock() {
            for handle in workers.drain(..) {
                let _ = handle.join();
            }
        }
    }

    pub(crate) fn submit<F>(&self, task: F) -> BraidResult<()>
    where
        F: FnOnce() + Send + 'static,
    {
        if self.inner.shutdown.load(Ordering::Acquire) {
            return Err(BraidError::ExecutorShutdown);
        }

        let mut queue = self
            .inner
            .queue
            .lock()
            .map_err(|_| BraidError::poisoned("executor.queue"))?;
        queue.push_back(Box::new(TaskFn { func: Some(task) }));
        drop(queue);
        self.inner.wake.notify_one();
        Ok(())
    }
}

impl Drop for BraidExecutor {
    fn drop(&mut self) {
        self.shutdown();
    }
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
