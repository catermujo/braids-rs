//! Starter CPU backend implementation.
//!
//! This backend is intentionally simple. It gives planners a fast way to execute real workloads
//! without committing to a custom device/runtime backend yet.

use crate::compute::ComputeBackend;
use crate::error::{BraidError, BraidResult};
use crate::job::{CancelFlag, JobPacket};
use crate::pipeline::{CompiledPlan, KernelKind, KernelSpec, StageSpec};
use crate::scratch::ComputeScratch;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

/// Prepared CPU kernel instance that can run against a mutable [`JobPacket`].
pub trait CpuKernel: Send + Sync {
    /// Execute the kernel.
    fn run(&self, packet: &mut JobPacket, cancel: &CancelFlag) -> BraidResult<()>;
}

/// Factory that turns compiled [`KernelSpec`] payloads into runnable [`CpuKernel`] instances.
pub trait CpuKernelFactory: Send + Sync {
    /// Prepare one kernel instance.
    fn prepare(
        &self,
        kernel: &KernelSpec,
        scratch: &mut ComputeScratch,
    ) -> BraidResult<Box<dyn CpuKernel>>;
}

#[derive(Default)]
/// Generic registry-based CPU backend for planner-defined kernel kinds.
pub struct CpuComputeBackend {
    factories: HashMap<KernelKind, Arc<dyn CpuKernelFactory>>,
}

impl CpuComputeBackend {
    /// Create an empty CPU backend with no registered kernel factories.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a factory for one kernel kind.
    pub fn register_factory(
        &mut self,
        kind_id: KernelKind,
        factory: Arc<dyn CpuKernelFactory>,
    ) -> &mut Self {
        self.factories.insert(kind_id, factory);
        self
    }
}

/// Backend-prepared CPU stage set.
pub struct CpuPrepared {
    stages: Vec<PreparedStage>,
}

struct PreparedStage {
    kernels: Vec<Box<dyn CpuKernel>>,
}

impl Debug for CpuPrepared {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuPrepared")
            .field("stage_count", &self.stages.len())
            .finish()
    }
}

impl ComputeBackend for CpuComputeBackend {
    type Prepared = CpuPrepared;

    fn prepare<M: Send + Sync + 'static>(
        &self,
        plan: &CompiledPlan<M>,
        _reuse: Option<Self::Prepared>,
        scratch: &mut ComputeScratch,
    ) -> BraidResult<Self::Prepared> {
        scratch.reset();
        let mut stages = Vec::with_capacity(plan.pipeline.stages.len());
        for stage in &plan.pipeline.stages {
            let mut kernels = Vec::with_capacity(stage.kernels.len());
            for kernel in &stage.kernels {
                let factory = self
                    .factories
                    .get(&kernel.kind_id)
                    .ok_or(BraidError::BackendRejectedKernel(kernel.kind_id))?;
                kernels.push(factory.prepare(kernel, scratch)?);
            }
            stages.push(PreparedStage { kernels });
        }
        Ok(CpuPrepared { stages })
    }

    fn run_stage(
        &self,
        prepared: &Self::Prepared,
        stage_index: usize,
        _stage: &StageSpec,
        packet: &mut JobPacket,
        cancel: &CancelFlag,
    ) -> BraidResult<()> {
        let prepared_stage = prepared
            .stages
            .get(stage_index)
            .ok_or_else(|| BraidError::from("prepared stage missing"))?;

        for kernel in &prepared_stage.kernels {
            if cancel.is_cancelled() {
                return Err(BraidError::Cancelled);
            }
            kernel.run(packet, cancel)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::CpuComputeBackend;
    use super::CpuKernel;
    use crate::BraidResult;
    use crate::compute::ComputeBackend;
    use crate::job::JobPacket;
    use crate::pipeline::{
        CompiledPlan, DispatchHint, KernelKind, KernelSpec, PipelineShape, StageSpec,
    };
    use crate::scratch::ComputeScratch;
    use crate::{BraidError, CpuKernelFactory};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[test]
    fn rejects_unknown_kernel_kind() {
        let backend = CpuComputeBackend::default();
        let plan = CompiledPlan {
            pipeline: PipelineShape {
                buffers: Vec::new(),
                stages: vec![StageSpec {
                    kernels: vec![KernelSpec {
                        kind_id: KernelKind(999),
                        payload: Arc::from([]),
                        bindings: Vec::new(),
                        dispatch: crate::pipeline::DispatchHint::Serial,
                    }],
                }],
            },
            static_buffers: Vec::new(),
            planner_meta: (),
        };

        let err = backend
            .prepare(&plan, None, &mut ComputeScratch::default())
            .unwrap_err();
        assert!(matches!(
            err,
            BraidError::BackendRejectedKernel(kind) if kind == KernelKind(999)
        ));
    }

    #[derive(Default)]
    struct CountingKernel {
        called: Arc<AtomicBool>,
    }

    impl CpuKernel for CountingKernel {
        fn run(
            &self,
            _packet: &mut JobPacket,
            _cancel: &crate::job::CancelFlag,
        ) -> BraidResult<()> {
            self.called.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    #[derive(Clone)]
    struct CountingKernelFactory {
        called: Arc<AtomicUsize>,
        ran: Arc<AtomicBool>,
    }

    impl CpuKernelFactory for CountingKernelFactory {
        fn prepare(
            &self,
            _kernel: &KernelSpec,
            _scratch: &mut ComputeScratch,
        ) -> BraidResult<Box<dyn CpuKernel>> {
            self.called.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(CountingKernel {
                called: Arc::clone(&self.ran),
            }))
        }
    }

    fn plan_with_kernel(kind_id: KernelKind) -> CompiledPlan<()> {
        CompiledPlan {
            pipeline: PipelineShape {
                buffers: Vec::new(),
                stages: vec![StageSpec {
                    kernels: vec![KernelSpec {
                        kind_id,
                        payload: Vec::new().into(),
                        bindings: Vec::new(),
                        dispatch: DispatchHint::Serial,
                    }],
                }],
            },
            static_buffers: Vec::new(),
            planner_meta: (),
        }
    }

    #[test]
    fn run_stage_short_circuits_when_cancelled() {
        let called = Arc::new(AtomicBool::new(false));
        let called_count = Arc::new(AtomicUsize::new(0));
        let backend = {
            let mut backend = CpuComputeBackend::new();
            backend.register_factory(
                KernelKind(1),
                Arc::new(CountingKernelFactory {
                    called: Arc::clone(&called_count),
                    ran: Arc::clone(&called),
                }),
            );
            backend
        };
        let mut scratch = ComputeScratch::default();
        let prepared = backend
            .prepare(&plan_with_kernel(KernelKind(1)), None, &mut scratch)
            .expect("prepare");
        let mut packet = crate::job::JobPacket::default();
        let cancel = crate::job::CancelFlag::default();
        cancel.cancel();

        let err = backend
            .run_stage(&prepared, 0, &StageSpec::default(), &mut packet, &cancel)
            .expect_err("cancelled");
        assert!(matches!(err, BraidError::Cancelled));
        assert_eq!(called_count.load(Ordering::SeqCst), 1);
        assert!(!called.load(Ordering::SeqCst));
    }

    #[test]
    fn run_stage_errors_on_missing_stage() {
        let called_count = Arc::new(AtomicUsize::new(0));
        let called = Arc::new(AtomicBool::new(false));
        let mut backend = CpuComputeBackend::new();
        backend.register_factory(
            KernelKind(1),
            Arc::new(CountingKernelFactory {
                called: Arc::clone(&called_count),
                ran: Arc::clone(&called),
            }),
        );
        let mut scratch = ComputeScratch::default();
        let prepared = backend
            .prepare(&plan_with_kernel(KernelKind(1)), None, &mut scratch)
            .expect("prepare");
        let mut packet = JobPacket::default();
        let cancel = crate::job::CancelFlag::default();

        let err = backend
            .run_stage(&prepared, 99, &StageSpec::default(), &mut packet, &cancel)
            .expect_err("missing stage");
        assert!(err.to_string().contains("prepared stage missing"));
    }
}
