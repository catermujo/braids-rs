use crate::compute::ComputeBackend;
use crate::error::{BraidError, BraidResult};
use crate::job::{CancelFlag, JobPacket};
use crate::pipeline::{CompiledPlan, KernelSpec, StageSpec};
use crate::scratch::ComputeScratch;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

pub trait CpuKernel: Send + Sync {
    fn run(&self, packet: &mut JobPacket, cancel: &CancelFlag) -> BraidResult<()>;
}

pub trait CpuKernelFactory: Send + Sync {
    fn prepare(
        &self,
        kernel: &KernelSpec,
        scratch: &mut ComputeScratch,
    ) -> BraidResult<Box<dyn CpuKernel>>;
}

#[derive(Default)]
pub struct CpuComputeBackend {
    factories: HashMap<u32, Arc<dyn CpuKernelFactory>>,
}

impl CpuComputeBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_factory(
        &mut self,
        kind_id: u32,
        factory: Arc<dyn CpuKernelFactory>,
    ) -> &mut Self {
        self.factories.insert(kind_id, factory);
        self
    }
}

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
    use crate::compute::ComputeBackend;
    use crate::error::BraidError;
    use crate::pipeline::{CompiledPlan, KernelSpec, PipelineShape, StageSpec};
    use crate::scratch::ComputeScratch;
    use std::sync::Arc;

    #[test]
    fn rejects_unknown_kernel_kind() {
        let backend = CpuComputeBackend::default();
        let plan = CompiledPlan {
            pipeline: PipelineShape {
                buffers: Vec::new(),
                stages: vec![StageSpec {
                    kernels: vec![KernelSpec {
                        kind_id: 999,
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
        assert!(matches!(err, BraidError::BackendRejectedKernel(999)));
    }
}
