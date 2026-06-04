use std::sync::Arc;

pub type JobId = u64;
pub type VersionId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementKind {
    U32,
    U64,
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferLayout {
    PerQueryScalar,
    PerQueryVector { width: usize },
    Dynamic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferAccess {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchHint {
    WholeBatch,
    QuerySharded,
    Serial,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferSpec {
    pub slot: u16,
    pub element_kind: ElementKind,
    pub layout: BufferLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferBinding {
    pub slot: u16,
    pub access: BufferAccess,
}

#[derive(Clone, Debug)]
pub struct KernelSpec {
    pub kind_id: u32,
    pub payload: Arc<[u8]>,
    pub bindings: Vec<BufferBinding>,
    pub dispatch: DispatchHint,
}

#[derive(Clone, Debug, Default)]
pub struct StageSpec {
    pub kernels: Vec<KernelSpec>,
}

#[derive(Clone, Debug, Default)]
pub struct PipelineShape {
    pub buffers: Vec<BufferSpec>,
    pub stages: Vec<StageSpec>,
}

#[derive(Clone, Debug)]
pub enum BufferData {
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
}

impl BufferData {
    pub fn kind(&self) -> ElementKind {
        match self {
            Self::U32(_) => ElementKind::U32,
            Self::U64(_) => ElementKind::U64,
            Self::F32(_) => ElementKind::F32,
        }
    }

    pub fn clear(&mut self) {
        match self {
            Self::U32(vals) => vals.clear(),
            Self::U64(vals) => vals.clear(),
            Self::F32(vals) => vals.clear(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::U32(vals) => vals.len(),
            Self::U64(vals) => vals.len(),
            Self::F32(vals) => vals.len(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct StaticBuffer {
    pub slot: u16,
    pub data: BufferData,
}

pub type StaticBufferSet = Vec<StaticBuffer>;

#[derive(Clone, Debug)]
pub struct CompiledPlan<M> {
    pub pipeline: PipelineShape,
    pub static_buffers: StaticBufferSet,
    pub planner_meta: M,
}
