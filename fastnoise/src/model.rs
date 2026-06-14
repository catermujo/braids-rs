use crate::fastnoise_lite::FastNoiseLite;
use braids::{BraidResult, BufferSlot, CpuComputeBackend, KernelKind, SlotKey, SlotTable};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

pub type FastNoiseCpuBackend = CpuComputeBackend;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FastNoiseKernel {
    InitGrid2d,
    InitGrid3d,
    Warp2d,
    Warp3d,
    Sample2d,
    Sample3d,
    Combine,
}

impl FastNoiseKernel {
    pub(crate) const fn kind(self) -> KernelKind {
        match self {
            Self::InitGrid2d => KernelKind(0xF001),
            Self::InitGrid3d => KernelKind(0xF002),
            Self::Warp2d => KernelKind(0xF100),
            Self::Warp3d => KernelKind(0xF101),
            Self::Sample2d => KernelKind(0xF200),
            Self::Sample3d => KernelKind(0xF201),
            Self::Combine => KernelKind(0xF300),
        }
    }

    pub(crate) fn from_kind(value: KernelKind) -> Option<Self> {
        match value.0 {
            0xF001 => Some(Self::InitGrid2d),
            0xF002 => Some(Self::InitGrid3d),
            0xF100 => Some(Self::Warp2d),
            0xF101 => Some(Self::Warp3d),
            0xF200 => Some(Self::Sample2d),
            0xF201 => Some(Self::Sample3d),
            0xF300 => Some(Self::Combine),
            _ => None,
        }
    }
}

pub(crate) const SLOT_QUERY_META: BufferSlot = BufferSlot(0);
pub(crate) const SLOT_QUERY_F32: BufferSlot = BufferSlot(1);
pub(crate) const SLOT_QUERY_OFFSETS: BufferSlot = BufferSlot(2);
pub(crate) const SLOT_BASE_X: BufferSlot = BufferSlot(10);
pub(crate) const SLOT_BASE_Y: BufferSlot = BufferSlot(11);
pub(crate) const SLOT_BASE_Z: BufferSlot = BufferSlot(12);
pub(crate) const SLOT_DYNAMIC_START: u16 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphDimension {
    D2,
    D3,
}

#[derive(Clone, Debug)]
pub struct FastNoiseGraphSpec {
    pub dimension: GraphDimension,
    pub nodes: Vec<NodeSpec>,
    pub final_field: String,
}

#[derive(Clone, Debug)]
pub enum PositionSource {
    Base,
    Node(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineOp {
    Add,
    Sub,
    Mul,
    Min,
    Max,
    Clamp,
    Remap,
    YGradient,
}

#[derive(Clone)]
pub struct Warp2DNode {
    pub id: String,
    pub source: PositionSource,
    pub noise: Arc<FastNoiseLite>,
}

#[derive(Clone)]
pub struct Warp3DNode {
    pub id: String,
    pub source: PositionSource,
    pub noise: Arc<FastNoiseLite>,
}

#[derive(Clone)]
pub struct Sample2DNode {
    pub id: String,
    pub source: PositionSource,
    pub noise: Arc<FastNoiseLite>,
}

#[derive(Clone)]
pub struct Sample3DNode {
    pub id: String,
    pub source: PositionSource,
    pub noise: Arc<FastNoiseLite>,
}

struct DebugNoiseArc<'a>(&'a Arc<FastNoiseLite>);

impl fmt::Debug for DebugNoiseArc<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FastNoiseLite")
            .field("ptr", &Arc::as_ptr(self.0))
            .field("strong_count", &Arc::strong_count(self.0))
            .finish()
    }
}

impl fmt::Debug for Warp2DNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Warp2DNode")
            .field("id", &self.id)
            .field("source", &self.source)
            .field("noise", &DebugNoiseArc(&self.noise))
            .finish()
    }
}

impl fmt::Debug for Warp3DNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Warp3DNode")
            .field("id", &self.id)
            .field("source", &self.source)
            .field("noise", &DebugNoiseArc(&self.noise))
            .finish()
    }
}

impl fmt::Debug for Sample2DNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sample2DNode")
            .field("id", &self.id)
            .field("source", &self.source)
            .field("noise", &DebugNoiseArc(&self.noise))
            .finish()
    }
}

impl fmt::Debug for Sample3DNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sample3DNode")
            .field("id", &self.id)
            .field("source", &self.source)
            .field("noise", &DebugNoiseArc(&self.noise))
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct CombineNode {
    pub id: String,
    pub inputs: Vec<String>,
    pub op: CombineOp,
    pub params: Vec<f32>,
}

#[derive(Clone, Debug)]
pub enum NodeSpec {
    Warp2D(Warp2DNode),
    Warp3D(Warp3DNode),
    Sample2D(Sample2DNode),
    Sample3D(Sample3DNode),
    Combine(CombineNode),
}

#[derive(Clone, Debug)]
pub enum FastNoiseChange {
    UpsertNode(NodeSpec),
    RemoveNode { id: String },
    SetFinalField { id: String },
}

#[derive(Clone, Debug)]
pub enum ChunkQuery {
    Grid2D {
        width: usize,
        height: usize,
        origin: [f32; 2],
        step: [f32; 2],
    },
    Grid3D {
        width: usize,
        height: usize,
        depth: usize,
        origin: [f32; 3],
        step: [f32; 3],
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChunkSummary {
    pub samples: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub checksum: u64,
    pub taps: [f32; 8],
}

#[derive(Default)]
pub struct FastNoisePlanner;

#[derive(Clone, Debug)]
pub struct FastNoisePlannerMeta {
    pub dimension: GraphDimension,
    pub final_slot: BufferSlot,
}

pub struct FastNoiseState {
    pub(crate) dimension: GraphDimension,
    pub(crate) final_field: String,
    pub(crate) nodes: SlotTable<NodeSpec>,
    pub(crate) node_keys: HashMap<String, SlotKey>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PositionSlots<const N: usize> {
    pub(crate) coords: [BufferSlot; N],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutputKind {
    Position(GraphDimension),
    Scalar(GraphDimension),
}

impl NodeSpec {
    pub(crate) fn id(&self) -> &str {
        match self {
            Self::Warp2D(node) => node.id.as_str(),
            Self::Warp3D(node) => node.id.as_str(),
            Self::Sample2D(node) => node.id.as_str(),
            Self::Sample3D(node) => node.id.as_str(),
            Self::Combine(node) => node.id.as_str(),
        }
    }

    pub(crate) fn output_kind(&self, graph_dimension: GraphDimension) -> OutputKind {
        match self {
            Self::Warp2D(_) => OutputKind::Position(GraphDimension::D2),
            Self::Warp3D(_) => OutputKind::Position(GraphDimension::D3),
            Self::Sample2D(_) => OutputKind::Scalar(GraphDimension::D2),
            Self::Sample3D(_) => OutputKind::Scalar(GraphDimension::D3),
            Self::Combine(_) => OutputKind::Scalar(graph_dimension),
        }
    }
}

impl ChunkQuery {
    pub fn samples(&self) -> BraidResult<usize> {
        Ok(match self {
            Self::Grid2D { width, height, .. } => width * height,
            Self::Grid3D {
                width,
                height,
                depth,
                ..
            } => width * height * depth,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CombineNode, CombineOp, FastNoiseKernel, GraphDimension, NodeSpec, PositionSource,
        Sample3DNode, Warp2DNode, Warp3DNode,
    };
    use crate::fastnoise_lite::FastNoiseLite;
    use std::sync::Arc;

    #[test]
    fn fastnoise_kernel_roundtrips_kind_mapping() {
        use braids::KernelKind;
        let pairs = [
            (FastNoiseKernel::InitGrid2d, 0xF001),
            (FastNoiseKernel::InitGrid3d, 0xF002),
            (FastNoiseKernel::Warp2d, 0xF100),
            (FastNoiseKernel::Warp3d, 0xF101),
            (FastNoiseKernel::Sample2d, 0xF200),
            (FastNoiseKernel::Sample3d, 0xF201),
            (FastNoiseKernel::Combine, 0xF300),
        ];
        for (kind, raw) in pairs {
            assert_eq!(kind.kind().0, raw);
            assert_eq!(FastNoiseKernel::from_kind(KernelKind(raw)), Some(kind));
        }
        assert_eq!(FastNoiseKernel::from_kind(KernelKind(0xDEAD)), None);
    }

    #[test]
    fn chunk_query_samples_matches_dimension_shape() {
        let two_d = super::ChunkQuery::Grid2D {
            width: 3,
            height: 4,
            origin: [0.0, 0.0],
            step: [1.0, 1.0],
        };
        let three_d = super::ChunkQuery::Grid3D {
            width: 2,
            height: 3,
            depth: 5,
            origin: [0.0, 0.0, 0.0],
            step: [1.0, 1.0, 1.0],
        };
        assert_eq!(two_d.samples().expect("2d samples"), 12);
        assert_eq!(three_d.samples().expect("3d samples"), 30);
    }

    #[test]
    fn node_spec_id_and_output_kind_reflect_shape() {
        let warp_2d = NodeSpec::Warp2D(Warp2DNode {
            id: "w2d".to_owned(),
            source: PositionSource::Base,
            noise: Arc::new(FastNoiseLite::with_seed(1)),
        });
        let sample_3d = NodeSpec::Sample3D(Sample3DNode {
            id: "s3d".to_owned(),
            source: PositionSource::Base,
            noise: Arc::new(FastNoiseLite::with_seed(2)),
        });
        let combine = NodeSpec::Combine(CombineNode {
            id: "blend".to_owned(),
            inputs: vec!["w2d".to_owned(), "s3d".to_owned()],
            op: CombineOp::Add,
            params: Vec::new(),
        });
        let warp_3d = NodeSpec::Warp3D(Warp3DNode {
            id: "w3d".to_owned(),
            source: PositionSource::Base,
            noise: Arc::new(FastNoiseLite::with_seed(3)),
        });

        assert_eq!(warp_2d.id(), "w2d");
        assert_eq!(sample_3d.id(), "s3d");
        assert_eq!(combine.id(), "blend");
        assert_eq!(warp_3d.id(), "w3d");
        assert!(matches!(
            warp_2d.output_kind(GraphDimension::D2),
            super::OutputKind::Position(GraphDimension::D2)
        ));
        assert!(matches!(
            sample_3d.output_kind(GraphDimension::D3),
            super::OutputKind::Scalar(GraphDimension::D3)
        ));
        assert!(matches!(
            combine.output_kind(GraphDimension::D3),
            super::OutputKind::Scalar(GraphDimension::D3)
        ));
    }
}
