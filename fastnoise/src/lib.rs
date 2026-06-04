mod fastnoise_lite;

pub use fastnoise_lite::{
    CellularDistanceFunction, CellularReturnType, DomainWarpType, FastNoiseLite, FractalType,
    NoiseType, RotationType3D,
};

use braid::{
    BatchScratch, BraidError, BraidResult, BufferBinding, BufferSlot, BufferSpec, CancelFlag,
    CompiledPlan, ComputeScratch, CpuComputeBackend, CpuKernel, CpuKernelFactory, ElementKind,
    JobPacket, KernelKind, KernelSpec, PlannerBackend, PlannerScratch, SlotKey, SlotTable,
    StageSpec,
};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

pub type FastNoiseCpuBackend = CpuComputeBackend;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FastNoiseKernel {
    InitGrid2d,
    InitGrid3d,
    Warp2d,
    Warp3d,
    Sample2d,
    Sample3d,
    Combine,
}

impl FastNoiseKernel {
    const fn kind(self) -> KernelKind {
        match self {
            Self::InitGrid2d => KernelKind::new(0xF001),
            Self::InitGrid3d => KernelKind::new(0xF002),
            Self::Warp2d => KernelKind::new(0xF100),
            Self::Warp3d => KernelKind::new(0xF101),
            Self::Sample2d => KernelKind::new(0xF200),
            Self::Sample3d => KernelKind::new(0xF201),
            Self::Combine => KernelKind::new(0xF300),
        }
    }
}

impl From<FastNoiseKernel> for KernelKind {
    fn from(value: FastNoiseKernel) -> Self {
        value.kind()
    }
}

impl TryFrom<KernelKind> for FastNoiseKernel {
    type Error = ();

    fn try_from(value: KernelKind) -> Result<Self, Self::Error> {
        match value.raw() {
            0xF001 => Ok(Self::InitGrid2d),
            0xF002 => Ok(Self::InitGrid3d),
            0xF100 => Ok(Self::Warp2d),
            0xF101 => Ok(Self::Warp3d),
            0xF200 => Ok(Self::Sample2d),
            0xF201 => Ok(Self::Sample3d),
            0xF300 => Ok(Self::Combine),
            _ => Err(()),
        }
    }
}

const SLOT_QUERY_META: BufferSlot = BufferSlot::new(0);
const SLOT_QUERY_F32: BufferSlot = BufferSlot::new(1);
const SLOT_QUERY_OFFSETS: BufferSlot = BufferSlot::new(2);
const SLOT_BASE_X: BufferSlot = BufferSlot::new(10);
const SLOT_BASE_Y: BufferSlot = BufferSlot::new(11);
const SLOT_BASE_Z: BufferSlot = BufferSlot::new(12);
const SLOT_DYNAMIC_START: u16 = 32;

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

#[derive(Clone, Debug)]
pub struct Warp2DNode {
    pub id: String,
    pub source: PositionSource,
    pub noise: FastNoiseLite,
}

#[derive(Clone, Debug)]
pub struct Warp3DNode {
    pub id: String,
    pub source: PositionSource,
    pub noise: FastNoiseLite,
}

#[derive(Clone, Debug)]
pub struct Sample2DNode {
    pub id: String,
    pub source: PositionSource,
    pub noise: FastNoiseLite,
}

#[derive(Clone, Debug)]
pub struct Sample3DNode {
    pub id: String,
    pub source: PositionSource,
    pub noise: FastNoiseLite,
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

pub fn make_cpu_backend() -> FastNoiseCpuBackend {
    let factory: Arc<dyn CpuKernelFactory> = Arc::new(FastNoiseKernelFactory);
    let mut backend = CpuComputeBackend::new();
    for kind in [
        FastNoiseKernel::InitGrid2d,
        FastNoiseKernel::InitGrid3d,
        FastNoiseKernel::Warp2d,
        FastNoiseKernel::Warp3d,
        FastNoiseKernel::Sample2d,
        FastNoiseKernel::Sample3d,
        FastNoiseKernel::Combine,
    ] {
        backend.register_factory(kind.into(), Arc::clone(&factory));
    }
    backend
}

#[derive(Clone, Debug)]
pub struct FastNoisePlannerMeta {
    dimension: GraphDimension,
    final_slot: BufferSlot,
}

#[derive(Clone)]
pub struct FastNoiseState {
    dimension: GraphDimension,
    final_field: String,
    nodes: SlotTable<NodeSpec>,
    node_keys: HashMap<String, SlotKey>,
}

#[derive(Clone, Copy, Debug)]
struct PositionSlots {
    x: BufferSlot,
    y: BufferSlot,
    z: Option<BufferSlot>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputKind {
    Position(GraphDimension),
    Scalar(GraphDimension),
}

struct FastNoiseKernelFactory;

struct GridInitKernel {
    dimension: GraphDimension,
}

struct WarpKernel {
    dimension: GraphDimension,
    source: PositionSlots,
    output: PositionSlots,
    noise: FastNoiseLite,
}

struct SampleKernel {
    dimension: GraphDimension,
    source: PositionSlots,
    output: BufferSlot,
    noise: FastNoiseLite,
}

struct CombineKernel {
    op: CombineOp,
    inputs: Vec<BufferSlot>,
    output: BufferSlot,
    params: Vec<f32>,
}

#[derive(Clone)]
struct WarpPayload {
    dimension: GraphDimension,
    source: PositionSlots,
    output: PositionSlots,
    noise: FastNoiseLite,
}

#[derive(Clone)]
struct SamplePayload {
    dimension: GraphDimension,
    source: PositionSlots,
    output: BufferSlot,
    noise: FastNoiseLite,
}

#[derive(Clone)]
struct CombinePayload {
    op: CombineOp,
    inputs: Vec<BufferSlot>,
    output: BufferSlot,
    params: Vec<f32>,
}

trait KernelPayload: Sized {
    fn kind(&self) -> FastNoiseKernel;
    fn encode_into(&self, writer: &mut PayloadWriter<'_>) -> BraidResult<()>;
    fn decode_from(kind: FastNoiseKernel, reader: &mut PayloadReader<'_>) -> BraidResult<Self>;

    fn encode(&self, scratch: &mut PlannerScratch) -> BraidResult<KernelSpec> {
        scratch.reset();
        let mut writer = PayloadWriter::new(&mut scratch.bytes);
        self.encode_into(&mut writer)?;
        Ok(KernelSpec::new(self.kind().into(), scratch.bytes.clone()))
    }

    fn decode(kind: FastNoiseKernel, bytes: &[u8]) -> BraidResult<Self> {
        let mut reader = PayloadReader::new(bytes);
        let payload = Self::decode_from(kind, &mut reader)?;
        reader.finish()?;
        Ok(payload)
    }
}

impl WarpPayload {
    fn into_kernel(self) -> WarpKernel {
        WarpKernel {
            dimension: self.dimension,
            source: self.source,
            output: self.output,
            noise: self.noise,
        }
    }
}

impl KernelPayload for WarpPayload {
    fn kind(&self) -> FastNoiseKernel {
        match self.dimension {
            GraphDimension::D2 => FastNoiseKernel::Warp2d,
            GraphDimension::D3 => FastNoiseKernel::Warp3d,
        }
    }

    fn encode_into(&self, writer: &mut PayloadWriter<'_>) -> BraidResult<()> {
        writer.slot(self.source.x);
        writer.slot(self.source.y);
        if self.dimension == GraphDimension::D3 {
            writer.slot(expect_slot(self.source.z, "warp payload source z")?);
        }
        writer.slot(self.output.x);
        writer.slot(self.output.y);
        if self.dimension == GraphDimension::D3 {
            writer.slot(expect_slot(self.output.z, "warp payload output z")?);
        }
        writer.noise(&self.noise);
        Ok(())
    }

    fn decode_from(kind: FastNoiseKernel, reader: &mut PayloadReader<'_>) -> BraidResult<Self> {
        let dimension = match kind {
            FastNoiseKernel::Warp2d => GraphDimension::D2,
            FastNoiseKernel::Warp3d => GraphDimension::D3,
            _ => {
                return Err(BraidError::InvalidSpec(
                    "warp payload received wrong kernel kind".to_owned(),
                ))
            }
        };
        Ok(Self {
            dimension,
            source: PositionSlots {
                x: reader.slot()?,
                y: reader.slot()?,
                z: match dimension {
                    GraphDimension::D2 => None,
                    GraphDimension::D3 => Some(reader.slot()?),
                },
            },
            output: PositionSlots {
                x: reader.slot()?,
                y: reader.slot()?,
                z: match dimension {
                    GraphDimension::D2 => None,
                    GraphDimension::D3 => Some(reader.slot()?),
                },
            },
            noise: reader.noise()?,
        })
    }
}

impl SamplePayload {
    fn into_kernel(self) -> SampleKernel {
        SampleKernel {
            dimension: self.dimension,
            source: self.source,
            output: self.output,
            noise: self.noise,
        }
    }
}

impl KernelPayload for SamplePayload {
    fn kind(&self) -> FastNoiseKernel {
        match self.dimension {
            GraphDimension::D2 => FastNoiseKernel::Sample2d,
            GraphDimension::D3 => FastNoiseKernel::Sample3d,
        }
    }

    fn encode_into(&self, writer: &mut PayloadWriter<'_>) -> BraidResult<()> {
        writer.slot(self.source.x);
        writer.slot(self.source.y);
        if self.dimension == GraphDimension::D3 {
            writer.slot(expect_slot(self.source.z, "sample payload source z")?);
        }
        writer.slot(self.output);
        writer.noise(&self.noise);
        Ok(())
    }

    fn decode_from(kind: FastNoiseKernel, reader: &mut PayloadReader<'_>) -> BraidResult<Self> {
        let dimension = match kind {
            FastNoiseKernel::Sample2d => GraphDimension::D2,
            FastNoiseKernel::Sample3d => GraphDimension::D3,
            _ => {
                return Err(BraidError::InvalidSpec(
                    "sample payload received wrong kernel kind".to_owned(),
                ))
            }
        };
        Ok(Self {
            dimension,
            source: PositionSlots {
                x: reader.slot()?,
                y: reader.slot()?,
                z: match dimension {
                    GraphDimension::D2 => None,
                    GraphDimension::D3 => Some(reader.slot()?),
                },
            },
            output: reader.slot()?,
            noise: reader.noise()?,
        })
    }
}

impl CombinePayload {
    fn into_kernel(self) -> CombineKernel {
        CombineKernel {
            op: self.op,
            inputs: self.inputs,
            output: self.output,
            params: self.params,
        }
    }
}

impl KernelPayload for CombinePayload {
    fn kind(&self) -> FastNoiseKernel {
        FastNoiseKernel::Combine
    }

    fn encode_into(&self, writer: &mut PayloadWriter<'_>) -> BraidResult<()> {
        writer.u32(encode_combine_op(self.op));
        writer.u32(usize_to_u32(self.inputs.len(), "combine inputs")?);
        for slot in &self.inputs {
            writer.slot(*slot);
        }
        writer.slot(self.output);
        writer.u32(usize_to_u32(self.params.len(), "combine params")?);
        for value in &self.params {
            writer.f32(*value);
        }
        Ok(())
    }

    fn decode_from(kind: FastNoiseKernel, reader: &mut PayloadReader<'_>) -> BraidResult<Self> {
        if kind != FastNoiseKernel::Combine {
            return Err(BraidError::InvalidSpec(
                "combine payload received wrong kernel kind".to_owned(),
            ));
        }
        let op = decode_combine_op(reader.u32()?)?;
        let input_count = usize::try_from(reader.u32()?)
            .map_err(|_| BraidError::InvalidSpec("combine input count overflow".to_owned()))?;
        let mut inputs = Vec::with_capacity(input_count);
        for _ in 0..input_count {
            inputs.push(reader.slot()?);
        }
        let output = reader.slot()?;
        let param_count = usize::try_from(reader.u32()?)
            .map_err(|_| BraidError::InvalidSpec("combine param count overflow".to_owned()))?;
        let mut params = Vec::with_capacity(param_count);
        for _ in 0..param_count {
            params.push(reader.f32()?);
        }
        Ok(Self {
            op,
            inputs,
            output,
            params,
        })
    }
}

impl PlannerBackend for FastNoisePlanner {
    type Spec = FastNoiseGraphSpec;
    type State = FastNoiseState;
    type Change = FastNoiseChange;
    type Query = ChunkQuery;
    type Resolution = ChunkSummary;
    type PlannerMeta = FastNoisePlannerMeta;

    fn init_state(&self, spec: &Self::Spec) -> BraidResult<Self::State> {
        FastNoiseState::from_spec(spec)
    }

    fn reset_state(&self, state: &mut Self::State, spec: &Self::Spec) -> BraidResult<()> {
        state.reset_from_spec(spec)
    }

    fn apply(&self, state: &mut Self::State, changes: &[Self::Change]) -> BraidResult<()> {
        for change in changes {
            match change {
                FastNoiseChange::UpsertNode(node) => state.upsert_node(node.clone())?,
                FastNoiseChange::RemoveNode { id } => state.remove_node(id)?,
                FastNoiseChange::SetFinalField { id } => state.final_field = id.clone(),
            }
        }
        Ok(())
    }

    fn compile(
        &self,
        state: &Self::State,
        scratch: &mut PlannerScratch,
    ) -> BraidResult<CompiledPlan<Self::PlannerMeta>> {
        compile_graph(state, scratch)
    }

    fn encode_batch(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        queries: &[Self::Query],
        packet: &mut JobPacket,
        scratch: &mut BatchScratch,
    ) -> BraidResult<()> {
        scratch.reset();
        packet.set_query_count(queries.len());

        let mut meta_values = vec![0u32; queries.len() * 3];
        let mut float_values = vec![0.0f32; queries.len() * 6];
        let mut offset_values = vec![0u32; queries.len() + 1];

        let mut cursor = 0usize;
        offset_values[0] = 0;
        for (index, query) in queries.iter().enumerate() {
            let meta_base = index * 3;
            let float_base = index * 6;
            match (plan.planner_meta.dimension, query) {
                (
                    GraphDimension::D2,
                    ChunkQuery::Grid2D {
                        width,
                        height,
                        origin,
                        step,
                    },
                ) => {
                    meta_values[meta_base] = usize_to_u32(*width, "grid2d width")?;
                    meta_values[meta_base + 1] = usize_to_u32(*height, "grid2d height")?;
                    meta_values[meta_base + 2] = 1;
                    float_values[float_base] = origin[0];
                    float_values[float_base + 1] = origin[1];
                    float_values[float_base + 2] = 0.0;
                    float_values[float_base + 3] = step[0];
                    float_values[float_base + 4] = step[1];
                    float_values[float_base + 5] = 1.0;
                    cursor = cursor
                        .checked_add(sample_count_2d(*width, *height)?)
                        .ok_or_else(|| {
                            BraidError::InvalidSpec("grid2d sample count overflow".to_owned())
                        })?;
                }
                (
                    GraphDimension::D3,
                    ChunkQuery::Grid3D {
                        width,
                        height,
                        depth,
                        origin,
                        step,
                    },
                ) => {
                    meta_values[meta_base] = usize_to_u32(*width, "grid3d width")?;
                    meta_values[meta_base + 1] = usize_to_u32(*height, "grid3d height")?;
                    meta_values[meta_base + 2] = usize_to_u32(*depth, "grid3d depth")?;
                    float_values[float_base] = origin[0];
                    float_values[float_base + 1] = origin[1];
                    float_values[float_base + 2] = origin[2];
                    float_values[float_base + 3] = step[0];
                    float_values[float_base + 4] = step[1];
                    float_values[float_base + 5] = step[2];
                    cursor = cursor
                        .checked_add(sample_count_3d(*width, *height, *depth)?)
                        .ok_or_else(|| {
                            BraidError::InvalidSpec("grid3d sample count overflow".to_owned())
                        })?;
                }
                (GraphDimension::D2, ChunkQuery::Grid3D { .. }) => {
                    return Err(BraidError::InvalidSpec(
                        "2d graph received 3d chunk query".to_owned(),
                    ));
                }
                (GraphDimension::D3, ChunkQuery::Grid2D { .. }) => {
                    return Err(BraidError::InvalidSpec(
                        "3d graph received 2d chunk query".to_owned(),
                    ));
                }
            }
            offset_values[index + 1] = usize_to_u32(cursor, "query offsets")?;
        }

        packet
            .ensure_u32(SLOT_QUERY_META, meta_values.len())
            .copy_from_slice(meta_values.as_slice());
        packet
            .ensure_f32(SLOT_QUERY_F32, float_values.len())
            .copy_from_slice(float_values.as_slice());
        packet
            .ensure_u32(SLOT_QUERY_OFFSETS, offset_values.len())
            .copy_from_slice(offset_values.as_slice());

        Ok(())
    }

    fn decode_batch(
        &self,
        plan: &CompiledPlan<Self::PlannerMeta>,
        packet: &JobPacket,
    ) -> BraidResult<Vec<Self::Resolution>> {
        let values = packet.f32(plan.planner_meta.final_slot)?;
        let offsets = packet.u32(SLOT_QUERY_OFFSETS)?;
        let mut summaries = Vec::with_capacity(offsets.len().saturating_sub(1));
        for window in offsets.windows(2) {
            let start = usize::try_from(window[0])
                .map_err(|_| BraidError::InvalidSpec("offset start overflow".to_owned()))?;
            let end = usize::try_from(window[1])
                .map_err(|_| BraidError::InvalidSpec("offset end overflow".to_owned()))?;
            if start > end || end > values.len() {
                return Err(BraidError::InvalidSpec(
                    "final field offset range out of bounds".to_owned(),
                ));
            }
            summaries.push(summarize_samples(&values[start..end]));
        }
        Ok(summaries)
    }
}

impl FastNoiseState {
    fn from_spec(spec: &FastNoiseGraphSpec) -> BraidResult<Self> {
        let mut state = Self {
            dimension: spec.dimension,
            final_field: spec.final_field.clone(),
            nodes: SlotTable::default(),
            node_keys: HashMap::new(),
        };
        state.reset_from_spec(spec)?;
        Ok(state)
    }

    fn reset_from_spec(&mut self, spec: &FastNoiseGraphSpec) -> BraidResult<()> {
        self.dimension = spec.dimension;
        self.final_field = spec.final_field.clone();
        self.nodes.clear_reuse();
        self.node_keys.clear();
        for node in &spec.nodes {
            self.upsert_node(node.clone())?;
        }
        Ok(())
    }

    fn upsert_node(&mut self, node: NodeSpec) -> BraidResult<()> {
        validate_node_id(node.id())?;
        if let Some(existing) = self.node_keys.get(node.id()).copied()
            && let Some(slot) = self.nodes.get_mut(existing)
        {
            *slot = node;
            return Ok(());
        }
        let id = node.id().to_owned();
        let key = self.nodes.insert(node);
        self.node_keys.insert(id, key);
        Ok(())
    }

    fn remove_node(&mut self, id: &str) -> BraidResult<()> {
        let key = self
            .node_keys
            .remove(id)
            .ok_or_else(|| BraidError::MissingReference {
                kind: "node",
                id: id.to_owned(),
                reference: id.to_owned(),
            })?;
        self.nodes.remove(key);
        Ok(())
    }
}

impl NodeSpec {
    fn id(&self) -> &str {
        match self {
            Self::Warp2D(node) => node.id.as_str(),
            Self::Warp3D(node) => node.id.as_str(),
            Self::Sample2D(node) => node.id.as_str(),
            Self::Sample3D(node) => node.id.as_str(),
            Self::Combine(node) => node.id.as_str(),
        }
    }

    fn output_kind(&self, graph_dimension: GraphDimension) -> OutputKind {
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
        match self {
            Self::Grid2D { width, height, .. } => sample_count_2d(*width, *height),
            Self::Grid3D {
                width,
                height,
                depth,
                ..
            } => sample_count_3d(*width, *height, *depth),
        }
    }
}

impl CpuKernelFactory for FastNoiseKernelFactory {
    fn prepare(
        &self,
        kernel: &KernelSpec,
        _scratch: &mut ComputeScratch,
    ) -> BraidResult<Box<dyn CpuKernel>> {
        let kind = FastNoiseKernel::try_from(kernel.kind_id)
            .map_err(|_| BraidError::BackendRejectedKernel(kernel.kind_id))?;
        match kind {
            FastNoiseKernel::InitGrid2d => Ok(Box::new(GridInitKernel {
                dimension: GraphDimension::D2,
            })),
            FastNoiseKernel::InitGrid3d => Ok(Box::new(GridInitKernel {
                dimension: GraphDimension::D3,
            })),
            FastNoiseKernel::Warp2d => Ok(Box::new(
                WarpPayload::decode(kind, &kernel.payload)?.into_kernel(),
            )),
            FastNoiseKernel::Warp3d => Ok(Box::new(
                WarpPayload::decode(kind, &kernel.payload)?.into_kernel(),
            )),
            FastNoiseKernel::Sample2d => Ok(Box::new(
                SamplePayload::decode(kind, &kernel.payload)?.into_kernel(),
            )),
            FastNoiseKernel::Sample3d => Ok(Box::new(
                SamplePayload::decode(kind, &kernel.payload)?.into_kernel(),
            )),
            FastNoiseKernel::Combine => Ok(Box::new(
                CombinePayload::decode(kind, &kernel.payload)?.into_kernel(),
            )),
        }
    }
}

impl CpuKernel for GridInitKernel {
    fn run(&self, packet: &mut JobPacket, cancel: &CancelFlag) -> BraidResult<()> {
        let meta = packet.u32(SLOT_QUERY_META)?.to_vec();
        let floats = packet.f32(SLOT_QUERY_F32)?.to_vec();
        let offsets = packet.u32(SLOT_QUERY_OFFSETS)?.to_vec();
        let total = total_samples_from_offsets(offsets.as_slice())?;
        let query_count = packet.query_count();
        packet.ensure_f32(SLOT_BASE_X, total);
        packet.ensure_f32(SLOT_BASE_Y, total);
        if self.dimension == GraphDimension::D3 {
            packet.ensure_f32(SLOT_BASE_Z, total);
            return packet.with_f32_buffers(&[SLOT_BASE_X, SLOT_BASE_Y, SLOT_BASE_Z], |buffers| {
                let [xs, ys, zs]: [&mut [f32]; 3] = buffers
                    .try_into()
                    .map_err(|_| BraidError::from("init grid3d buffer view mismatch"))?;
                for (query_index, offset_value) in offsets.iter().take(query_count).copied().enumerate()
                {
                    let meta_base = query_index * 3;
                    let float_base = query_index * 6;
                    let width = usize::try_from(meta[meta_base]).map_err(|_| {
                        BraidError::InvalidSpec("grid3d width overflow".to_owned())
                    })?;
                    let height = usize::try_from(meta[meta_base + 1]).map_err(|_| {
                        BraidError::InvalidSpec("grid3d height overflow".to_owned())
                    })?;
                    let depth = usize::try_from(meta[meta_base + 2]).map_err(|_| {
                        BraidError::InvalidSpec("grid3d depth overflow".to_owned())
                    })?;
                    let offset = usize::try_from(offset_value).map_err(|_| {
                        BraidError::InvalidSpec("grid3d offset overflow".to_owned())
                    })?;
                    let origin_x = floats[float_base];
                    let origin_y = floats[float_base + 1];
                    let origin_z = floats[float_base + 2];
                    let step_x = floats[float_base + 3];
                    let step_y = floats[float_base + 4];
                    let step_z = floats[float_base + 5];
                    for z in 0..depth {
                        for y in 0..height {
                            for x in 0..width {
                                let index = offset + ((z * height + y) * width) + x;
                                xs[index] = origin_x + (x as f32 * step_x);
                                ys[index] = origin_y + (y as f32 * step_y);
                                zs[index] = origin_z + (z as f32 * step_z);
                            }
                            if y & 15 == 0 && cancel.is_cancelled() {
                                return Err(BraidError::Cancelled);
                            }
                        }
                    }
                }
                Ok(())
            });
        }
        packet.with_f32_buffers(&[SLOT_BASE_X, SLOT_BASE_Y], |buffers| {
            let [xs, ys]: [&mut [f32]; 2] = buffers
                .try_into()
                .map_err(|_| BraidError::from("init grid2d buffer view mismatch"))?;
            for (query_index, offset_value) in offsets.iter().take(query_count).copied().enumerate()
            {
                let meta_base = query_index * 3;
                let float_base = query_index * 6;
                let width = usize::try_from(meta[meta_base])
                    .map_err(|_| BraidError::InvalidSpec("grid2d width overflow".to_owned()))?;
                let height = usize::try_from(meta[meta_base + 1])
                    .map_err(|_| BraidError::InvalidSpec("grid2d height overflow".to_owned()))?;
                let offset = usize::try_from(offset_value)
                    .map_err(|_| BraidError::InvalidSpec("grid2d offset overflow".to_owned()))?;
                let origin_x = floats[float_base];
                let origin_y = floats[float_base + 1];
                let step_x = floats[float_base + 3];
                let step_y = floats[float_base + 4];
                for y in 0..height {
                    for x in 0..width {
                        let index = offset + (y * width) + x;
                        xs[index] = origin_x + (x as f32 * step_x);
                        ys[index] = origin_y + (y as f32 * step_y);
                    }
                    if y & 31 == 0 && cancel.is_cancelled() {
                        return Err(BraidError::Cancelled);
                    }
                }
            }
            Ok(())
        })
    }
}

impl CpuKernel for WarpKernel {
    fn run(&self, packet: &mut JobPacket, cancel: &CancelFlag) -> BraidResult<()> {
        let total = total_samples_from_offsets(packet.u32(SLOT_QUERY_OFFSETS)?)?;
        packet.ensure_f32(self.output.x, total);
        packet.ensure_f32(self.output.y, total);
        if self.dimension == GraphDimension::D3 {
            let source_z = expect_slot(self.source.z, "warp source z")?;
            let output_z = expect_slot(self.output.z, "warp output z")?;
            packet.ensure_f32(output_z, total);
            return packet.with_f32_buffers(
                &[
                    self.source.x,
                    self.source.y,
                    source_z,
                    self.output.x,
                    self.output.y,
                    output_z,
                ],
                |buffers| {
                    let [xs, ys, zs, out_xs, out_ys, out_zs]: [&mut [f32]; 6] = buffers
                        .try_into()
                        .map_err(|_| BraidError::from("warp3d buffer view mismatch"))?;
                    for index in 0..out_xs.len() {
                        let (warp_x, warp_y, warp_z) =
                            self.noise.domain_warp_3d(xs[index], ys[index], zs[index]);
                        out_xs[index] = warp_x;
                        out_ys[index] = warp_y;
                        out_zs[index] = warp_z;
                        if index & 2047 == 0 && cancel.is_cancelled() {
                            return Err(BraidError::Cancelled);
                        }
                    }
                    Ok(())
                },
            );
        }
        packet.with_f32_buffers(
            &[self.source.x, self.source.y, self.output.x, self.output.y],
            |buffers| {
                let [xs, ys, out_xs, out_ys]: [&mut [f32]; 4] = buffers
                    .try_into()
                    .map_err(|_| BraidError::from("warp2d buffer view mismatch"))?;
                for index in 0..out_xs.len() {
                    let (warp_x, warp_y) = self.noise.domain_warp_2d(xs[index], ys[index]);
                    out_xs[index] = warp_x;
                    out_ys[index] = warp_y;
                    if index & 4095 == 0 && cancel.is_cancelled() {
                        return Err(BraidError::Cancelled);
                    }
                }
                Ok(())
            },
        )
    }
}

impl CpuKernel for SampleKernel {
    fn run(&self, packet: &mut JobPacket, cancel: &CancelFlag) -> BraidResult<()> {
        let total = total_samples_from_offsets(packet.u32(SLOT_QUERY_OFFSETS)?)?;
        packet.ensure_f32(self.output, total);
        if self.dimension == GraphDimension::D3 {
            let source_z = expect_slot(self.source.z, "sample source z")?;
            return packet.with_f32_buffers(
                &[self.source.x, self.source.y, source_z, self.output],
                |buffers| {
                    let [xs, ys, zs, out]: [&mut [f32]; 4] = buffers
                        .try_into()
                        .map_err(|_| BraidError::from("sample3d buffer view mismatch"))?;
                    for index in 0..out.len() {
                        out[index] = self.noise.get_noise_3d(xs[index], ys[index], zs[index]);
                        if index & 2047 == 0 && cancel.is_cancelled() {
                            return Err(BraidError::Cancelled);
                        }
                    }
                    Ok(())
                },
            );
        }
        packet.with_f32_buffers(&[self.source.x, self.source.y, self.output], |buffers| {
            let [xs, ys, out]: [&mut [f32]; 3] = buffers
                .try_into()
                .map_err(|_| BraidError::from("sample2d buffer view mismatch"))?;
            for index in 0..out.len() {
                out[index] = self.noise.get_noise_2d(xs[index], ys[index]);
                if index & 4095 == 0 && cancel.is_cancelled() {
                    return Err(BraidError::Cancelled);
                }
            }
            Ok(())
        })
    }
}

impl CpuKernel for CombineKernel {
    fn run(&self, packet: &mut JobPacket, cancel: &CancelFlag) -> BraidResult<()> {
        let total = total_samples_from_offsets(packet.u32(SLOT_QUERY_OFFSETS)?)?;
        packet.ensure_f32(self.output, total);
        match self.op {
            CombineOp::Add => self.run_binary(packet, cancel, |a, b| a + b),
            CombineOp::Sub => self.run_binary(packet, cancel, |a, b| a - b),
            CombineOp::Mul => self.run_binary(packet, cancel, |a, b| a * b),
            CombineOp::Min => self.run_binary(packet, cancel, |a, b| a.min(b)),
            CombineOp::Max => self.run_binary(packet, cancel, |a, b| a.max(b)),
            CombineOp::Clamp => {
                let [min_value, max_value] = expect_params(self.params.as_slice(), 2, "clamp")?;
                let input = expect_input(self.inputs.as_slice(), 1, "clamp")?;
                packet.with_f32_buffers(&[input, self.output], |buffers| {
                    let [values, out]: [&mut [f32]; 2] = buffers
                        .try_into()
                        .map_err(|_| BraidError::from("clamp buffer view mismatch"))?;
                    for index in 0..out.len() {
                        out[index] = values[index].clamp(min_value, max_value);
                        if index & 4095 == 0 && cancel.is_cancelled() {
                            return Err(BraidError::Cancelled);
                        }
                    }
                    Ok(())
                })
            }
            CombineOp::Remap => {
                let [src_min, src_max, dst_min, dst_max] =
                    expect_params(self.params.as_slice(), 4, "remap")?;
                let input = expect_input(self.inputs.as_slice(), 1, "remap")?;
                packet.with_f32_buffers(&[input, self.output], |buffers| {
                    let [values, out]: [&mut [f32]; 2] = buffers
                        .try_into()
                        .map_err(|_| BraidError::from("remap buffer view mismatch"))?;
                    for index in 0..out.len() {
                        let denom = src_max - src_min;
                        let t = if denom.abs() <= f32::EPSILON {
                            0.0
                        } else {
                            ((values[index] - src_min) / denom).clamp(0.0, 1.0)
                        };
                        out[index] = dst_min + ((dst_max - dst_min) * t);
                        if index & 4095 == 0 && cancel.is_cancelled() {
                            return Err(BraidError::Cancelled);
                        }
                    }
                    Ok(())
                })
            }
            CombineOp::YGradient => {
                let [y_min, y_max, out_min, out_max] =
                    expect_params(self.params.as_slice(), 4, "ygradient")?;
                let input = expect_input(self.inputs.as_slice(), 1, "ygradient")?;
                packet.with_f32_buffers(&[input, SLOT_BASE_Y, self.output], |buffers| {
                    let [values, ys, out]: [&mut [f32]; 3] = buffers
                        .try_into()
                        .map_err(|_| BraidError::from("ygradient buffer view mismatch"))?;
                    for index in 0..out.len() {
                        let denom = y_max - y_min;
                        let t = if denom.abs() <= f32::EPSILON {
                            0.0
                        } else {
                            ((ys[index] - y_min) / denom).clamp(0.0, 1.0)
                        };
                        let gradient = out_min + ((out_max - out_min) * t);
                        out[index] = values[index] + gradient;
                        if index & 2047 == 0 && cancel.is_cancelled() {
                            return Err(BraidError::Cancelled);
                        }
                    }
                    Ok(())
                })
            }
        }
    }
}

impl CombineKernel {
    fn run_binary(
        &self,
        packet: &mut JobPacket,
        cancel: &CancelFlag,
        op: impl Fn(f32, f32) -> f32,
    ) -> BraidResult<()> {
        let [left, right] = expect_two_inputs(self.inputs.as_slice(), "binary combine")?;
        packet.with_f32_buffers(&[left, right, self.output], |buffers| {
            let [lhs, rhs, out]: [&mut [f32]; 3] = buffers
                .try_into()
                .map_err(|_| BraidError::from("binary combine buffer view mismatch"))?;
            for index in 0..out.len() {
                out[index] = op(lhs[index], rhs[index]);
                if index & 4095 == 0 && cancel.is_cancelled() {
                    return Err(BraidError::Cancelled);
                }
            }
            Ok(())
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct NodeHandle(usize);

impl NodeHandle {
    fn index(self) -> usize {
        self.0
    }
}

struct CompileNodes<'a> {
    nodes: Vec<&'a NodeSpec>,
    handles_by_id: HashMap<&'a str, NodeHandle>,
    outputs: Vec<OutputKind>,
}

impl<'a> CompileNodes<'a> {
    fn build(state: &'a FastNoiseState) -> BraidResult<Self> {
        let mut nodes = Vec::with_capacity(state.nodes.len());
        let mut handles_by_id = HashMap::with_capacity(state.nodes.len());
        for (_, node) in state.nodes.iter() {
            let handle = NodeHandle(nodes.len());
            if handles_by_id.insert(node.id(), handle).is_some() {
                return Err(BraidError::DuplicateId {
                    kind: "node",
                    id: node.id().to_owned(),
                });
            }
            nodes.push(node);
        }
        let outputs = nodes
            .iter()
            .map(|node| node.output_kind(state.dimension))
            .collect();
        Ok(Self {
            nodes,
            handles_by_id,
            outputs,
        })
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn handles(&self) -> impl Iterator<Item = NodeHandle> + '_ {
        (0..self.nodes.len()).map(NodeHandle)
    }

    fn node(&self, handle: NodeHandle) -> &'a NodeSpec {
        self.nodes[handle.index()]
    }

    fn output(&self, handle: NodeHandle) -> OutputKind {
        self.outputs[handle.index()]
    }

    fn resolve(&self, id: &str) -> Option<NodeHandle> {
        self.handles_by_id.get(id).copied()
    }
}

fn compile_graph(
    state: &FastNoiseState,
    scratch: &mut PlannerScratch,
) -> BraidResult<CompiledPlan<FastNoisePlannerMeta>> {
    let graph = CompileNodes::build(state)?;
    let mut indegree = vec![0usize; graph.len()];
    let mut adjacency = vec![Vec::new(); graph.len()];
    for handle in graph.handles() {
        let deps = validate_and_collect_dependencies(&graph, handle, state.dimension)?;
        for dep in deps {
            adjacency[dep.index()].push(handle);
            indegree[handle.index()] += 1;
        }
    }

    let final_handle =
        graph
            .resolve(state.final_field.as_str())
            .ok_or_else(|| BraidError::MissingReference {
                kind: "graph",
                id: "final_field".to_owned(),
                reference: state.final_field.clone(),
            })?;
    if !matches!(graph.output(final_handle), OutputKind::Scalar(dim) if dim == state.dimension) {
        return Err(BraidError::InvalidSpec(
            "final_field must reference a scalar node in graph dimension".to_owned(),
        ));
    }

    let mut queue = VecDeque::new();
    for handle in graph.handles() {
        if indegree[handle.index()] == 0 {
            queue.push_back(handle);
        }
    }
    let mut sorted = Vec::with_capacity(graph.len());
    while let Some(handle) = queue.pop_front() {
        sorted.push(handle);
        for child in adjacency[handle.index()].iter().copied() {
            let entry = &mut indegree[child.index()];
            *entry = entry.saturating_sub(1);
            if *entry == 0 {
                queue.push_back(child);
            }
        }
    }
    if sorted.len() != graph.len() {
        return Err(BraidError::InvalidSpec(
            "fastnoise graph contains a cycle".to_owned(),
        ));
    }

    let mut next_slot = SLOT_DYNAMIC_START;
    let mut position_slots = vec![None; graph.len()];
    let mut scalar_slots = vec![None; graph.len()];
    let mut buffers = vec![
        BufferSpec::dynamic(SLOT_QUERY_META, ElementKind::U32),
        BufferSpec::dynamic(SLOT_QUERY_F32, ElementKind::F32),
        BufferSpec::dynamic(SLOT_QUERY_OFFSETS, ElementKind::U32),
        BufferSpec::dynamic(SLOT_BASE_X, ElementKind::F32),
        BufferSpec::dynamic(SLOT_BASE_Y, ElementKind::F32),
    ];
    if state.dimension == GraphDimension::D3 {
        buffers.push(BufferSpec::dynamic(SLOT_BASE_Z, ElementKind::F32));
    }

    for handle in sorted.iter().copied() {
        match graph.output(handle) {
            OutputKind::Position(GraphDimension::D2) => {
                let slots = allocate_position_slots(&mut next_slot, GraphDimension::D2);
                position_slots[handle.index()] = Some(slots);
                buffers.push(dynamic_f32_buffer(slots.x));
                buffers.push(dynamic_f32_buffer(slots.y));
            }
            OutputKind::Position(GraphDimension::D3) => {
                let slots = allocate_position_slots(&mut next_slot, GraphDimension::D3);
                position_slots[handle.index()] = Some(slots);
                buffers.push(dynamic_f32_buffer(slots.x));
                buffers.push(dynamic_f32_buffer(slots.y));
                let slot_z = expect_slot(slots.z, "position z alloc")?;
                buffers.push(dynamic_f32_buffer(slot_z));
            }
            OutputKind::Scalar(_) => {
                let slot = allocate_slot(&mut next_slot);
                scalar_slots[handle.index()] = Some(slot);
                buffers.push(dynamic_f32_buffer(slot));
            }
        }
    }

    let mut stages = Vec::with_capacity(sorted.len() + 1);
    stages.push(StageSpec::single(
        KernelSpec::empty(match state.dimension {
            GraphDimension::D2 => FastNoiseKernel::InitGrid2d.into(),
            GraphDimension::D3 => FastNoiseKernel::InitGrid3d.into(),
        })
        .with_bindings([
            BufferBinding::read(SLOT_QUERY_META),
            BufferBinding::read(SLOT_QUERY_F32),
            BufferBinding::read(SLOT_QUERY_OFFSETS),
        ]),
    ));

    for handle in sorted.iter().copied() {
        let node = graph.node(handle);
        let kernel = match node {
            NodeSpec::Warp2D(node) => {
                let source = resolve_position_source(
                    &node.source,
                    GraphDimension::D2,
                    &graph,
                    &position_slots,
                )?;
                let output =
                    expect_position_slots(position_slots.as_slice(), handle, "warp2d output slot")?;
                WarpPayload {
                    dimension: GraphDimension::D2,
                    source,
                    output,
                    noise: node.noise.clone(),
                }
                .encode(scratch)
            }
            NodeSpec::Warp3D(node) => {
                let source = resolve_position_source(
                    &node.source,
                    GraphDimension::D3,
                    &graph,
                    &position_slots,
                )?;
                let output =
                    expect_position_slots(position_slots.as_slice(), handle, "warp3d output slot")?;
                WarpPayload {
                    dimension: GraphDimension::D3,
                    source,
                    output,
                    noise: node.noise.clone(),
                }
                .encode(scratch)
            }
            NodeSpec::Sample2D(node) => {
                let source = resolve_position_source(
                    &node.source,
                    GraphDimension::D2,
                    &graph,
                    &position_slots,
                )?;
                let output =
                    expect_scalar_slot(scalar_slots.as_slice(), handle, "sample2d output slot")?;
                SamplePayload {
                    dimension: GraphDimension::D2,
                    source,
                    output,
                    noise: node.noise.clone(),
                }
                .encode(scratch)
            }
            NodeSpec::Sample3D(node) => {
                let source = resolve_position_source(
                    &node.source,
                    GraphDimension::D3,
                    &graph,
                    &position_slots,
                )?;
                let output =
                    expect_scalar_slot(scalar_slots.as_slice(), handle, "sample3d output slot")?;
                SamplePayload {
                    dimension: GraphDimension::D3,
                    source,
                    output,
                    noise: node.noise.clone(),
                }
                .encode(scratch)
            }
            NodeSpec::Combine(node) => {
                let output =
                    expect_scalar_slot(scalar_slots.as_slice(), handle, "combine output slot")?;
                let mut inputs = Vec::with_capacity(node.inputs.len());
                for input in &node.inputs {
                    let input_handle = graph.resolve(input.as_str()).ok_or_else(|| {
                        BraidError::MissingReference {
                            kind: "combine",
                            id: node.id.clone(),
                            reference: input.clone(),
                        }
                    })?;
                    let slot = expect_scalar_slot(
                        scalar_slots.as_slice(),
                        input_handle,
                        "combine input slot",
                    )?;
                    inputs.push(slot);
                }
                CombinePayload {
                    op: node.op,
                    inputs,
                    output,
                    params: node.params.clone(),
                }
                .encode(scratch)
            }
        }?;
        stages.push(StageSpec::single(kernel));
    }

    let final_slot = expect_scalar_slot(scalar_slots.as_slice(), final_handle, "final field slot")?;

    let mut plan = CompiledPlan::builder(FastNoisePlannerMeta {
        dimension: state.dimension,
        final_slot,
    });
    for buffer in buffers {
        plan.buffer(buffer);
    }
    for stage in stages {
        plan.stage(stage);
    }
    Ok(plan.build())
}

fn validate_node_id(id: &str) -> BraidResult<()> {
    if id.trim().is_empty() {
        return Err(BraidError::InvalidSpec(
            "node id cannot be empty".to_owned(),
        ));
    }
    Ok(())
}

fn validate_and_collect_dependencies(
    graph: &CompileNodes<'_>,
    handle: NodeHandle,
    dimension: GraphDimension,
) -> BraidResult<Vec<NodeHandle>> {
    let node = graph.node(handle);
    match node {
        NodeSpec::Warp2D(node) => {
            ensure_dimension(dimension, GraphDimension::D2, node.id.as_str(), "warp2d")?;
            Ok(
                validate_position_source(
                    &node.source,
                    GraphDimension::D2,
                    node.id.as_str(),
                    graph,
                )?
                .into_iter()
                .collect(),
            )
        }
        NodeSpec::Warp3D(node) => {
            ensure_dimension(dimension, GraphDimension::D3, node.id.as_str(), "warp3d")?;
            Ok(
                validate_position_source(
                    &node.source,
                    GraphDimension::D3,
                    node.id.as_str(),
                    graph,
                )?
                .into_iter()
                .collect(),
            )
        }
        NodeSpec::Sample2D(node) => {
            ensure_dimension(dimension, GraphDimension::D2, node.id.as_str(), "sample2d")?;
            Ok(
                validate_position_source(
                    &node.source,
                    GraphDimension::D2,
                    node.id.as_str(),
                    graph,
                )?
                .into_iter()
                .collect(),
            )
        }
        NodeSpec::Sample3D(node) => {
            ensure_dimension(dimension, GraphDimension::D3, node.id.as_str(), "sample3d")?;
            Ok(
                validate_position_source(
                    &node.source,
                    GraphDimension::D3,
                    node.id.as_str(),
                    graph,
                )?
                .into_iter()
                .collect(),
            )
        }
        NodeSpec::Combine(node) => {
            validate_combine_shape(node, dimension)?;
            let mut deps = Vec::with_capacity(node.inputs.len());
            for input in &node.inputs {
                let Some(dep) = graph.resolve(input.as_str()) else {
                    return Err(BraidError::MissingReference {
                        kind: "combine",
                        id: node.id.clone(),
                        reference: input.clone(),
                    });
                };
                if graph.output(dep) != OutputKind::Scalar(dimension) {
                    return Err(BraidError::InvalidSpec(format!(
                        "combine '{}' input '{}' must be a scalar {} node",
                        node.id,
                        input,
                        dimension_label(dimension)
                    )));
                }
                if deps.contains(&dep) {
                    return Err(BraidError::InvalidSpec(format!(
                        "combine '{}' cannot repeat input '{}'",
                        node.id, input
                    )));
                }
                deps.push(dep);
            }
            Ok(deps)
        }
    }
}

fn validate_position_source(
    source: &PositionSource,
    expected: GraphDimension,
    node_id: &str,
    graph: &CompileNodes<'_>,
) -> BraidResult<Option<NodeHandle>> {
    match source {
        PositionSource::Base => Ok(None),
        PositionSource::Node(id) => {
            let Some(handle) = graph.resolve(id.as_str()) else {
                return Err(BraidError::MissingReference {
                    kind: "node",
                    id: node_id.to_owned(),
                    reference: id.clone(),
                });
            };
            if graph.output(handle) != OutputKind::Position(expected) {
                return Err(BraidError::InvalidSpec(format!(
                    "node '{}' source '{}' must be a position {} node",
                    node_id,
                    id,
                    dimension_label(expected)
                )));
            }
            Ok(Some(handle))
        }
    }
}

fn ensure_dimension(
    actual: GraphDimension,
    expected: GraphDimension,
    node_id: &str,
    node_kind: &str,
) -> BraidResult<()> {
    if actual != expected {
        return Err(BraidError::InvalidSpec(format!(
            "{} '{}' does not match {} graph",
            node_kind,
            node_id,
            dimension_label(actual)
        )));
    }
    Ok(())
}

fn validate_combine_shape(node: &CombineNode, dimension: GraphDimension) -> BraidResult<()> {
    match node.op {
        CombineOp::Add | CombineOp::Sub | CombineOp::Mul | CombineOp::Min | CombineOp::Max => {
            if node.inputs.len() != 2 || !node.params.is_empty() {
                return Err(BraidError::InvalidSpec(format!(
                    "combine '{}' {:?} requires exactly 2 inputs and 0 params",
                    node.id, node.op
                )));
            }
        }
        CombineOp::Clamp => {
            if node.inputs.len() != 1 || node.params.len() != 2 {
                return Err(BraidError::InvalidSpec(format!(
                    "combine '{}' clamp requires 1 input and 2 params",
                    node.id
                )));
            }
        }
        CombineOp::Remap => {
            if node.inputs.len() != 1 || node.params.len() != 4 {
                return Err(BraidError::InvalidSpec(format!(
                    "combine '{}' remap requires 1 input and 4 params",
                    node.id
                )));
            }
        }
        CombineOp::YGradient => {
            if dimension != GraphDimension::D3 {
                return Err(BraidError::InvalidSpec(format!(
                    "combine '{}' ygradient requires 3d graph",
                    node.id
                )));
            }
            if node.inputs.len() != 1 || node.params.len() != 4 {
                return Err(BraidError::InvalidSpec(format!(
                    "combine '{}' ygradient requires 1 input and 4 params",
                    node.id
                )));
            }
        }
    }
    Ok(())
}

fn resolve_position_source(
    source: &PositionSource,
    expected: GraphDimension,
    graph: &CompileNodes<'_>,
    position_slots: &[Option<PositionSlots>],
) -> BraidResult<PositionSlots> {
    match source {
        PositionSource::Base => Ok(match expected {
            GraphDimension::D2 => PositionSlots {
                x: SLOT_BASE_X,
                y: SLOT_BASE_Y,
                z: None,
            },
            GraphDimension::D3 => PositionSlots {
                x: SLOT_BASE_X,
                y: SLOT_BASE_Y,
                z: Some(SLOT_BASE_Z),
            },
        }),
        PositionSource::Node(id) => graph
            .resolve(id.as_str())
            .ok_or_else(|| BraidError::MissingReference {
                kind: "node",
                id: "position_source".to_owned(),
                reference: id.clone(),
            })
            .and_then(|handle| {
                expect_position_slots(position_slots, handle, "position source slots")
            }),
    }
}

fn allocate_slot(next_slot: &mut u16) -> BufferSlot {
    let slot = BufferSlot::new(*next_slot);
    *next_slot += 1;
    slot
}

fn allocate_position_slots(next_slot: &mut u16, dimension: GraphDimension) -> PositionSlots {
    PositionSlots {
        x: allocate_slot(next_slot),
        y: allocate_slot(next_slot),
        z: match dimension {
            GraphDimension::D2 => None,
            GraphDimension::D3 => Some(allocate_slot(next_slot)),
        },
    }
}

fn dynamic_f32_buffer(slot: BufferSlot) -> BufferSpec {
    BufferSpec::dynamic(slot, ElementKind::F32)
}

fn summarize_samples(values: &[f32]) -> ChunkSummary {
    if values.is_empty() {
        return ChunkSummary {
            samples: 0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            checksum: 0,
            taps: [0.0; 8],
        };
    }

    let mut min = values[0];
    let mut max = values[0];
    let mut sum = 0.0f64;
    let mut checksum = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        min = min.min(*value);
        max = max.max(*value);
        sum += f64::from(*value);
        checksum ^= u64::from(value.to_bits()) + 0x9e37_79b9;
        checksum = checksum.wrapping_mul(0x1000_0000_01b3);
    }

    let mut taps = [0.0; 8];
    let tap_count = taps.len();
    for (index, tap) in taps.iter_mut().enumerate() {
        let sample_index = if values.len() == 1 {
            0
        } else {
            index * (values.len() - 1) / (tap_count - 1)
        };
        *tap = values[sample_index];
    }

    ChunkSummary {
        samples: values.len(),
        min,
        max,
        mean: (sum / values.len() as f64) as f32,
        checksum,
        taps,
    }
}

fn sample_count_2d(width: usize, height: usize) -> BraidResult<usize> {
    width
        .checked_mul(height)
        .ok_or_else(|| BraidError::InvalidSpec("grid2d sample count overflow".to_owned()))
}

fn sample_count_3d(width: usize, height: usize, depth: usize) -> BraidResult<usize> {
    width
        .checked_mul(height)
        .and_then(|value| value.checked_mul(depth))
        .ok_or_else(|| BraidError::InvalidSpec("grid3d sample count overflow".to_owned()))
}

fn total_samples_from_offsets(offsets: &[u32]) -> BraidResult<usize> {
    let Some(last) = offsets.last().copied() else {
        return Ok(0);
    };
    usize::try_from(last).map_err(|_| BraidError::InvalidSpec("offset total overflow".to_owned()))
}

fn usize_to_u32(value: usize, label: &str) -> BraidResult<u32> {
    u32::try_from(value)
        .map_err(|_| BraidError::InvalidSpec(format!("{} does not fit into u32", label)))
}

fn expect_slot(slot: Option<BufferSlot>, label: &str) -> BraidResult<BufferSlot> {
    slot.ok_or_else(|| BraidError::InvalidSpec(format!("missing {}", label)))
}

fn expect_position_slots(
    slots: &[Option<PositionSlots>],
    handle: NodeHandle,
    label: &str,
) -> BraidResult<PositionSlots> {
    slots[handle.index()].ok_or_else(|| BraidError::InvalidSpec(format!("missing {}", label)))
}

fn expect_scalar_slot(
    slots: &[Option<BufferSlot>],
    handle: NodeHandle,
    label: &str,
) -> BraidResult<BufferSlot> {
    slots[handle.index()].ok_or_else(|| BraidError::InvalidSpec(format!("missing {}", label)))
}

fn expect_two_inputs(inputs: &[BufferSlot], label: &str) -> BraidResult<[BufferSlot; 2]> {
    inputs
        .try_into()
        .map_err(|_| BraidError::InvalidSpec(format!("{} expects two inputs", label)))
}

fn expect_input(inputs: &[BufferSlot], count: usize, label: &str) -> BraidResult<BufferSlot> {
    if inputs.len() != count {
        return Err(BraidError::InvalidSpec(format!(
            "{} expects {} inputs",
            label, count
        )));
    }
    Ok(inputs[0])
}

fn expect_params<const N: usize>(
    params: &[f32],
    count: usize,
    label: &str,
) -> BraidResult<[f32; N]> {
    if params.len() != count {
        return Err(BraidError::InvalidSpec(format!(
            "{} expects {} params",
            label, count
        )));
    }
    params
        .try_into()
        .map_err(|_| BraidError::InvalidSpec(format!("{} param shape mismatch", label)))
}

fn dimension_label(dimension: GraphDimension) -> &'static str {
    match dimension {
        GraphDimension::D2 => "2d",
        GraphDimension::D3 => "3d",
    }
}

struct PayloadWriter<'a> {
    bytes: &'a mut Vec<u8>,
}

impl<'a> PayloadWriter<'a> {
    fn new(bytes: &'a mut Vec<u8>) -> Self {
        Self { bytes }
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn slot(&mut self, value: BufferSlot) {
        self.u16(value.raw());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn i32(&mut self, value: i32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn f32(&mut self, value: f32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn noise(&mut self, noise: &FastNoiseLite) {
        self.i32(noise.seed);
        self.f32(noise.frequency);
        self.u32(encode_noise_type(noise.noise_type));
        self.u32(encode_rotation_type(noise.rotation_type_3d));
        self.u32(encode_fractal_type(noise.fractal_type));
        self.i32(noise.octaves);
        self.f32(noise.lacunarity);
        self.f32(noise.gain);
        self.f32(noise.weighted_strength);
        self.f32(noise.ping_pong_strength);
        self.u32(encode_cellular_distance_function(
            noise.cellular_distance_function,
        ));
        self.u32(encode_cellular_return_type(noise.cellular_return_type));
        self.f32(noise.cellular_jitter_modifier);
        self.u32(encode_domain_warp_type(noise.domain_warp_type));
        self.f32(noise.domain_warp_amp);
    }
}

fn encode_noise_type(value: NoiseType) -> u32 {
    match value {
        NoiseType::OpenSimplex2 => 0,
        NoiseType::OpenSimplex2S => 1,
        NoiseType::Cellular => 2,
        NoiseType::Perlin => 3,
        NoiseType::ValueCubic => 4,
        NoiseType::Value => 5,
    }
}

fn decode_noise_type(value: u32) -> BraidResult<NoiseType> {
    match value {
        0 => Ok(NoiseType::OpenSimplex2),
        1 => Ok(NoiseType::OpenSimplex2S),
        2 => Ok(NoiseType::Cellular),
        3 => Ok(NoiseType::Perlin),
        4 => Ok(NoiseType::ValueCubic),
        5 => Ok(NoiseType::Value),
        _ => Err(BraidError::InvalidSpec(format!(
            "unknown noise type tag {}",
            value
        ))),
    }
}

fn encode_rotation_type(value: RotationType3D) -> u32 {
    match value {
        RotationType3D::None => 0,
        RotationType3D::ImproveXYPlanes => 1,
        RotationType3D::ImproveXZPlanes => 2,
    }
}

fn decode_rotation_type(value: u32) -> BraidResult<RotationType3D> {
    match value {
        0 => Ok(RotationType3D::None),
        1 => Ok(RotationType3D::ImproveXYPlanes),
        2 => Ok(RotationType3D::ImproveXZPlanes),
        _ => Err(BraidError::InvalidSpec(format!(
            "unknown rotation type tag {}",
            value
        ))),
    }
}

fn encode_fractal_type(value: FractalType) -> u32 {
    match value {
        FractalType::None => 0,
        FractalType::FBm => 1,
        FractalType::Ridged => 2,
        FractalType::PingPong => 3,
        FractalType::DomainWarpProgressive => 4,
        FractalType::DomainWarpIndependent => 5,
    }
}

fn decode_fractal_type(value: u32) -> BraidResult<FractalType> {
    match value {
        0 => Ok(FractalType::None),
        1 => Ok(FractalType::FBm),
        2 => Ok(FractalType::Ridged),
        3 => Ok(FractalType::PingPong),
        4 => Ok(FractalType::DomainWarpProgressive),
        5 => Ok(FractalType::DomainWarpIndependent),
        _ => Err(BraidError::InvalidSpec(format!(
            "unknown fractal type tag {}",
            value
        ))),
    }
}

fn encode_cellular_distance_function(value: CellularDistanceFunction) -> u32 {
    match value {
        CellularDistanceFunction::Euclidean => 0,
        CellularDistanceFunction::EuclideanSq => 1,
        CellularDistanceFunction::Manhattan => 2,
        CellularDistanceFunction::Hybrid => 3,
    }
}

fn decode_cellular_distance_function(value: u32) -> BraidResult<CellularDistanceFunction> {
    match value {
        0 => Ok(CellularDistanceFunction::Euclidean),
        1 => Ok(CellularDistanceFunction::EuclideanSq),
        2 => Ok(CellularDistanceFunction::Manhattan),
        3 => Ok(CellularDistanceFunction::Hybrid),
        _ => Err(BraidError::InvalidSpec(format!(
            "unknown cellular distance tag {}",
            value
        ))),
    }
}

fn encode_cellular_return_type(value: CellularReturnType) -> u32 {
    match value {
        CellularReturnType::CellValue => 0,
        CellularReturnType::Distance => 1,
        CellularReturnType::Distance2 => 2,
        CellularReturnType::Distance2Add => 3,
        CellularReturnType::Distance2Sub => 4,
        CellularReturnType::Distance2Mul => 5,
        CellularReturnType::Distance2Div => 6,
    }
}

fn decode_cellular_return_type(value: u32) -> BraidResult<CellularReturnType> {
    match value {
        0 => Ok(CellularReturnType::CellValue),
        1 => Ok(CellularReturnType::Distance),
        2 => Ok(CellularReturnType::Distance2),
        3 => Ok(CellularReturnType::Distance2Add),
        4 => Ok(CellularReturnType::Distance2Sub),
        5 => Ok(CellularReturnType::Distance2Mul),
        6 => Ok(CellularReturnType::Distance2Div),
        _ => Err(BraidError::InvalidSpec(format!(
            "unknown cellular return tag {}",
            value
        ))),
    }
}

fn encode_domain_warp_type(value: DomainWarpType) -> u32 {
    match value {
        DomainWarpType::OpenSimplex2 => 0,
        DomainWarpType::OpenSimplex2Reduced => 1,
        DomainWarpType::BasicGrid => 2,
    }
}

fn decode_domain_warp_type(value: u32) -> BraidResult<DomainWarpType> {
    match value {
        0 => Ok(DomainWarpType::OpenSimplex2),
        1 => Ok(DomainWarpType::OpenSimplex2Reduced),
        2 => Ok(DomainWarpType::BasicGrid),
        _ => Err(BraidError::InvalidSpec(format!(
            "unknown domain warp tag {}",
            value
        ))),
    }
}

fn encode_combine_op(value: CombineOp) -> u32 {
    match value {
        CombineOp::Add => 0,
        CombineOp::Sub => 1,
        CombineOp::Mul => 2,
        CombineOp::Min => 3,
        CombineOp::Max => 4,
        CombineOp::Clamp => 5,
        CombineOp::Remap => 6,
        CombineOp::YGradient => 7,
    }
}

fn decode_combine_op(value: u32) -> BraidResult<CombineOp> {
    match value {
        0 => Ok(CombineOp::Add),
        1 => Ok(CombineOp::Sub),
        2 => Ok(CombineOp::Mul),
        3 => Ok(CombineOp::Min),
        4 => Ok(CombineOp::Max),
        5 => Ok(CombineOp::Clamp),
        6 => Ok(CombineOp::Remap),
        7 => Ok(CombineOp::YGradient),
        _ => Err(BraidError::InvalidSpec(format!(
            "unknown combine op tag {}",
            value
        ))),
    }
}

struct PayloadReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> PayloadReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn finish(&self) -> BraidResult<()> {
        if self.offset == self.bytes.len() {
            return Ok(());
        }
        Err(BraidError::InvalidSpec(
            "kernel payload has trailing bytes".to_owned(),
        ))
    }

    fn read_exact<const N: usize>(&mut self) -> BraidResult<[u8; N]> {
        let end = self.offset + N;
        if end > self.bytes.len() {
            return Err(BraidError::InvalidSpec(
                "kernel payload truncated".to_owned(),
            ));
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.offset..end]);
        self.offset = end;
        Ok(out)
    }

    fn u16(&mut self) -> BraidResult<u16> {
        Ok(u16::from_le_bytes(self.read_exact()?))
    }

    fn slot(&mut self) -> BraidResult<BufferSlot> {
        Ok(BufferSlot::new(self.u16()?))
    }

    fn u32(&mut self) -> BraidResult<u32> {
        Ok(u32::from_le_bytes(self.read_exact()?))
    }

    fn i32(&mut self) -> BraidResult<i32> {
        Ok(i32::from_le_bytes(self.read_exact()?))
    }

    fn f32(&mut self) -> BraidResult<f32> {
        Ok(f32::from_le_bytes(self.read_exact()?))
    }

    fn noise(&mut self) -> BraidResult<FastNoiseLite> {
        let seed = self.i32()?;
        let frequency = self.f32()?;
        let noise_type = decode_noise_type(self.u32()?)?;
        let rotation_type = decode_rotation_type(self.u32()?)?;
        let fractal_type = decode_fractal_type(self.u32()?)?;
        let octaves = self.i32()?;
        let lacunarity = self.f32()?;
        let gain = self.f32()?;
        let weighted_strength = self.f32()?;
        let ping_pong_strength = self.f32()?;
        let cellular_distance = decode_cellular_distance_function(self.u32()?)?;
        let cellular_return = decode_cellular_return_type(self.u32()?)?;
        let cellular_jitter = self.f32()?;
        let domain_warp_type = decode_domain_warp_type(self.u32()?)?;
        let domain_warp_amp = self.f32()?;

        let mut noise = FastNoiseLite::with_seed(seed);
        noise.set_frequency(Some(frequency));
        noise.set_noise_type(Some(noise_type));
        noise.set_rotation_type_3d(Some(rotation_type));
        noise.set_fractal_type(Some(fractal_type));
        noise.set_fractal_octaves(Some(octaves));
        noise.set_fractal_lacunarity(Some(lacunarity));
        noise.set_fractal_gain(Some(gain));
        noise.set_fractal_weighted_strength(Some(weighted_strength));
        noise.set_fractal_ping_pong_strength(Some(ping_pong_strength));
        noise.set_cellular_distance_function(Some(cellular_distance));
        noise.set_cellular_return_type(Some(cellular_return));
        noise.set_cellular_jitter(Some(cellular_jitter));
        noise.set_domain_warp_type(Some(domain_warp_type));
        noise.set_domain_warp_amp(Some(domain_warp_amp));
        Ok(noise)
    }
}

pub mod scenarios {
    use super::{
        ChunkSummary, CombineNode, CombineOp, DomainWarpType, FastNoiseChange, FastNoiseGraphSpec,
        FastNoiseLite, FractalType, GraphDimension, NodeSpec, NoiseType, PositionSource,
        Sample2DNode, Sample3DNode, Warp2DNode, Warp3DNode,
    };

    pub const BIOME_WARP_NODE: &str = "biome_warp";
    pub const BIOME_MOISTURE_NODE: &str = "biome_moisture";
    pub const BIOME_TEMPERATURE_NODE: &str = "biome_temperature";
    pub const BIOME_FINAL_NODE: &str = "biome_final";

    pub const TERRAIN_WARP_NODE: &str = "terrain_warp";
    pub const TERRAIN_CONTINENT_NODE: &str = "terrain_continent";
    pub const TERRAIN_EROSION_NODE: &str = "terrain_erosion";
    pub const TERRAIN_PEAKS_NODE: &str = "terrain_peaks";
    pub const TERRAIN_DETAIL_NODE: &str = "terrain_detail";
    pub const TERRAIN_FINAL_NODE: &str = "terrain_final";

    pub const VOXEL_WARP_NODE: &str = "voxel_warp";
    pub const VOXEL_BASE_NODE: &str = "voxel_base";
    pub const VOXEL_CAVE_NODE: &str = "voxel_cave";
    pub const VOXEL_SHAPE_NODE: &str = "voxel_shape";
    pub const VOXEL_FINAL_NODE: &str = "voxel_final";

    pub fn biome_control_2d() -> FastNoiseGraphSpec {
        FastNoiseGraphSpec {
            dimension: GraphDimension::D2,
            final_field: BIOME_FINAL_NODE.to_owned(),
            nodes: vec![
                NodeSpec::Warp2D(Warp2DNode {
                    id: BIOME_WARP_NODE.to_owned(),
                    source: PositionSource::Base,
                    noise: warp_noise(701, 0.0025, 18.0),
                }),
                NodeSpec::Sample2D(Sample2DNode {
                    id: BIOME_MOISTURE_NODE.to_owned(),
                    source: PositionSource::Node(BIOME_WARP_NODE.to_owned()),
                    noise: sample_noise(711, NoiseType::Perlin, FractalType::FBm, 0.0022, 4),
                }),
                NodeSpec::Sample2D(Sample2DNode {
                    id: BIOME_TEMPERATURE_NODE.to_owned(),
                    source: PositionSource::Node(BIOME_WARP_NODE.to_owned()),
                    noise: sample_noise(719, NoiseType::OpenSimplex2, FractalType::FBm, 0.0018, 4),
                }),
                NodeSpec::Combine(CombineNode {
                    id: BIOME_FINAL_NODE.to_owned(),
                    inputs: vec![
                        BIOME_MOISTURE_NODE.to_owned(),
                        BIOME_TEMPERATURE_NODE.to_owned(),
                    ],
                    op: CombineOp::Add,
                    params: Vec::new(),
                }),
            ],
        }
    }

    pub fn terrain_height_2d() -> FastNoiseGraphSpec {
        FastNoiseGraphSpec {
            dimension: GraphDimension::D2,
            final_field: TERRAIN_FINAL_NODE.to_owned(),
            nodes: vec![
                NodeSpec::Warp2D(Warp2DNode {
                    id: TERRAIN_WARP_NODE.to_owned(),
                    source: PositionSource::Base,
                    noise: warp_noise(1001, 0.0015, 24.0),
                }),
                NodeSpec::Sample2D(Sample2DNode {
                    id: TERRAIN_CONTINENT_NODE.to_owned(),
                    source: PositionSource::Node(TERRAIN_WARP_NODE.to_owned()),
                    noise: sample_noise(1011, NoiseType::OpenSimplex2, FractalType::FBm, 0.0012, 5),
                }),
                NodeSpec::Sample2D(Sample2DNode {
                    id: TERRAIN_EROSION_NODE.to_owned(),
                    source: PositionSource::Node(TERRAIN_WARP_NODE.to_owned()),
                    noise: sample_noise(1019, NoiseType::Perlin, FractalType::FBm, 0.0032, 4),
                }),
                NodeSpec::Sample2D(Sample2DNode {
                    id: TERRAIN_PEAKS_NODE.to_owned(),
                    source: PositionSource::Node(TERRAIN_WARP_NODE.to_owned()),
                    noise: sample_noise(
                        1027,
                        NoiseType::OpenSimplex2S,
                        FractalType::Ridged,
                        0.0056,
                        4,
                    ),
                }),
                NodeSpec::Sample2D(Sample2DNode {
                    id: TERRAIN_DETAIL_NODE.to_owned(),
                    source: PositionSource::Node(TERRAIN_WARP_NODE.to_owned()),
                    noise: terrain_detail_noise(1033),
                }),
                NodeSpec::Combine(CombineNode {
                    id: "terrain_base".to_owned(),
                    inputs: vec![
                        TERRAIN_CONTINENT_NODE.to_owned(),
                        TERRAIN_PEAKS_NODE.to_owned(),
                    ],
                    op: CombineOp::Add,
                    params: Vec::new(),
                }),
                NodeSpec::Combine(CombineNode {
                    id: "terrain_shaped".to_owned(),
                    inputs: vec!["terrain_base".to_owned(), TERRAIN_EROSION_NODE.to_owned()],
                    op: CombineOp::Sub,
                    params: Vec::new(),
                }),
                NodeSpec::Combine(CombineNode {
                    id: TERRAIN_FINAL_NODE.to_owned(),
                    inputs: vec!["terrain_shaped".to_owned(), TERRAIN_DETAIL_NODE.to_owned()],
                    op: CombineOp::Add,
                    params: Vec::new(),
                }),
            ],
        }
    }

    pub fn voxel_density_3d() -> FastNoiseGraphSpec {
        FastNoiseGraphSpec {
            dimension: GraphDimension::D3,
            final_field: VOXEL_FINAL_NODE.to_owned(),
            nodes: vec![
                NodeSpec::Warp3D(Warp3DNode {
                    id: VOXEL_WARP_NODE.to_owned(),
                    source: PositionSource::Base,
                    noise: warp_noise(2001, 0.011, 6.0),
                }),
                NodeSpec::Sample3D(Sample3DNode {
                    id: VOXEL_BASE_NODE.to_owned(),
                    source: PositionSource::Base,
                    noise: sample_noise(2011, NoiseType::OpenSimplex2, FractalType::FBm, 0.018, 5),
                }),
                NodeSpec::Sample3D(Sample3DNode {
                    id: VOXEL_CAVE_NODE.to_owned(),
                    source: PositionSource::Node(VOXEL_WARP_NODE.to_owned()),
                    noise: sample_noise(2019, NoiseType::Cellular, FractalType::Ridged, 0.03, 3),
                }),
                NodeSpec::Combine(CombineNode {
                    id: VOXEL_SHAPE_NODE.to_owned(),
                    inputs: vec![VOXEL_BASE_NODE.to_owned(), VOXEL_CAVE_NODE.to_owned()],
                    op: CombineOp::Sub,
                    params: Vec::new(),
                }),
                NodeSpec::Combine(CombineNode {
                    id: VOXEL_FINAL_NODE.to_owned(),
                    inputs: vec![VOXEL_SHAPE_NODE.to_owned()],
                    op: CombineOp::YGradient,
                    params: vec![-32.0, 32.0, 1.0, -1.0],
                }),
            ],
        }
    }

    pub fn terrain_patch_from_biome(summary: &ChunkSummary) -> Vec<FastNoiseChange> {
        let mut noise = terrain_detail_noise(1033);
        noise.set_frequency(Some(0.015 + summary.taps[0].abs() * 0.02));
        noise.set_fractal_gain(Some((0.35 + summary.taps[3].abs() * 0.35).min(0.95)));
        vec![FastNoiseChange::UpsertNode(NodeSpec::Sample2D(
            Sample2DNode {
                id: TERRAIN_DETAIL_NODE.to_owned(),
                source: PositionSource::Node(TERRAIN_WARP_NODE.to_owned()),
                noise,
            },
        ))]
    }

    pub fn voxel_patch_from_terrain(summary: &ChunkSummary) -> Vec<FastNoiseChange> {
        let mut noise = sample_noise(2011, NoiseType::OpenSimplex2, FractalType::FBm, 0.018, 5);
        noise.set_frequency(Some(0.014 + summary.mean.abs() * 0.01));
        noise.set_fractal_gain(Some((0.4 + summary.taps[6].abs() * 0.2).min(0.95)));
        vec![FastNoiseChange::UpsertNode(NodeSpec::Sample3D(
            Sample3DNode {
                id: VOXEL_BASE_NODE.to_owned(),
                source: PositionSource::Base,
                noise,
            },
        ))]
    }

    fn sample_noise(
        seed: i32,
        noise_type: NoiseType,
        fractal_type: FractalType,
        frequency: f32,
        octaves: i32,
    ) -> FastNoiseLite {
        let mut noise = FastNoiseLite::with_seed(seed);
        noise.set_noise_type(Some(noise_type));
        noise.set_frequency(Some(frequency));
        noise.set_fractal_type(Some(fractal_type));
        noise.set_fractal_octaves(Some(octaves));
        noise.set_fractal_lacunarity(Some(2.0));
        noise.set_fractal_gain(Some(0.5));
        noise.set_fractal_weighted_strength(Some(0.15));
        noise
    }

    fn terrain_detail_noise(seed: i32) -> FastNoiseLite {
        let mut noise = sample_noise(seed, NoiseType::ValueCubic, FractalType::FBm, 0.022, 3);
        noise.set_fractal_gain(Some(0.45));
        noise
    }

    fn warp_noise(seed: i32, frequency: f32, amplitude: f32) -> FastNoiseLite {
        let mut noise = FastNoiseLite::with_seed(seed);
        noise.set_frequency(Some(frequency));
        noise.set_domain_warp_type(Some(DomainWarpType::OpenSimplex2));
        noise.set_domain_warp_amp(Some(amplitude));
        noise.set_fractal_type(Some(FractalType::DomainWarpProgressive));
        noise.set_fractal_octaves(Some(3));
        noise.set_fractal_gain(Some(0.5));
        noise
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ChunkQuery, CombineNode, CombineOp, FastNoiseChange, FastNoiseGraphSpec, FastNoiseLite,
        FastNoisePlanner, FractalType, GraphDimension, KernelPayload, NodeSpec, NoiseType,
        PositionSlots, PositionSource, Sample2DNode, Sample3DNode, SamplePayload, Warp2DNode,
        scenarios, summarize_samples,
    };
    use braid::{BackendConfig, BraidExecutor, BufferSlot, PlannerBackend, Stack};
    use std::sync::Arc;

    fn sample_noise(seed: i32, noise_type: NoiseType, frequency: f32) -> FastNoiseLite {
        let mut noise = FastNoiseLite::with_seed(seed);
        noise.set_noise_type(Some(noise_type));
        noise.set_frequency(Some(frequency));
        noise.set_fractal_type(Some(FractalType::FBm));
        noise.set_fractal_octaves(Some(3));
        noise
    }

    fn make_stack(
        spec: FastNoiseGraphSpec,
    ) -> braid::BraidResult<Stack<FastNoisePlanner, super::FastNoiseCpuBackend>> {
        let executor = Arc::new(BraidExecutor::new(4));
        let backend = executor.register_backend(
            Arc::new(super::make_cpu_backend()),
            BackendConfig { lane_count: 4 },
        );
        Stack::create(executor, Arc::new(FastNoisePlanner), backend, spec)
    }

    fn run_one(
        stack: &Stack<FastNoisePlanner, super::FastNoiseCpuBackend>,
        query: ChunkQuery,
    ) -> braid::BraidResult<super::ChunkSummary> {
        let job = stack.dispatch(vec![query])?;
        let mut values = stack.collect(job)?;
        Ok(values.remove(0))
    }

    #[test]
    fn planner_rejects_missing_refs() {
        let planner = FastNoisePlanner;
        let spec = FastNoiseGraphSpec {
            dimension: GraphDimension::D2,
            final_field: "sample".to_owned(),
            nodes: vec![NodeSpec::Sample2D(Sample2DNode {
                id: "sample".to_owned(),
                source: PositionSource::Node("missing".to_owned()),
                noise: sample_noise(1, NoiseType::Perlin, 0.01),
            })],
        };

        let state = planner.init_state(&spec).expect("state init");
        let error = planner
            .compile(&state, &mut braid::PlannerScratch::default())
            .expect_err("missing ref should fail");
        assert!(error.to_string().contains("missing"));
    }

    #[test]
    fn planner_rejects_dimension_mismatch_and_cycles() {
        let planner = FastNoisePlanner;
        let bad_dimension = FastNoiseGraphSpec {
            dimension: GraphDimension::D2,
            final_field: "sample3d".to_owned(),
            nodes: vec![NodeSpec::Sample3D(Sample3DNode {
                id: "sample3d".to_owned(),
                source: PositionSource::Base,
                noise: sample_noise(2, NoiseType::OpenSimplex2, 0.02),
            })],
        };
        let state = planner.init_state(&bad_dimension).expect("state init");
        let error = planner
            .compile(&state, &mut braid::PlannerScratch::default())
            .expect_err("dimension mismatch should fail");
        assert!(error.to_string().contains("3d"));

        let cycle = FastNoiseGraphSpec {
            dimension: GraphDimension::D2,
            final_field: "a".to_owned(),
            nodes: vec![
                NodeSpec::Combine(CombineNode {
                    id: "a".to_owned(),
                    inputs: vec!["b".to_owned()],
                    op: CombineOp::Clamp,
                    params: vec![-1.0, 1.0],
                }),
                NodeSpec::Combine(CombineNode {
                    id: "b".to_owned(),
                    inputs: vec!["a".to_owned()],
                    op: CombineOp::Clamp,
                    params: vec![-1.0, 1.0],
                }),
            ],
        };
        let state = planner.init_state(&cycle).expect("cycle state init");
        let error = planner
            .compile(&state, &mut braid::PlannerScratch::default())
            .expect_err("cycle should fail");
        assert!(error.to_string().contains("cycle"));
    }

    #[test]
    fn update_remove_and_set_final_field_recompile() {
        let spec = FastNoiseGraphSpec {
            dimension: GraphDimension::D2,
            final_field: "left".to_owned(),
            nodes: vec![
                NodeSpec::Sample2D(Sample2DNode {
                    id: "left".to_owned(),
                    source: PositionSource::Base,
                    noise: sample_noise(11, NoiseType::Perlin, 0.01),
                }),
                NodeSpec::Sample2D(Sample2DNode {
                    id: "right".to_owned(),
                    source: PositionSource::Base,
                    noise: sample_noise(19, NoiseType::Value, 0.02),
                }),
            ],
        };
        let stack = make_stack(spec).expect("stack");
        let query = ChunkQuery::Grid2D {
            width: 8,
            height: 6,
            origin: [0.0, 0.0],
            step: [1.0, 1.0],
        };
        let left = run_one(&stack, query.clone()).expect("left");

        let mut updated = sample_noise(23, NoiseType::OpenSimplex2, 0.03);
        updated.set_fractal_gain(Some(0.7));
        stack
            .update(vec![
                FastNoiseChange::UpsertNode(NodeSpec::Sample2D(Sample2DNode {
                    id: "right".to_owned(),
                    source: PositionSource::Base,
                    noise: updated,
                })),
                FastNoiseChange::SetFinalField {
                    id: "right".to_owned(),
                },
            ])
            .expect("update");
        let right = run_one(&stack, query.clone()).expect("right");
        assert_ne!(left.checksum, right.checksum);

        stack
            .update(vec![FastNoiseChange::RemoveNode {
                id: "left".to_owned(),
            }])
            .expect("remove unused node");
        let after_remove = run_one(&stack, query).expect("after remove");
        assert_eq!(right.samples, after_remove.samples);
        assert_eq!(right.checksum, after_remove.checksum);
    }

    #[test]
    fn packet_sizing_and_offsets_follow_query_shapes() {
        let planner = FastNoisePlanner;
        let spec = FastNoiseGraphSpec {
            dimension: GraphDimension::D2,
            final_field: "sample".to_owned(),
            nodes: vec![NodeSpec::Sample2D(Sample2DNode {
                id: "sample".to_owned(),
                source: PositionSource::Base,
                noise: sample_noise(31, NoiseType::Perlin, 0.02),
            })],
        };
        let state = planner.init_state(&spec).expect("state");
        let plan = planner
            .compile(&state, &mut braid::PlannerScratch::default())
            .expect("plan");
        let mut packet = braid::JobPacket::default();
        planner
            .encode_batch(
                &plan,
                &[
                    ChunkQuery::Grid2D {
                        width: 4,
                        height: 3,
                        origin: [0.0, 0.0],
                        step: [1.0, 1.0],
                    },
                    ChunkQuery::Grid2D {
                        width: 2,
                        height: 2,
                        origin: [10.0, 5.0],
                        step: [0.5, 0.5],
                    },
                ],
                &mut packet,
                &mut braid::BatchScratch::default(),
            )
            .expect("encode");

        assert_eq!(packet.query_count(), 2);
        assert_eq!(
            packet.u32(super::SLOT_QUERY_META).expect("meta"),
            &[4, 3, 1, 2, 2, 1]
        );
        assert_eq!(
            packet.u32(super::SLOT_QUERY_OFFSETS).expect("offsets"),
            &[0, 12, 16]
        );
    }

    #[test]
    fn sample3d_payload_roundtrip_preserves_fields() {
        let mut noise = sample_noise(37, NoiseType::OpenSimplex2S, 0.125);
        noise.set_fractal_lacunarity(Some(2.5));
        noise.set_fractal_gain(Some(0.45));

        let payload = SamplePayload {
            dimension: GraphDimension::D3,
            source: PositionSlots {
                x: BufferSlot::new(41),
                y: BufferSlot::new(42),
                z: Some(BufferSlot::new(43)),
            },
            output: BufferSlot::new(99),
            noise: noise.clone(),
        };

        let mut scratch = braid::PlannerScratch::default();
        let kernel = payload.encode(&mut scratch).expect("encode");
        let decoded = SamplePayload::decode(super::FastNoiseKernel::Sample3d, &kernel.payload)
            .expect("decode");

        assert_eq!(decoded.dimension, payload.dimension);
        assert_eq!(decoded.source.x, payload.source.x);
        assert_eq!(decoded.source.y, payload.source.y);
        assert_eq!(decoded.source.z, payload.source.z);
        assert_eq!(decoded.output, payload.output);
        assert_eq!(decoded.noise.seed, noise.seed);
        assert_eq!(decoded.noise.frequency, noise.frequency);
        assert_eq!(decoded.noise.noise_type, noise.noise_type);
        assert_eq!(decoded.noise.fractal_type, noise.fractal_type);
        assert_eq!(decoded.noise.octaves, noise.octaves);
        assert_eq!(decoded.noise.lacunarity, noise.lacunarity);
        assert_eq!(decoded.noise.gain, noise.gain);
    }

    #[test]
    fn sample2d_and_warp2d_match_direct_fastnoise() {
        let sample = sample_noise(41, NoiseType::Perlin, 0.015);
        let warp = {
            let mut noise = FastNoiseLite::with_seed(43);
            noise.set_frequency(Some(0.02));
            noise.set_domain_warp_amp(Some(4.0));
            noise.set_fractal_type(Some(FractalType::DomainWarpProgressive));
            noise
        };
        let spec = FastNoiseGraphSpec {
            dimension: GraphDimension::D2,
            final_field: "sample".to_owned(),
            nodes: vec![
                NodeSpec::Warp2D(Warp2DNode {
                    id: "warp".to_owned(),
                    source: PositionSource::Base,
                    noise: warp.clone(),
                }),
                NodeSpec::Sample2D(Sample2DNode {
                    id: "sample".to_owned(),
                    source: PositionSource::Node("warp".to_owned()),
                    noise: sample.clone(),
                }),
            ],
        };
        let stack = make_stack(spec).expect("stack");
        let query = ChunkQuery::Grid2D {
            width: 7,
            height: 5,
            origin: [2.0, -3.0],
            step: [0.75, 1.25],
        };
        let summary = run_one(&stack, query.clone()).expect("summary");

        let mut direct = Vec::new();
        if let ChunkQuery::Grid2D {
            width,
            height,
            origin,
            step,
        } = query
        {
            for y in 0..height {
                for x in 0..width {
                    let px = origin[0] + x as f32 * step[0];
                    let py = origin[1] + y as f32 * step[1];
                    let (wx, wy) = warp.domain_warp_2d(px, py);
                    direct.push(sample.get_noise_2d(wx, wy));
                }
            }
        }
        assert_eq!(summary, summarize_samples(direct.as_slice()));
    }

    #[test]
    fn combine_and_ygradient_match_expected_math() {
        let sample_a = sample_noise(53, NoiseType::Value, 0.025);
        let sample_b = sample_noise(59, NoiseType::Perlin, 0.031);
        let spec = FastNoiseGraphSpec {
            dimension: GraphDimension::D3,
            final_field: "final".to_owned(),
            nodes: vec![
                NodeSpec::Sample3D(Sample3DNode {
                    id: "a".to_owned(),
                    source: PositionSource::Base,
                    noise: sample_a.clone(),
                }),
                NodeSpec::Sample3D(Sample3DNode {
                    id: "b".to_owned(),
                    source: PositionSource::Base,
                    noise: sample_b.clone(),
                }),
                NodeSpec::Combine(CombineNode {
                    id: "sum".to_owned(),
                    inputs: vec!["a".to_owned(), "b".to_owned()],
                    op: CombineOp::Add,
                    params: Vec::new(),
                }),
                NodeSpec::Combine(CombineNode {
                    id: "clamped".to_owned(),
                    inputs: vec!["sum".to_owned()],
                    op: CombineOp::Clamp,
                    params: vec![-0.8, 0.8],
                }),
                NodeSpec::Combine(CombineNode {
                    id: "final".to_owned(),
                    inputs: vec!["clamped".to_owned()],
                    op: CombineOp::YGradient,
                    params: vec![-4.0, 4.0, 0.5, -0.5],
                }),
            ],
        };
        let stack = make_stack(spec).expect("stack");
        let query = ChunkQuery::Grid3D {
            width: 4,
            height: 5,
            depth: 3,
            origin: [1.0, -4.0, 2.0],
            step: [0.5, 2.0, 0.75],
        };
        let summary = run_one(&stack, query.clone()).expect("summary");

        let mut direct = Vec::new();
        if let ChunkQuery::Grid3D {
            width,
            height,
            depth,
            origin,
            step,
        } = query
        {
            for z in 0..depth {
                for y in 0..height {
                    for x in 0..width {
                        let px = origin[0] + x as f32 * step[0];
                        let py = origin[1] + y as f32 * step[1];
                        let pz = origin[2] + z as f32 * step[2];
                        let mut value =
                            sample_a.get_noise_3d(px, py, pz) + sample_b.get_noise_3d(px, py, pz);
                        value = value.clamp(-0.8, 0.8);
                        let t = ((py + 4.0) / 8.0).clamp(0.0, 1.0);
                        value += 0.5 + ((-0.5 - 0.5) * t);
                        direct.push(value);
                    }
                }
            }
        }
        assert_eq!(summary, summarize_samples(direct.as_slice()));
    }

    #[test]
    fn terrain_and_voxel_scenarios_run_through_stack() {
        let terrain = make_stack(scenarios::terrain_height_2d()).expect("terrain");
        let voxel = make_stack(scenarios::voxel_density_3d()).expect("voxel");

        let terrain_summary = run_one(
            &terrain,
            ChunkQuery::Grid2D {
                width: 16,
                height: 16,
                origin: [64.0, 32.0],
                step: [1.0, 1.0],
            },
        )
        .expect("terrain summary");
        let voxel_summary = run_one(
            &voxel,
            ChunkQuery::Grid3D {
                width: 8,
                height: 10,
                depth: 8,
                origin: [0.0, -16.0, 0.0],
                step: [1.0, 1.0, 1.0],
            },
        )
        .expect("voxel summary");

        assert_eq!(terrain_summary.samples, 256);
        assert_eq!(voxel_summary.samples, 640);
    }

    #[test]
    fn voxel_scenario_matches_direct_fastnoise() {
        let spec = scenarios::voxel_density_3d();
        let stack = make_stack(spec.clone()).expect("stack");
        let query = ChunkQuery::Grid3D {
            width: 7,
            height: 9,
            depth: 5,
            origin: [11.0, -16.0, -7.0],
            step: [1.25, 0.5, 0.75],
        };
        let summary = run_one(&stack, query.clone()).expect("summary");

        let warp = spec
            .nodes
            .iter()
            .find_map(|node| match node {
                NodeSpec::Warp3D(node) if node.id == scenarios::VOXEL_WARP_NODE => {
                    Some(node.noise.clone())
                }
                _ => None,
            })
            .expect("warp");
        let base = spec
            .nodes
            .iter()
            .find_map(|node| match node {
                NodeSpec::Sample3D(node) if node.id == scenarios::VOXEL_BASE_NODE => {
                    Some(node.noise.clone())
                }
                _ => None,
            })
            .expect("base");
        let cave = spec
            .nodes
            .iter()
            .find_map(|node| match node {
                NodeSpec::Sample3D(node) if node.id == scenarios::VOXEL_CAVE_NODE => {
                    Some(node.noise.clone())
                }
                _ => None,
            })
            .expect("cave");

        let mut direct = Vec::new();
        if let ChunkQuery::Grid3D {
            width,
            height,
            depth,
            origin,
            step,
        } = query
        {
            for z in 0..depth {
                for y in 0..height {
                    for x in 0..width {
                        let px = origin[0] + x as f32 * step[0];
                        let py = origin[1] + y as f32 * step[1];
                        let pz = origin[2] + z as f32 * step[2];
                        let base_value = base.get_noise_3d(px, py, pz);
                        let (wx, wy, wz) = warp.domain_warp_3d(px, py, pz);
                        let cave_value = cave.get_noise_3d(wx, wy, wz);
                        let mut value = base_value;
                        value -= cave_value;
                        let t = ((py + 32.0) / 64.0).clamp(0.0, 1.0);
                        let gradient = 1.0 + ((-1.0 - 1.0) * t);
                        value += gradient;
                        direct.push(value);
                    }
                }
            }
        }

        assert_eq!(summary, summarize_samples(direct.as_slice()));
    }

    #[test]
    fn dependency_chain_updates_downstream_without_cross_stack_corruption() {
        let biome = make_stack(scenarios::biome_control_2d()).expect("biome");
        let terrain = make_stack(scenarios::terrain_height_2d()).expect("terrain");
        let voxel = make_stack(scenarios::voxel_density_3d()).expect("voxel");

        let biome_query = ChunkQuery::Grid2D {
            width: 12,
            height: 12,
            origin: [10.0, 20.0],
            step: [1.0, 1.0],
        };
        let terrain_query = ChunkQuery::Grid2D {
            width: 12,
            height: 12,
            origin: [10.0, 20.0],
            step: [1.0, 1.0],
        };
        let voxel_query = ChunkQuery::Grid3D {
            width: 8,
            height: 12,
            depth: 8,
            origin: [10.0, -16.0, 20.0],
            step: [1.0, 1.0, 1.0],
        };

        let biome_summary = run_one(&biome, biome_query.clone()).expect("biome summary");
        terrain
            .update(scenarios::terrain_patch_from_biome(&biome_summary))
            .expect("terrain patch");
        let terrain_summary = run_one(&terrain, terrain_query.clone()).expect("terrain summary");
        voxel
            .update(scenarios::voxel_patch_from_terrain(&terrain_summary))
            .expect("voxel patch");
        let voxel_summary = run_one(&voxel, voxel_query).expect("voxel summary");

        let biome_again = run_one(&biome, biome_query).expect("biome again");
        assert_eq!(biome_summary.checksum, biome_again.checksum);
        assert_ne!(terrain_summary.checksum, 0);
        assert_ne!(voxel_summary.checksum, 0);
    }
}
