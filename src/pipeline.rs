//! Generic compiled pipeline types shared between planners and backends.
//!
//! Planners emit these shapes. Backends consume them. The types here intentionally avoid
//! planner-specific meaning.

use crate::error::{BraidError, BraidResult};
use crate::job::JobPacket;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

/// Opaque stack-local job identifier.
pub type JobId = u64;
/// Monotonic identifier for frozen compiled stack versions.
pub type VersionId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Primitive element types supported by packet buffers.
pub enum ElementKind {
    /// Unsigned 32-bit integers.
    U32,
    /// Unsigned 64-bit integers.
    U64,
    /// 32-bit floating-point values.
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Declared layout contract for one pipeline buffer slot.
pub enum BufferLayout {
    /// Exactly one element per query.
    PerQueryScalar,
    /// A fixed-width vector per query.
    PerQueryVector {
        /// Element count for each query.
        width: usize,
    },
    /// Planner/backend-managed variable-length buffer.
    Dynamic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Access mode declared for one kernel binding.
pub enum BufferAccess {
    /// Read-only access.
    Read,
    /// Write-only access.
    Write,
    /// Read-write access.
    ReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Scheduling hint for how one kernel prefers to run.
pub enum DispatchHint {
    /// Run one kernel invocation across the whole batch.
    WholeBatch,
    /// Split the query batch across shards when backend supports it.
    QuerySharded,
    /// Run strictly serially.
    Serial,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
/// Opaque numeric slot identifier for packet and static buffers.
pub struct BufferSlot(pub u16);

impl Display for BufferSlot {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u16> for BufferSlot {
    fn from(value: u16) -> Self {
        Self(value)
    }
}

impl From<BufferSlot> for u16 {
    fn from(value: BufferSlot) -> Self {
        value.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
/// Opaque numeric identifier for backend kernel implementations.
pub struct KernelKind(pub u32);

impl Display for KernelKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u32> for KernelKind {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<KernelKind> for u32 {
    fn from(value: KernelKind) -> Self {
        value.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Declares one buffer slot used by a compiled pipeline.
pub struct BufferSpec {
    /// Slot id used to address the buffer in packets and bindings.
    pub slot: BufferSlot,
    /// Element type stored in this slot.
    pub element_kind: ElementKind,
    /// Declared logical layout of the buffer.
    pub layout: BufferLayout,
}

impl BufferSpec {
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    fn validate_len(&self, query_count: usize, len: usize) -> BraidResult<()> {
        let expected_len = match self.layout {
            BufferLayout::PerQueryScalar => Some(query_count),
            BufferLayout::PerQueryVector { width } => query_count.checked_mul(width),
            BufferLayout::Dynamic => return Ok(()),
        };

        let Some(expected_len) = expected_len else {
            return Err(BraidError::InvalidSpec(format!(
                "buffer slot {} length overflow for declared layout",
                self.slot
            )));
        };

        if len != expected_len {
            return Err(BraidError::InvalidSpec(format!(
                "buffer slot {} has length {} but declared layout expects {}",
                self.slot, len, expected_len
            )));
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// One kernel's view of one pipeline buffer slot.
pub struct BufferBinding {
    /// Slot referenced by the kernel.
    pub slot: BufferSlot,
    /// Declared access mode.
    pub access: BufferAccess,
}

#[derive(Clone, Debug)]
/// One compiled kernel invocation inside a stage.
pub struct KernelSpec {
    /// Backend kernel kind to instantiate.
    pub kind_id: KernelKind,
    /// Planner-defined opaque payload for backend preparation.
    pub payload: Arc<[u8]>,
    /// Buffers this kernel reads or writes.
    pub bindings: Vec<BufferBinding>,
    /// Scheduler hint for batch execution.
    pub dispatch: DispatchHint,
}

#[derive(Clone, Debug, Default)]
/// Barrier-separated group of kernels.
pub struct StageSpec {
    /// Kernels executed within this stage.
    pub kernels: Vec<KernelSpec>,
}

#[derive(Clone, Debug, Default)]
/// Full buffer and stage layout for a compiled plan.
pub struct PipelineShape {
    /// Declared packet/static buffer slots used by the pipeline.
    pub buffers: Vec<BufferSpec>,
    /// Ordered stage list.
    pub stages: Vec<StageSpec>,
}

#[derive(Clone, Debug)]
/// Type-erased buffer storage for packet and static buffers.
pub enum BufferData {
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
}

impl BufferData {
    /// Create an empty buffer for one declared element kind.
    pub(crate) fn empty(kind: ElementKind) -> Self {
        match kind {
            ElementKind::U32 => Self::U32(Vec::new()),
            ElementKind::U64 => Self::U64(Vec::new()),
            ElementKind::F32 => Self::F32(Vec::new()),
        }
    }

    /// Return the element type of this buffer.
    pub fn kind(&self) -> ElementKind {
        match self {
            Self::U32(_) => ElementKind::U32,
            Self::U64(_) => ElementKind::U64,
            Self::F32(_) => ElementKind::F32,
        }
    }

    /// Clear logical contents while keeping capacity for reuse.
    pub fn clear(&mut self) {
        match self {
            Self::U32(vals) => vals.clear(),
            Self::U64(vals) => vals.clear(),
            Self::F32(vals) => vals.clear(),
        }
    }

    /// Return the logical element count.
    pub fn len(&self) -> usize {
        match self {
            Self::U32(vals) => vals.len(),
            Self::U64(vals) => vals.len(),
            Self::F32(vals) => vals.len(),
        }
    }
}

#[derive(Clone, Debug)]
/// Immutable static buffer loaded into packets before stage execution.
pub struct StaticBuffer {
    /// Slot addressed by the static buffer.
    pub slot: BufferSlot,
    /// Static data stored in that slot.
    pub data: BufferData,
}

/// Collection of static buffers attached to a compiled plan.
pub type StaticBufferSet = Vec<StaticBuffer>;

#[derive(Clone, Debug)]
/// Planner output consumed by `Stack` creation, recompile, and backend prepare.
pub struct CompiledPlan<M> {
    /// Generic pipeline layout.
    pub pipeline: PipelineShape,
    /// Planner-provided immutable static buffers.
    pub static_buffers: StaticBufferSet,
    /// Planner-specific metadata preserved for encode/decode.
    pub planner_meta: M,
}

impl<M> CompiledPlan<M> {
    /// Validate slot declarations, static buffers, and kernel bindings.
    pub fn validate(&self) -> BraidResult<()> {
        let specs = self.specs_by_slot()?;
        let mut static_slots = HashMap::with_capacity(self.static_buffers.len());
        for buffer in &self.static_buffers {
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

        for (stage_index, stage) in self.pipeline.stages.iter().enumerate() {
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

    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) fn validate_packet(&self, packet: &JobPacket) -> BraidResult<()> {
        for (slot, kind, len) in packet.buffer_descriptors() {
            let Some(spec) = self.pipeline.buffers.iter().find(|spec| spec.slot == slot) else {
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
            spec.validate_len(packet.query_count, len)?;
        }

        Ok(())
    }

    fn specs_by_slot(&self) -> BraidResult<HashMap<BufferSlot, &BufferSpec>> {
        let mut specs = HashMap::with_capacity(self.pipeline.buffers.len());
        for spec in &self.pipeline.buffers {
            if specs.insert(spec.slot, spec).is_some() {
                return Err(BraidError::InvalidSpec(format!(
                    "duplicate buffer slot {} in pipeline",
                    spec.slot
                )));
            }
        }
        Ok(specs)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BufferAccess, BufferBinding, BufferData, BufferLayout, BufferSlot, BufferSpec,
        CompiledPlan, DispatchHint, ElementKind, KernelKind, KernelSpec, PipelineShape, StageSpec,
        StaticBuffer,
    };
    use crate::job::JobPacket;

    #[test]
    fn bufferdata_utility_methods_cover_len_and_kind() {
        let mut u32_values = BufferData::empty(ElementKind::U32);
        assert_eq!(u32_values.kind(), ElementKind::U32);
        u32_values.clear();
        let u64_values = BufferData::empty(ElementKind::U64);
        assert_eq!(u64_values.kind(), ElementKind::U64);
        assert_eq!(u64_values.len(), 0);
        let f32_values = BufferData::empty(ElementKind::F32);
        assert_eq!(f32_values.kind(), ElementKind::F32);
    }

    #[test]
    fn compiled_plan_validate_catches_decl_errors() {
        let mut plan = CompiledPlan {
            pipeline: PipelineShape {
                buffers: vec![
                    BufferSpec {
                        slot: BufferSlot(1),
                        element_kind: ElementKind::U32,
                        layout: BufferLayout::PerQueryScalar,
                    },
                    BufferSpec {
                        slot: BufferSlot(1),
                        element_kind: ElementKind::U64,
                        layout: BufferLayout::PerQueryVector { width: 3 },
                    },
                ],
                stages: Vec::new(),
            },
            static_buffers: Vec::new(),
            planner_meta: (),
        };
        assert!(plan.validate().is_err());

        plan.pipeline.buffers.pop();
        plan.static_buffers.push(StaticBuffer {
            slot: BufferSlot(9),
            data: BufferData::empty(ElementKind::U32),
        });
        assert!(plan.validate().is_err());

        plan.static_buffers[0].data = BufferData::empty(ElementKind::U64);
        assert!(plan.validate().is_err());
        plan.static_buffers[0].slot = BufferSlot(1);
        assert!(plan.validate().is_err());

        let wrong = CompiledPlan {
            pipeline: PipelineShape {
                buffers: vec![BufferSpec {
                    slot: BufferSlot(2),
                    element_kind: ElementKind::U32,
                    layout: BufferLayout::PerQueryScalar,
                }],
                stages: vec![StageSpec {
                    kernels: vec![KernelSpec {
                        kind_id: KernelKind(1),
                        payload: Vec::<u8>::new().into(),
                        bindings: vec![BufferBinding {
                            slot: BufferSlot(4),
                            access: BufferAccess::Read,
                        }],
                        dispatch: DispatchHint::Serial,
                    }],
                }],
            },
            static_buffers: Vec::new(),
            planner_meta: (),
        };
        assert!(wrong.validate().is_err());
    }

    #[test]
    fn compiled_plan_validate_packet_checks_sizes_and_types() {
        let plan = CompiledPlan {
            pipeline: PipelineShape {
                buffers: vec![
                    BufferSpec {
                        slot: BufferSlot(1),
                        element_kind: ElementKind::U32,
                        layout: BufferLayout::PerQueryVector { width: 2 },
                    },
                    BufferSpec {
                        slot: BufferSlot(2),
                        element_kind: ElementKind::F32,
                        layout: BufferLayout::PerQueryScalar,
                    },
                ],
                stages: Vec::new(),
            },
            static_buffers: Vec::new(),
            planner_meta: (),
        };

        let mut packet = JobPacket::default();
        packet.ensure::<u32>(BufferSlot(3), 0);
        packet.ensure::<u64>(BufferSlot(2), 1);
        assert!(
            plan.validate_packet(&packet).is_err(),
            "wrong packet element kind should fail"
        );

        let mut packet = JobPacket::default();
        packet.ensure::<u32>(BufferSlot(1), 5);
        packet.query_count = 3;
        assert!(
            plan.validate_packet(&packet).is_err(),
            "vector buffer should reject non-multiple lengths"
        );
        packet.ensure::<u32>(BufferSlot(1), 6);
        packet.query_count = 3;
        packet.ensure::<f32>(BufferSlot(2), 1);
        assert!(
            plan.validate_packet(&packet).is_err(),
            "wrong kind in second slot should fail"
        );

        let mut packet = JobPacket::default();
        packet.ensure::<u32>(BufferSlot(1), 4);
        packet.ensure::<f32>(BufferSlot(2), 1);
        packet.query_count = 2;
        assert!(plan.validate_packet(&packet).is_err());

        packet.ensure::<u64>(BufferSlot(3), 0);
        assert!(
            plan.validate_packet(&packet).is_err(),
            "undeclared slot with len > 0 should fail"
        );

        let mut packet = JobPacket::default();
        packet.ensure::<u32>(BufferSlot(1), 4);
        packet.ensure::<f32>(BufferSlot(2), 2);
        packet.query_count = 2;
        assert!(plan.validate_packet(&packet).is_ok());
    }
}
