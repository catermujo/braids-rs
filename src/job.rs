use crate::error::{BraidError, BraidResult};
use crate::pipeline::{BufferData, ElementKind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Default)]
pub struct CancelFlag {
    inner: Arc<AtomicBool>,
}

impl CancelFlag {
    pub fn cancel(&self) {
        self.inner.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug)]
pub struct PacketBuffer {
    pub slot: u16,
    pub data: BufferData,
}

#[derive(Debug, Default)]
pub struct JobPacket {
    query_count: usize,
    buffers: Vec<PacketBuffer>,
}

impl JobPacket {
    pub fn clear_for_reuse(&mut self) {
        self.query_count = 0;
        for buffer in &mut self.buffers {
            buffer.data.clear();
        }
    }

    pub fn query_count(&self) -> usize {
        self.query_count
    }

    pub fn set_query_count(&mut self, query_count: usize) {
        self.query_count = query_count;
    }

    fn ensure_slot(&mut self, slot: u16, expected: ElementKind) -> usize {
        for (idx, buffer) in self.buffers.iter_mut().enumerate() {
            if buffer.slot != slot {
                continue;
            }
            if buffer.data.kind() != expected {
                buffer.data = match expected {
                    ElementKind::U32 => BufferData::U32(Vec::new()),
                    ElementKind::U64 => BufferData::U64(Vec::new()),
                    ElementKind::F32 => BufferData::F32(Vec::new()),
                };
            }
            return idx;
        }

        self.buffers.push(PacketBuffer {
            slot,
            data: match expected {
                ElementKind::U32 => BufferData::U32(Vec::new()),
                ElementKind::U64 => BufferData::U64(Vec::new()),
                ElementKind::F32 => BufferData::F32(Vec::new()),
            },
        });
        self.buffers.len() - 1
    }

    pub fn ensure_u32(&mut self, slot: u16, len: usize) -> &mut Vec<u32> {
        let idx = self.ensure_slot(slot, ElementKind::U32);
        match &mut self.buffers[idx].data {
            BufferData::U32(vals) => {
                vals.resize(len, 0);
                vals
            }
            _ => unreachable!(),
        }
    }

    pub fn ensure_u64(&mut self, slot: u16, len: usize) -> &mut Vec<u64> {
        let idx = self.ensure_slot(slot, ElementKind::U64);
        match &mut self.buffers[idx].data {
            BufferData::U64(vals) => {
                vals.resize(len, 0);
                vals
            }
            _ => unreachable!(),
        }
    }

    pub fn ensure_f32(&mut self, slot: u16, len: usize) -> &mut Vec<f32> {
        let idx = self.ensure_slot(slot, ElementKind::F32);
        match &mut self.buffers[idx].data {
            BufferData::F32(vals) => {
                vals.resize(len, 0.0);
                vals
            }
            _ => unreachable!(),
        }
    }

    pub(crate) fn load_static_buffer(&mut self, slot: u16, data: &BufferData) {
        match data {
            BufferData::U32(values) => {
                self.ensure_u32(slot, values.len()).copy_from_slice(values);
            }
            BufferData::U64(values) => {
                self.ensure_u64(slot, values.len()).copy_from_slice(values);
            }
            BufferData::F32(values) => {
                self.ensure_f32(slot, values.len()).copy_from_slice(values);
            }
        }
    }

    pub(crate) fn buffer_descriptors(
        &self,
    ) -> impl Iterator<Item = (u16, ElementKind, usize)> + '_ {
        self.buffers
            .iter()
            .map(|buffer| (buffer.slot, buffer.data.kind(), buffer.data.len()))
    }

    pub fn u32(&self, slot: u16) -> BraidResult<&[u32]> {
        self.view(slot, ElementKind::U32)
            .and_then(|buffer| match buffer {
                BufferData::U32(vals) => Ok(vals.as_slice()),
                _ => unreachable!(),
            })
    }

    pub fn u32_mut(&mut self, slot: u16) -> BraidResult<&mut [u32]> {
        self.view_mut(slot, ElementKind::U32)
            .and_then(|buffer| match buffer {
                BufferData::U32(vals) => Ok(vals.as_mut_slice()),
                _ => unreachable!(),
            })
    }

    pub fn u64(&self, slot: u16) -> BraidResult<&[u64]> {
        self.view(slot, ElementKind::U64)
            .and_then(|buffer| match buffer {
                BufferData::U64(vals) => Ok(vals.as_slice()),
                _ => unreachable!(),
            })
    }

    pub fn f32(&self, slot: u16) -> BraidResult<&[f32]> {
        self.view(slot, ElementKind::F32)
            .and_then(|buffer| match buffer {
                BufferData::F32(vals) => Ok(vals.as_slice()),
                _ => unreachable!(),
            })
    }

    pub fn f32_mut(&mut self, slot: u16) -> BraidResult<&mut [f32]> {
        self.view_mut(slot, ElementKind::F32)
            .and_then(|buffer| match buffer {
                BufferData::F32(vals) => Ok(vals.as_mut_slice()),
                _ => unreachable!(),
            })
    }

    pub fn with_f32_buffers<R>(
        &mut self,
        slots: &[u16],
        f: impl FnOnce(Vec<&mut [f32]>) -> BraidResult<R>,
    ) -> BraidResult<R> {
        let mut indices = Vec::with_capacity(slots.len());
        for slot in slots {
            let Some(index) = self.buffers.iter().position(|buffer| buffer.slot == *slot) else {
                return Err(BraidError::MissingBuffer(*slot));
            };
            if indices.contains(&index) {
                return Err(BraidError::from("duplicate f32 buffer slot request"));
            }
            let buffer = &self.buffers[index];
            if buffer.data.kind() != ElementKind::F32 {
                return Err(BraidError::InvalidBufferType {
                    slot: *slot,
                    expected: ElementKind::F32,
                });
            }
            indices.push(index);
        }

        let mut ptrs = Vec::with_capacity(indices.len());
        for index in indices {
            let buffer = &mut self.buffers[index];
            match &mut buffer.data {
                BufferData::F32(vals) => ptrs.push(vals.as_mut_slice() as *mut [f32]),
                _ => unreachable!(),
            }
        }

        let mut views = Vec::with_capacity(ptrs.len());
        for ptr in ptrs {
            // The requested slots are unique, so these mutable views do not alias.
            unsafe {
                views.push(&mut *ptr);
            }
        }
        f(views)
    }

    fn view(&self, slot: u16, expected: ElementKind) -> BraidResult<&BufferData> {
        let buffer = self
            .buffers
            .iter()
            .find(|buffer| buffer.slot == slot)
            .ok_or(BraidError::MissingBuffer(slot))?;
        if buffer.data.kind() != expected {
            return Err(BraidError::InvalidBufferType { slot, expected });
        }
        Ok(&buffer.data)
    }

    fn view_mut(&mut self, slot: u16, expected: ElementKind) -> BraidResult<&mut BufferData> {
        let buffer = self
            .buffers
            .iter_mut()
            .find(|buffer| buffer.slot == slot)
            .ok_or(BraidError::MissingBuffer(slot))?;
        if buffer.data.kind() != expected {
            return Err(BraidError::InvalidBufferType { slot, expected });
        }
        Ok(&mut buffer.data)
    }
}
