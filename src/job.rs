//! Per-job packet storage and cancellation/status types.

use crate::error::{BraidError, BraidResult};
use crate::pipeline::{BufferData, BufferSlot, ElementKind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Typed element access contract for [`JobPacket`] buffers.
pub trait PacketElement: Default + Sized {
    /// Buffer element kind associated with this Rust type.
    const KIND: ElementKind;

    /// Return a shared typed slice view if the backing buffer matches.
    fn get(data: &BufferData) -> Option<&[Self]>;

    /// Return a mutable typed vector view if the backing buffer matches.
    fn get_mut(data: &mut BufferData) -> Option<&mut Vec<Self>>;
}

macro_rules! impl_packet_element {
    ($ty:ty, $tag:ident) => {
        impl PacketElement for $ty {
            const KIND: ElementKind = ElementKind::$tag;

            fn get(data: &BufferData) -> Option<&[Self]> {
                match data {
                    BufferData::$tag(values) => Some(values.as_slice()),
                    _ => None,
                }
            }

            fn get_mut(data: &mut BufferData) -> Option<&mut Vec<Self>> {
                match data {
                    BufferData::$tag(values) => Some(values),
                    _ => None,
                }
            }
        }
    };
}

impl_packet_element!(u32, U32);
impl_packet_element!(u64, U64);
impl_packet_element!(f32, F32);

#[derive(Clone, Debug, Default)]
/// Cooperative cancellation flag shared with backend stage execution.
pub struct CancelFlag {
    inner: Arc<AtomicBool>,
}

impl CancelFlag {
    /// Request cancellation.
    pub fn cancel(&self) {
        self.inner.store(true, Ordering::Release);
    }

    /// Reset cancellation back to the clear state.
    pub fn reset(&self) {
        self.inner.store(false, Ordering::Release);
    }

    /// Return whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.inner.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Coarse lifecycle state for a stack-local job.
pub enum JobStatus {
    /// Job is queued but has not started running.
    Queued,
    /// Job is actively encoding, staging, or decoding.
    Running,
    /// Job finished successfully and results can be collected.
    Completed,
    /// Job failed with an error.
    Failed,
    /// Job was cancelled cooperatively.
    Cancelled,
}

#[derive(Debug)]
/// One slot payload inside a [`JobPacket`].
pub struct PacketBuffer {
    /// Buffer slot id.
    pub slot: BufferSlot,
    /// Backing storage for the slot.
    pub data: BufferData,
}

#[derive(Debug, Default)]
/// Reusable per-job mutable buffer set shared between planner and backend.
pub struct JobPacket {
    pub query_count: usize,
    buffers: Vec<PacketBuffer>,
    slot_indices: Vec<Option<usize>>,
}

impl JobPacket {
    /// Clear logical contents while keeping allocations for reuse.
    pub fn clear_for_reuse(&mut self) {
        self.query_count = 0;
        for buffer in &mut self.buffers {
            buffer.data.clear();
        }
    }

    fn slot_index(&self, slot: BufferSlot) -> Option<usize> {
        self.slot_indices.get(slot.0 as usize).copied().flatten()
    }

    fn record_slot_index(&mut self, slot: BufferSlot, index: usize) {
        let raw = slot.0 as usize;
        if self.slot_indices.len() <= raw {
            self.slot_indices.resize(raw + 1, None);
        }
        self.slot_indices[raw] = Some(index);
    }

    fn ensure_slot(&mut self, slot: BufferSlot, expected: ElementKind) -> usize {
        if let Some(idx) = self.slot_index(slot) {
            let buffer = &mut self.buffers[idx];
            if buffer.data.kind() != expected {
                buffer.data = BufferData::empty(expected);
            }
            return idx;
        }

        self.buffers.push(PacketBuffer {
            slot,
            data: BufferData::empty(expected),
        });
        let index = self.buffers.len() - 1;
        self.record_slot_index(slot, index);
        index
    }

    /// Ensure a typed buffer exists at `slot` and resize it to `len`.
    pub fn ensure<T: PacketElement>(&mut self, slot: BufferSlot, len: usize) -> &mut Vec<T> {
        let idx = self.ensure_slot(slot, T::KIND);
        let values = T::get_mut(&mut self.buffers[idx].data).expect("buffer kind mismatch");
        values.resize_with(len, T::default);
        values
    }

    pub(crate) fn load_static_buffer(&mut self, slot: BufferSlot, data: &BufferData) {
        match data {
            BufferData::U32(values) => {
                self.ensure::<u32>(slot, values.len())
                    .copy_from_slice(values);
            }
            BufferData::U64(values) => {
                self.ensure::<u64>(slot, values.len())
                    .copy_from_slice(values);
            }
            BufferData::F32(values) => {
                self.ensure::<f32>(slot, values.len())
                    .copy_from_slice(values);
            }
        }
    }

    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) fn buffer_descriptors(
        &self,
    ) -> impl Iterator<Item = (BufferSlot, ElementKind, usize)> + '_ {
        self.buffers
            .iter()
            .map(|buffer| (buffer.slot, buffer.data.kind(), buffer.data.len()))
    }

    /// Read-only typed slice for one slot.
    pub fn slice<T: PacketElement>(&self, slot: BufferSlot) -> BraidResult<&[T]> {
        let idx = self
            .slot_index(slot)
            .ok_or(BraidError::MissingBuffer(slot))?;
        let buffer = &self.buffers[idx];
        T::get(&buffer.data).ok_or(BraidError::InvalidBufferType {
            slot,
            expected: T::KIND,
        })
    }

    /// Mutable typed slice for one slot.
    pub fn slice_mut<T: PacketElement>(&mut self, slot: BufferSlot) -> BraidResult<&mut [T]> {
        let idx = self
            .slot_index(slot)
            .ok_or(BraidError::MissingBuffer(slot))?;
        let buffer = &mut self.buffers[idx];
        let values = T::get_mut(&mut buffer.data).ok_or(BraidError::InvalidBufferType {
            slot,
            expected: T::KIND,
        })?;
        Ok(values.as_mut_slice())
    }

    /// Read one slot as fixed-width groups of `N` elements.
    pub fn slice_many<T: PacketElement, const N: usize>(
        &self,
        slot: BufferSlot,
    ) -> BraidResult<&[[T; N]]> {
        if N == 0 {
            return Err(BraidError::InvalidSpec(
                "slice_many requires width greater than zero".to_owned(),
            ));
        }
        let values = self.slice::<T>(slot)?;
        let (chunks, remainder) = values.as_chunks::<N>();
        if !remainder.is_empty() {
            return Err(BraidError::InvalidSpec(format!(
                "buffer slot {} length {} not divisible by width {}",
                slot,
                values.len(),
                N
            )));
        }
        Ok(chunks)
    }

    /// Mutably borrow one slot as fixed-width groups of `N` elements.
    pub fn slice_many_mut<T: PacketElement, const N: usize>(
        &mut self,
        slot: BufferSlot,
    ) -> BraidResult<&mut [[T; N]]> {
        if N == 0 {
            return Err(BraidError::InvalidSpec(
                "slice_many_mut requires width greater than zero".to_owned(),
            ));
        }
        let values = self.slice_mut::<T>(slot)?;
        let len = values.len();
        let (chunks, remainder) = values.as_chunks_mut::<N>();
        if !remainder.is_empty() {
            return Err(BraidError::InvalidSpec(format!(
                "buffer slot {} length {} not divisible by width {}",
                slot, len, N
            )));
        }
        Ok(chunks)
    }

    /// Borrow several distinct typed slot slices at once.
    ///
    /// This is useful for kernels that need multiple mutable slices from the same packet without
    /// copying.
    pub fn with_slices<T: PacketElement, const N: usize, R>(
        &mut self,
        slots: [BufferSlot; N],
        f: impl FnOnce([&mut [T]; N]) -> BraidResult<R>,
    ) -> BraidResult<R> {
        let mut indices = [0usize; N];
        for (i, slot) in slots.into_iter().enumerate() {
            let Some(index) = self.slot_index(slot) else {
                return Err(BraidError::MissingBuffer(slot));
            };
            for other in indices.iter().take(i) {
                if *other == index {
                    return Err(BraidError::from("duplicate typed buffer slot request"));
                }
            }
            let buffer = &self.buffers[index];
            if buffer.data.kind() != T::KIND {
                return Err(BraidError::InvalidBufferType {
                    slot,
                    expected: T::KIND,
                });
            }
            indices[i] = index;
        }

        let mut ptrs = [std::ptr::slice_from_raw_parts_mut(std::ptr::null_mut::<T>(), 0); N];
        for (i, index) in indices.into_iter().enumerate() {
            let buffer = &mut self.buffers[index];
            let values = T::get_mut(&mut buffer.data).expect("buffer kind mismatch");
            ptrs[i] = values.as_mut_slice() as *mut [T];
        }

        let views = std::array::from_fn(|i| {
            // The requested slots are unique, so these mutable views do not alias.
            unsafe { &mut *ptrs[i] }
        });
        f(views)
    }
}
