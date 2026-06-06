//! Reusable scratch buffers shared across compile, encode, and prepare paths.

#[derive(Debug, Default)]
/// Reusable pool of same-typed temporary vectors.
pub struct SpareVecs<T> {
    buffers: Vec<Vec<T>>,
}

impl<T> SpareVecs<T> {
    /// Check out one reusable vector.
    pub fn checkout(&mut self) -> Vec<T> {
        self.buffers.pop().unwrap_or_default()
    }

    /// Return one reusable vector after clearing logical contents.
    pub fn give_back(&mut self, mut values: Vec<T>) {
        values.clear();
        self.buffers.push(values);
    }

    pub(crate) fn reset(&mut self) {
        for values in &mut self.buffers {
            values.clear();
        }
    }
}

#[derive(Debug, Default)]
/// Planner-side scratch reused across compiles.
pub struct PlannerScratch {
    /// Opaque byte scratch.
    pub bytes: Vec<u8>,
    /// `u32` scratch storage.
    pub u32s: Vec<u32>,
    /// `u64` scratch storage.
    pub u64s: Vec<u64>,
    /// `f32` scratch storage.
    pub f32s: Vec<f32>,
    /// Extra reusable `u32` vectors when one primary scratch vector is not enough.
    pub spare_u32s: SpareVecs<u32>,
    /// Extra reusable `u64` vectors when one primary scratch vector is not enough.
    pub spare_u64s: SpareVecs<u64>,
    /// Extra reusable `f32` vectors when one primary scratch vector is not enough.
    pub spare_f32s: SpareVecs<f32>,
}

impl PlannerScratch {
    /// Clear logical contents while keeping capacity.
    pub fn reset(&mut self) {
        self.bytes.clear();
        self.u32s.clear();
        self.u64s.clear();
        self.f32s.clear();
        self.spare_u32s.reset();
        self.spare_u64s.reset();
        self.spare_f32s.reset();
    }
}

#[derive(Debug, Default)]
/// Per-dispatch batch scratch reused during query encoding.
pub struct BatchScratch {
    /// `u32` scratch storage.
    pub u32s: Vec<u32>,
    /// `u64` scratch storage.
    pub u64s: Vec<u64>,
    /// `f32` scratch storage.
    pub f32s: Vec<f32>,
    /// Extra reusable `u32` vectors when one primary scratch vector is not enough.
    pub spare_u32s: SpareVecs<u32>,
    /// Extra reusable `u64` vectors when one primary scratch vector is not enough.
    pub spare_u64s: SpareVecs<u64>,
    /// Extra reusable `f32` vectors when one primary scratch vector is not enough.
    pub spare_f32s: SpareVecs<f32>,
}

impl BatchScratch {
    /// Clear logical contents while keeping capacity.
    pub fn reset(&mut self) {
        self.u32s.clear();
        self.u64s.clear();
        self.f32s.clear();
        self.spare_u32s.reset();
        self.spare_u64s.reset();
        self.spare_f32s.reset();
    }
}

#[derive(Debug, Default)]
/// Backend-side scratch reused during prepare.
pub struct ComputeScratch {
    /// Opaque byte scratch.
    pub bytes: Vec<u8>,
    /// `u32` scratch storage.
    pub u32s: Vec<u32>,
    /// `f32` scratch storage.
    pub f32s: Vec<f32>,
    /// Extra reusable `u32` vectors when one primary scratch vector is not enough.
    pub spare_u32s: SpareVecs<u32>,
    /// Extra reusable `f32` vectors when one primary scratch vector is not enough.
    pub spare_f32s: SpareVecs<f32>,
}

impl ComputeScratch {
    /// Clear logical contents while keeping capacity.
    pub fn reset(&mut self) {
        self.bytes.clear();
        self.u32s.clear();
        self.f32s.clear();
        self.spare_u32s.reset();
        self.spare_f32s.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::BatchScratch;

    #[test]
    fn batch_scratch_reuses_extra_same_typed_vectors() {
        let mut scratch = BatchScratch::default();
        let mut first = scratch.spare_u32s.checkout();
        first.reserve(64);
        let capacity = first.capacity();
        first.extend([1, 2, 3]);
        scratch.spare_u32s.give_back(first);

        let second = scratch.spare_u32s.checkout();
        assert!(second.capacity() >= capacity);
        assert!(second.is_empty());
    }
}
