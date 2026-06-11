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
        if !values.is_empty() {
            values.clear();
        }
        self.buffers.push(values);
    }

    pub(crate) fn reset(&mut self) {
        for values in &mut self.buffers {
            if !values.is_empty() {
                values.clear();
            }
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
        if !self.bytes.is_empty() {
            self.bytes.clear();
        }
        if !self.u32s.is_empty() {
            self.u32s.clear();
        }
        if !self.u64s.is_empty() {
            self.u64s.clear();
        }
        if !self.f32s.is_empty() {
            self.f32s.clear();
        }
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
        if !self.u32s.is_empty() {
            self.u32s.clear();
        }
        if !self.u64s.is_empty() {
            self.u64s.clear();
        }
        if !self.f32s.is_empty() {
            self.f32s.clear();
        }
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
        if !self.bytes.is_empty() {
            self.bytes.clear();
        }
        if !self.u32s.is_empty() {
            self.u32s.clear();
        }
        if !self.f32s.is_empty() {
            self.f32s.clear();
        }
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

    #[test]
    fn spare_vecs_recycles_non_empty_vectors() {
        use super::SpareVecs;

        let mut spare: SpareVecs<u32> = SpareVecs::default();
        let mut values = spare.checkout();
        values.push(7);
        spare.give_back(values);

        let mut reused = spare.checkout();
        assert!(reused.is_empty());
        assert_eq!(reused.len(), 0);

        reused.push(1);
        spare.give_back(reused);
        let reused_again = spare.checkout();
        assert!(reused_again.is_empty());
    }

    #[test]
    fn scratch_reset_clears_values_and_spare_vectors() {
        use super::{ComputeScratch, PlannerScratch};

        let mut planner_scratch = PlannerScratch {
            bytes: vec![1, 2, 3],
            u32s: vec![1, 2, 3],
            u64s: vec![4, 5],
            f32s: vec![6.0],
            spare_u32s: super::SpareVecs {
                buffers: vec![vec![10, 11]],
            },
            spare_u64s: super::SpareVecs {
                buffers: vec![vec![20]],
            },
            spare_f32s: super::SpareVecs {
                buffers: vec![vec![3.0]],
            },
        };
        planner_scratch.reset();
        assert!(planner_scratch.bytes.is_empty());
        assert!(planner_scratch.u32s.is_empty());
        assert!(planner_scratch.u64s.is_empty());
        assert!(planner_scratch.f32s.is_empty());
        assert!(planner_scratch.spare_u32s.checkout().is_empty());
        assert!(planner_scratch.spare_u64s.checkout().is_empty());
        assert!(planner_scratch.spare_f32s.checkout().is_empty());

        let mut compute_scratch = ComputeScratch {
            bytes: vec![1],
            u32s: vec![2],
            f32s: vec![3.0],
            spare_u32s: super::SpareVecs {
                buffers: vec![vec![11]],
            },
            spare_f32s: super::SpareVecs {
                buffers: vec![vec![22.0]],
            },
        };
        compute_scratch.reset();
        assert!(compute_scratch.bytes.is_empty());
        assert!(compute_scratch.u32s.is_empty());
        assert!(compute_scratch.f32s.is_empty());
        assert!(compute_scratch.spare_u32s.checkout().is_empty());
        assert!(compute_scratch.spare_f32s.checkout().is_empty());
    }
}
