//! Reusable scratch buffers shared across compile, encode, and prepare paths.

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
}

impl PlannerScratch {
    /// Clear logical contents while keeping capacity.
    pub fn reset(&mut self) {
        self.bytes.clear();
        self.u32s.clear();
        self.u64s.clear();
        self.f32s.clear();
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
}

impl BatchScratch {
    /// Clear logical contents while keeping capacity.
    pub fn reset(&mut self) {
        self.u32s.clear();
        self.u64s.clear();
        self.f32s.clear();
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
}

impl ComputeScratch {
    /// Clear logical contents while keeping capacity.
    pub fn reset(&mut self) {
        self.bytes.clear();
        self.u32s.clear();
        self.f32s.clear();
    }
}
