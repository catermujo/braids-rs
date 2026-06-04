#[derive(Debug, Default)]
pub struct PlannerScratch {
    pub bytes: Vec<u8>,
    pub u32s: Vec<u32>,
    pub u64s: Vec<u64>,
    pub f32s: Vec<f32>,
}

impl PlannerScratch {
    pub fn reset(&mut self) {
        self.bytes.clear();
        self.u32s.clear();
        self.u64s.clear();
        self.f32s.clear();
    }
}

#[derive(Debug, Default)]
pub struct BatchScratch {
    pub fact_values: Vec<f32>,
    pub u32s: Vec<u32>,
    pub u64s: Vec<u64>,
    pub f32s: Vec<f32>,
}

impl BatchScratch {
    pub fn reset(&mut self) {
        self.fact_values.clear();
        self.u32s.clear();
        self.u64s.clear();
        self.f32s.clear();
    }
}

#[derive(Debug, Default)]
pub struct ComputeScratch {
    pub bytes: Vec<u8>,
    pub u32s: Vec<u32>,
    pub f32s: Vec<f32>,
}

impl ComputeScratch {
    pub fn reset(&mut self) {
        self.bytes.clear();
        self.u32s.clear();
        self.f32s.clear();
    }
}
