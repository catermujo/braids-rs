#![allow(missing_docs)]
//! FastNoise-based worldgen adapter for `braid`.
#![doc = include_str!("../README.md")]

#[allow(missing_docs)]
mod fastnoise_lite;
mod model;
mod runtime;
pub mod scenarios;

pub use fastnoise_lite::{
    CellularDistanceFunction, CellularReturnType, DomainWarpType, FastNoiseLite, FractalType,
    NoiseType, RotationType3D,
};
pub use model::{
    ChunkQuery, ChunkSummary, CombineNode, CombineOp, FastNoiseChange, FastNoiseCpuBackend,
    FastNoiseGraphSpec, FastNoisePlanner, FastNoisePlannerMeta, GraphDimension, NodeSpec,
    PositionSource, Sample2DNode, Sample3DNode, Warp2DNode, Warp3DNode,
};
pub use runtime::make_cpu_backend;

#[cfg(test)]
mod tests;
