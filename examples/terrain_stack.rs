//! Smallest real `braids` stack example using the FastNoise adapter.

use braids::{BackendConfig, BraidExecutor, Stack};
use braids_fastnoise::{ChunkQuery, FastNoisePlanner, make_cpu_backend, scenarios};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let executor = Arc::new(BraidExecutor::new(4));
    let backend = executor.register_backend(
        Arc::new(make_cpu_backend()),
        BackendConfig { lane_count: 4 },
    );
    let stack = Stack::create(
        Arc::clone(&executor),
        Arc::new(FastNoisePlanner),
        backend,
        scenarios::terrain_height_2d(),
    )?;

    let job = stack.dispatch(vec![ChunkQuery::Grid2D {
        width: 128,
        height: 128,
        origin: [0.0, 0.0],
        step: [1.0, 1.0],
    }])?;
    let summaries = stack.collect(job)?;
    let summary = &summaries[0];

    println!(
        "samples={} min={:.4} max={:.4} mean={:.4} checksum={}",
        summary.samples, summary.min, summary.max, summary.mean, summary.checksum
    );

    Ok(())
}
