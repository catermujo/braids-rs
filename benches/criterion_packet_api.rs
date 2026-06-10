use std::hint::black_box;
use std::time::Duration;

use braid::{
    BatchScratch, BufferSlot, ComputeScratch, JobPacket, PlannerScratch,
};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};

fn fill_u32_slot(packet: &mut JobPacket, slot: BufferSlot, count: usize) {
    packet.query_count = count;
    let values = packet.ensure::<u32>(slot, count);
    for (index, value) in values.iter_mut().enumerate().take(count) {
        *value = index as u32;
    }
}

fn make_packet(count: usize) -> JobPacket {
    let mut packet = JobPacket::default();
    fill_u32_slot(&mut packet, BufferSlot(0), count);
    packet
}

fn bench_job_packet(c: &mut Criterion) {
    let mut group = c.benchmark_group("job_packet_api");
    group.throughput(Throughput::Elements(1024));
    group.measurement_time(Duration::from_secs(2));

    group.bench_function("ensure_u32_1024", |b| {
        b.iter_batched(
            || (),
            |_| {
                let mut packet = JobPacket::default();
                let values = packet.ensure::<u32>(BufferSlot(0), 1024);
                black_box(values.len())
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("copy_slice_1024", |b| {
        b.iter_batched(
            || make_packet(1024),
            |packet| {
                let source = packet.slice::<u32>(BufferSlot(0)).unwrap();
                let mut total = 0u64;
                for value in source {
                    total = total.wrapping_add(*value as u64);
                }
                black_box(total)
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_with_input(BenchmarkId::new("slice_many_4", 1024), &1024usize, |b, &count| {
        b.iter_batched(
            || make_packet((count / 4) * 4),
            |packet| {
                let chunked = packet.slice_many::<u32, 4>(BufferSlot(0)).unwrap();
                black_box(chunked.len())
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("packet_recycle_and_clear", |b| {
        b.iter_batched(
            || {
                let mut planner_scratch = PlannerScratch::default();
                let mut batch_scratch = BatchScratch::default();
                let mut compute_scratch = ComputeScratch::default();

                planner_scratch.bytes.resize(1024, 9);
                planner_scratch.u32s = (0..1024u32).collect();
                batch_scratch.u32s = (0..1024u32).collect();
                compute_scratch.u32s = (0..1024u32).collect();
                (planner_scratch, batch_scratch, compute_scratch)
            },
            |(mut planner_scratch, mut batch_scratch, mut compute_scratch)| {
                planner_scratch.reset();
                batch_scratch.reset();
                compute_scratch.reset();
                black_box((
                    planner_scratch.bytes.len(),
                    batch_scratch.u32s.len(),
                    compute_scratch.u32s.len(),
                ))
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(packet_api_benches, bench_job_packet);
criterion_main!(packet_api_benches);
