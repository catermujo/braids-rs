use std::hint::black_box;
use std::time::Duration;

use braid::{BatchScratch, BufferSlot, ComputeScratch, JobPacket, PlannerScratch};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

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

fn make_scratch_state(size: usize) -> (PlannerScratch, BatchScratch, ComputeScratch) {
    let mut planner_scratch = PlannerScratch::default();
    let mut batch_scratch = BatchScratch::default();
    let mut compute_scratch = ComputeScratch::default();

    planner_scratch.bytes.resize(size, 9);
    planner_scratch.u32s.resize(size, 0);
    batch_scratch.u32s.resize(size, 0);
    compute_scratch.u32s.resize(size, 0);
    (planner_scratch, batch_scratch, compute_scratch)
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

    group.bench_with_input(
        BenchmarkId::new("slice_many_4", 1024),
        &1024usize,
        |b, &count| {
            b.iter_batched(
                || make_packet((count / 4) * 4),
                |packet| {
                    let chunked = packet.slice_many::<u32, 4>(BufferSlot(0)).unwrap();
                    black_box(chunked.len())
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("packet_clear_for_reuse", 1024),
        &1024usize,
        |b, &count| {
            let mut packet = make_packet(count);

            b.iter(|| {
                packet.clear_for_reuse();
                black_box(packet.query_count);
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("packet_recycle_and_clear", 1024),
        &1024usize,
        |b, &size| {
            let (mut planner_scratch, mut batch_scratch, mut compute_scratch) =
                make_scratch_state(size);
            let mut packet = make_packet(size);

            b.iter(|| {
                planner_scratch.reset();
                batch_scratch.reset();
                compute_scratch.reset();
                packet.clear_for_reuse();
                black_box((
                    planner_scratch.bytes.len(),
                    batch_scratch.u32s.len(),
                    compute_scratch.u32s.len(),
                    packet.query_count,
                ));
            });
        },
    );

    group.bench_function("packet_recycle_and_clear_4k", |b| {
        let (mut planner_scratch, mut batch_scratch, mut compute_scratch) =
            make_scratch_state(4096);
        let mut packet = make_packet(4096);

        b.iter(|| {
            planner_scratch.reset();
            batch_scratch.reset();
            compute_scratch.reset();
            packet.clear_for_reuse();
            black_box((
                planner_scratch.bytes.len(),
                batch_scratch.u32s.len(),
                compute_scratch.u32s.len(),
                packet.query_count,
            ));
        });
    });

    group.finish();
}

criterion_group!(packet_api_benches, bench_job_packet);
criterion_main!(packet_api_benches);
