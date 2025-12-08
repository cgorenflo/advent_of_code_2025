mod common;

use common::generate_test_values;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use dec_02::{is_doubled_sequence, is_doubled_sequence_string};

fn bench_string(c: &mut Criterion) {
    let test_values = generate_test_values();

    c.bench_function("string", |b| {
        b.iter(|| {
            for &id in &test_values {
                black_box(is_doubled_sequence_string(black_box(&id)));
            }
        })
    });
}

fn bench_arithmetic(c: &mut Criterion) {
    let test_values = generate_test_values();

    c.bench_function("arithmetic", |b| {
        b.iter(|| {
            for &id in &test_values {
                black_box(is_doubled_sequence(black_box(&id)));
            }
        })
    });
}

criterion_group!(benches, bench_string, bench_arithmetic);
criterion_main!(benches);
