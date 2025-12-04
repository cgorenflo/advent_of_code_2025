mod common;

use common::{generate_test_values, EXAMPLE_RANGES};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use dec_02::{
    find_repeated_sequences_in_range, find_sum_of_invalid_ids, is_repeated_sequence,
    is_repeated_sequence_div,
};

fn bench_mul(c: &mut Criterion) {
    let test_values = generate_test_values();

    c.bench_function("mul", |b| {
        b.iter(|| {
            for &id in &test_values {
                black_box(is_repeated_sequence(black_box(&id)));
            }
        })
    });
}

fn bench_div(c: &mut Criterion) {
    let test_values = generate_test_values();

    c.bench_function("div", |b| {
        b.iter(|| {
            for &id in &test_values {
                black_box(is_repeated_sequence_div(black_box(&id)));
            }
        })
    });
}

fn bench_range_iterative(c: &mut Criterion) {
    c.bench_function("range_iterative", |b| {
        b.iter(|| {
            black_box(find_sum_of_invalid_ids(
                black_box(EXAMPLE_RANGES),
                |range| range.filter(is_repeated_sequence),
            ))
        })
    });
}

fn bench_range_generative(c: &mut Criterion) {
    c.bench_function("range_generative", |b| {
        b.iter(|| {
            black_box(find_sum_of_invalid_ids(
                black_box(EXAMPLE_RANGES),
                find_repeated_sequences_in_range,
            ))
        })
    });
}

criterion_group!(benches, bench_mul, bench_div, bench_range_iterative, bench_range_generative);
criterion_main!(benches);