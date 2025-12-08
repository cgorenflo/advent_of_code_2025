#[path = "../../benches/common.rs"]
mod common;

use std::hint::black_box;

use common::{EXAMPLE_RANGES, generate_test_values};
use dec_02::{
    find_repeated_sequences_in_range, find_sum_of_invalid_ids, is_doubled_sequence,
    is_doubled_sequence_string, is_repeated_sequence, is_repeated_sequence_div,
};

fn main() {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "help".to_string());

    match mode.as_str() {
        "doubled-arithmetic" => {
            let values = generate_test_values();
            for &id in &values {
                black_box(is_doubled_sequence(black_box(&id)));
            }
        }
        "doubled-string" => {
            let values = generate_test_values();
            for &id in &values {
                black_box(is_doubled_sequence_string(black_box(&id)));
            }
        }
        "repeated-mul" => {
            let values = generate_test_values();
            for &id in &values {
                black_box(is_repeated_sequence(black_box(&id)));
            }
        }
        "repeated-div" => {
            let values = generate_test_values();
            for &id in &values {
                black_box(is_repeated_sequence_div(black_box(&id)));
            }
        }
        "range-iterative" => {
            black_box(find_sum_of_invalid_ids(EXAMPLE_RANGES, |range| {
                range.filter(is_repeated_sequence)
            }));
        }
        "range-generative" => {
            black_box(find_sum_of_invalid_ids(
                EXAMPLE_RANGES,
                find_repeated_sequences_in_range,
            ));
        }
        _ => {
            eprintln!("Usage: profile <mode>");
            eprintln!("Modes:");
            eprintln!("  doubled-arithmetic  - Part 1 arithmetic approach");
            eprintln!("  doubled-string      - Part 1 string approach");
            eprintln!("  repeated-mul        - Part 2 multiplication approach");
            eprintln!("  repeated-div        - Part 2 division approach");
            eprintln!("  range-iterative     - Part 2 iterative range approach");
            eprintln!("  range-generative    - Part 2 generative range approach");
        }
    }
}
