use anyhow::Result;
use dec_02::{
    filter_by, find_repeated_sequences_in_range, find_sum_of_invalid_ids, is_doubled_sequence,
    is_doubled_sequence_string, is_repeated_sequence, is_repeated_sequence_div, read_id_ranges,
};

fn main() -> Result<()> {
    let ranges = read_id_ranges("dec_02/input.txt")?;

    println!("Part 1:");
    let sum = find_sum_of_invalid_ids(ranges.clone(), filter_by(is_doubled_sequence));
    println!("  arithmetic (compare halves via division): {sum}");
    let sum = find_sum_of_invalid_ids(ranges.clone(), filter_by(is_doubled_sequence_string));
    println!("  string (split and compare):              {sum}");

    println!("Part 2:");
    let sum = find_sum_of_invalid_ids(ranges.clone(), filter_by(is_repeated_sequence));
    println!("  multiply (build up test number):   {sum}");
    let sum = find_sum_of_invalid_ids(ranges.clone(), filter_by(is_repeated_sequence_div));
    println!("  divide (break down and check):     {sum}");
    let sum = find_sum_of_invalid_ids(ranges, find_repeated_sequences_in_range);
    println!("  generative (produce invalid IDs):  {sum}");

    Ok(())
}
