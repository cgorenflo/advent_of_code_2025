use anyhow::Result;
use dec_03::{find_total_joltage, read_input};

fn main() -> Result<()> {
    let banks = read_input("dec_03/input.txt")?;
    let total = find_total_joltage(&banks, 2);
    println!("Part 1: {total}");
    let total = find_total_joltage(&banks, 12);
    println!("Part 2: {total}");
    Ok(())
}
