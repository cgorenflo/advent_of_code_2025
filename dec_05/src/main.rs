use anyhow::Result;
use dec_05::{analyze_fresh_ingredients, read_input};

fn main() -> Result<()> {
    let (ranges, ids) = read_input("dec_05/input.txt")?;
    let (available_fresh, total_fresh) = analyze_fresh_ingredients(ranges, &ids);
    println!("Part 1: {available_fresh}");
    println!("Part 2: {total_fresh}");
    Ok(())
}
