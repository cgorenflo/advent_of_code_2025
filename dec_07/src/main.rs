use anyhow::Result;
use dec_07::{count_splits, many_worlds, read_input};

fn main() -> Result<()> {
    let mut input = read_input("dec_07/input.txt")?;
    let splits = count_splits(&mut input).unwrap_or(0);
    let worlds = many_worlds(&input).unwrap_or(0);
    println!("Part 1: {splits}");
    println!("{input}");
    println!("Part 2: {worlds}");
    Ok(())
}
