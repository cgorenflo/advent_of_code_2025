use anyhow::{anyhow, Result};
use dec_08::{parse_input, solve_part1};

fn main() -> Result<()> {
    let input = std::fs::read_to_string("dec_08/input.txt")?;
    let mut junctions = parse_input(&input)?;
    let result =
        solve_part1(junctions.as_mut_slice(), 1000, 3).ok_or(anyhow!("no solution found"))?;
    println!("Part 1: {result}");
    Ok(())
}
