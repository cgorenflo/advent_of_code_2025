use anyhow::Result;
use dec_06::{parse_problems_part1, parse_problems_part2, read_input, solve_worksheet};

fn main() -> Result<()> {
    let problems = read_input("dec_06/input.txt")?;
    let total = solve_worksheet(&problems, parse_problems_part1);
    println!("Part 1: {total}");

    let total = solve_worksheet(&problems, parse_problems_part2);
    println!("Part 2: {total}");
    Ok(())
}
