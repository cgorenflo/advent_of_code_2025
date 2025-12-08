use anyhow::Result;
use dec_04::{
    compute_touching_rolls_in_row, count_all_accessible_rolls, find_accessible_rolls, read_input,
};

fn main() -> Result<()> {
    let grid = read_input("dec_04/input.txt")?;
    let touching_rolls_in_row = compute_touching_rolls_in_row(&grid);
    let part1 = find_accessible_rolls(&grid, &touching_rolls_in_row).len();
    println!("Part 1: {part1}");
    let part2 = count_all_accessible_rolls(grid);
    println!("Part 2: {part2}");
    Ok(())
}
