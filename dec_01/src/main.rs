use anyhow::Result;
use dec_01::{Dial, find_password, read_rotations};

fn main() -> Result<()> {
    let rotations = read_rotations("dec_01/input.txt")?;
    let password = find_password(Dial::new(50, 100)?, rotations);

    println!("Part 1: {}", password.landed_on_zero);
    println!("Part 2: {}", password.passed_zero);

    Ok(())
}
