use std::fs;
use std::ops::{Add, Neg};
use std::str::FromStr;

use anyhow::{Result, bail};

pub struct Dial {
    value: u8,
    size: u8,
}

impl Dial {
    pub fn new(value: u8, size: u8) -> Result<Self> {
        if value >= size {
            bail!("Invalid dial value")
        }

        Ok(Dial { value, size })
    }

    pub fn rotate(&mut self, rotation: Rotation) -> u16 {
        let raw = rotation + self.value;

        let passed_zero =
            raw.unsigned_abs().div_euclid(self.size as u16) + (raw <= 0 && self.value > 0) as u16;

        self.value = raw
            .rem_euclid(self.size as i16)
            .try_into()
            .expect("the dial's max value cannot exceed u8");

        passed_zero
    }
}

impl PartialEq<u8> for Dial {
    fn eq(&self, other: &u8) -> bool {
        self.value == *other
    }
}

#[derive(Debug, Default)]
pub struct Password {
    pub landed_on_zero: u16,
    pub passed_zero: u16,
}

#[derive(Debug, Copy, Clone)]
pub enum Rotation {
    Left(u16),
    Right(u16),
}

impl<T> Add<T> for Rotation
where
    T: Into<i16>,
{
    type Output = i16;

    fn add(self, rhs: T) -> Self::Output {
        i16::from(self) + rhs.into()
    }
}

// Safe to cast to i16: FromStr validates that val <= i16::MAX.
impl From<Rotation> for i16 {
    fn from(value: Rotation) -> i16 {
        match value {
            Rotation::Left(val) => (val as i16).neg(),
            Rotation::Right(val) => val as i16,
        }
    }
}

impl FromStr for Rotation {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        let (dir, val) = s.split_at(1);
        let val = u16::from_str(val)?;

        if val > i16::MAX as u16 {
            bail!("rotation value larger than expected: {}", val);
        }

        match dir {
            "L" => Ok(Rotation::Left(val)),
            "R" => Ok(Rotation::Right(val)),
            _ => bail!("Invalid direction"),
        }
    }
}

pub fn find_password(
    mut dial: Dial,
    dial_rotations: impl IntoIterator<Item = Rotation>,
) -> Password {
    let mut pw = Password::default();

    for rotation in dial_rotations {
        pw.passed_zero += dial.rotate(rotation);
        pw.landed_on_zero += (dial == 0) as u16;
    }

    pw
}

pub fn read_rotations(path: &str) -> Result<Vec<Rotation>> {
    fs::read_to_string(path)?
        .lines()
        .map(Rotation::from_str)
        .collect()
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::Rotation::{Left, Right};
    use crate::{Dial, Rotation, find_password};

    const SEQUENCE: [Rotation; 10] = [
        Left(68),
        Left(30),
        Right(48),
        Left(5),
        Right(60),
        Left(55),
        Left(1),
        Left(99),
        Right(14),
        Left(82),
    ];

    #[test]
    fn solve_example_part1() -> Result<()> {
        let result = find_password(Dial::new(50, 100)?, SEQUENCE).landed_on_zero;
        assert_eq!(result, 3);

        Ok(())
    }

    #[test]
    fn solve_example_part2() -> Result<()> {
        let result = find_password(Dial::new(50, 100)?, SEQUENCE).passed_zero;
        assert_eq!(result, 6);

        Ok(())
    }
}
