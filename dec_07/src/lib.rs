//! Tachyon beam splitter simulation.
//!
//! Beams start at S and move downward. When hitting a splitter (^),
//! the beam stops and two new beams emit left and right.
//!
//! **Part 1**: Count total splits. Simulate beams top-down, tracking active
//! columns with a boolean vector (merges are implicit). Draw paths as `|`.
//!
//! **Part 2**: Count timelines (unique paths). Uses bottom-up DP on the drawn
//! paths from part 1. Each `|` at bottom = 1 timeline. Going up: `|` carries
//! count, `^` sums left+right branches. O(rows×cols) time, O(cols) space.

use std::fs;

use anyhow::Result;

pub fn read_input(path: &str) -> Result<String> {
    Ok(fs::read_to_string(path)?)
}

/// Simulate beams and count splits. Mutates input to draw paths as `|`.
pub fn count_splits(input: &mut str) -> Option<usize> {
    let first_line = input.lines().next()?;
    let width = first_line.len();
    let start = first_line.find('S')?;
    let stride = width + 1; // +1 for newline

    let mut rays = vec![false; width];
    let mut next_rays = vec![false; width];
    rays[start] = true;
    let mut splits = 0;

    // SAFETY: replacing '.' with '|' (both ASCII) preserves UTF-8 validity
    let bytes = unsafe { input.as_bytes_mut() };

    for offset in (stride..bytes.len()).step_by(stride) {
        for col in 0..width {
            if !rays[col] {
                continue;
            }
            match bytes.get(offset + col) {
                Some(b'^') => {
                    // Splitter: count it, draw branches, continue left and right
                    splits += 1;
                    bytes[offset + col - 1] = b'|';
                    bytes[offset + col + 1] = b'|';
                    next_rays[col - 1] = true;
                    next_rays[col + 1] = true;
                }
                Some(b'.') => {
                    // Empty space: draw path, continue straight
                    bytes[offset + col] = b'|';
                    next_rays[col] = true;
                }
                _ => {} // Beam exits grid or hits unknown
            }
        }
        std::mem::swap(&mut rays, &mut next_rays);
        next_rays.fill(false);
    }

    Some(splits)
}

/// Count timelines via bottom-up DP on drawn paths. Requires `count_splits` first.
pub fn many_worlds(input: &str) -> Option<usize> {
    let mut lines = input.lines().rev();
    let last_line = lines.next()?;

    // Bottom row: each | is one timeline ending here
    let mut counts: Vec<usize> = last_line.bytes().map(|b| (b == b'|') as usize).collect();
    let mut new_counts = vec![0; counts.len()];

    // Propagate counts upward
    for line in lines {
        for (col, b) in line.bytes().enumerate() {
            new_counts[col] = match b {
                b'S' => return Some(counts[col]), // Reached start, return total
                b'|' => counts[col],              // Pipe: carry count from below
                b'^' => counts[col - 1] + counts[col + 1], // Splitter: sum branches
                _ => 0,
            }
        }
        std::mem::swap(&mut counts, &mut new_counts);
        new_counts.fill(0);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = "\
.......S.......
...............
.......^.......
...............
......^.^......
...............
.....^.^.^.....
...............
....^.^...^....
...............
...^.^...^.^...
...............
..^...^.....^..
...............
.^.^.^.^.^...^.
...............";

    #[test]
    fn part1() {
        let mut input = EXAMPLE.to_string();
        assert_eq!(count_splits(&mut input), Some(21));
        println!("{input}")
    }

    #[test]
    fn part2() {
        let mut input = EXAMPLE.to_string();
        count_splits(&mut input);
        assert_eq!(many_worlds(&input), Some(40));
    }
}
