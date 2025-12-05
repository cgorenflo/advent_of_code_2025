//! # Algorithm
//!
//! A roll is accessible if it has fewer than 4 neighbors (8-directional).
//!
//! **O(1) neighbor counting**: Instead of checking 8 cells each time, we precompute
//! `touching_rolls_in_row[pos]` = count of rolls in the 3-cell horizontal span at `pos`.
//! Then neighbor count = sum of 3 vertical lookups minus 1 (to exclude self).
//!
//! **Work queue for Part 2**: After removing rolls, only their neighbors might become
//! newly accessible. We track candidates in a HashSet (for deduplication, since neighbors
//! of adjacent removed rolls overlap) instead of rescanning the grid.

use std::collections::HashSet;
use std::fs;
use std::ops::{Index, IndexMut};

use anyhow::Result;
use itertools::iproduct;

pub struct Grid {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl Index<(usize, usize)> for Grid {
    type Output = u8;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row * self.width + col]
    }
}

impl IndexMut<(usize, usize)> for Grid {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[row * self.width + col]
    }
}

impl Grid {
    fn new_zeroed(width: usize, height: usize) -> Self {
        Self {
            data: vec![0u8; width * height],
            width,
            height,
        }
    }
}

impl<const W: usize, const H: usize> From<[[u8; W]; H]> for Grid {
    fn from(rows: [[u8; W]; H]) -> Self {
        Self {
            data: rows.into_iter().flatten().collect(),
            width: W,
            height: H,
        }
    }
}

/// Reads the grid from file with padding: 1 row top/bottom, 2 columns left/right.
/// This avoids bounds checks when accessing neighbors up to 2 cells away.
pub fn read_input(path: &str) -> Result<Grid> {
    let content = fs::read_to_string(path)?;
    let width = content.lines().next().map(str::len).unwrap_or(0) + 4; // 2 padding columns each side

    let data: Vec<u8> = std::iter::repeat_n(0, width) // 1 top padding row
        .chain(content.lines().flat_map(|line| {
            [0, 0]
                .into_iter()
                .chain(line.chars().map(|c| u8::from(c == '@')))
                .chain([0, 0])
        }))
        .chain(std::iter::repeat_n(0, width)) // 1 bottom padding row
        .collect();

    let height = data.len() / width;
    Ok(Grid {
        data,
        width,
        height,
    })
}

/// Precomputes touching rolls in each horizontal 3-cell span for O(1) neighbor counting.
/// Neighbor count at (row, col) = sum of three vertical lookups minus 1 (to exclude self).
pub fn compute_touching_rolls_in_row(grid: &Grid) -> Grid {
    let mut touching_rolls_in_row = Grid::new_zeroed(grid.width, grid.height);

    iproduct!(1..grid.height - 1, 1..grid.width - 1).for_each(|(row, col)| {
        touching_rolls_in_row[(row, col)] =
            grid[(row, col - 1)] + grid[(row, col)] + grid[(row, col + 1)];
    });

    touching_rolls_in_row
}

/// Finds all rolls accessible by forklift (fewer than 4 neighbors).
pub fn find_accessible_rolls(grid: &Grid, touching_rolls_in_row: &Grid) -> HashSet<(usize, usize)> {
    iproduct!(1..grid.height - 1, 2..grid.width - 2)
        .filter(|&pos| is_accessible(grid, touching_rolls_in_row, pos))
        .collect()
}

/// Iteratively removes accessible rolls until none remain.
pub fn count_all_accessible_rolls(mut grid: Grid) -> usize {
    let mut touching_rolls_in_row = compute_touching_rolls_in_row(&grid);
    let mut accessible_candidates = find_accessible_rolls(&grid, &touching_rolls_in_row);
    let mut count = 0;

    while !accessible_candidates.is_empty() {
        let (removed, next) = remove_accessible_rolls(accessible_candidates, &mut grid, &mut touching_rolls_in_row);
        count += removed;
        accessible_candidates = next;
    }

    count
}

fn has_at_least_four_neighbors(touching_rolls_in_row: &Grid, (row, col): (usize, usize)) -> bool {
    // >= 5 because the sum includes the cell itself, so 5 means 4 neighbors
    touching_rolls_in_row[(row - 1, col)]
        + touching_rolls_in_row[(row, col)]
        + touching_rolls_in_row[(row + 1, col)]
        >= 5
}

fn is_accessible(grid: &Grid, touching_rolls_in_row: &Grid, pos: (usize, usize)) -> bool {
    grid[pos] == 1 && !has_at_least_four_neighbors(touching_rolls_in_row, pos)
}

fn remove_accessible_rolls(
    candidates: HashSet<(usize, usize)>,
    grid: &mut Grid,
    touching_rolls_in_row: &mut Grid,
) -> (usize, HashSet<(usize, usize)>) {
    let removable: Vec<_> = candidates
        .into_iter()
        .filter(|&pos| is_accessible(grid, touching_rolls_in_row, pos))
        .collect();

    let removed_count = removable.len();

    for &(row, col) in &removable {
        grid[(row, col)] = 0;
        touching_rolls_in_row[(row, col - 1)] -= 1;
        touching_rolls_in_row[(row, col)] -= 1;
        touching_rolls_in_row[(row, col + 1)] -= 1;
    }

    let next_pending = removable
        .into_iter()
        .flat_map(|(row, col)| {
            iproduct!(-1..=1_isize, -1..=1_isize).map(move |(row_offset, col_offset)| {
                (
                    row.strict_add_signed(row_offset),
                    col.strict_add_signed(col_offset),
                )
            })
        })
        .collect();

    (removed_count, next_pending)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    const EXAMPLE: [[u8; 14]; 12] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // top padding
        [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0], // ..@@.@@@@.
        [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0], // @@@.@.@.@@
        [0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0], // @@@@@.@.@@
        [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0], // @.@@@@..@.
        [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0], // @@.@@@@.@@
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0], // .@@@@@@@.@
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0], // .@.@.@.@@@
        [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0], // @.@@@.@@@@
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], // .@@@@@@@@.
        [0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0], // @.@.@@@.@.
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // bottom padding
    ];

    #[test]
    fn part1() {
        let grid = Grid::from(EXAMPLE);
        let touching_rolls_in_row = compute_touching_rolls_in_row(&grid);
        assert_eq!(find_accessible_rolls(&grid, &touching_rolls_in_row).len(), 13);
    }

    #[test]
    fn part2() {
        assert_eq!(count_all_accessible_rolls(Grid::from(EXAMPLE)), 43);
    }
}
