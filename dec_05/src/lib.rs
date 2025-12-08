//! Fresh ingredient ID range analysis.
//!
//! Merges overlapping/adjacent ranges using a single-pass coalesce algorithm,
//! then uses binary search for O(log n) membership lookup.

use std::cmp::Ordering;
use std::ops::RangeInclusive;

use anyhow::{Context, Result};
use itertools::Itertools;

pub fn read_input(path: &str) -> Result<(Vec<RangeInclusive<u64>>, Vec<u64>)> {
    let content = std::fs::read_to_string(path)?;
    let (ranges_section, ids_section) = content
        .split_once("\n\n")
        .context("expected blank line separator")?;

    let ranges = ranges_section
        .lines()
        .map(|line| {
            let (start, end) = line.split_once('-').context("invalid range")?;
            Ok(start.parse()?..=end.parse()?)
        })
        .collect::<Result<_>>()?;

    let ids = ids_section
        .lines()
        .map(|line| Ok(line.parse()?))
        .collect::<Result<_>>()?;

    Ok((ranges, ids))
}

pub fn analyze_fresh_ingredients(ranges: Vec<RangeInclusive<u64>>, ids: &[u64]) -> (u64, u64) {
    let merged = merge_ranges(ranges);

    let available_fresh = ids
        .iter()
        .copied()
        .filter(|&id| is_fresh(&merged, id))
        .count() as u64;

    let total_fresh = merged
        .iter()
        .map(|range| range.end() - range.start() + 1)
        .sum();

    (available_fresh, total_fresh)
}

fn merge_ranges(mut ranges: Vec<RangeInclusive<u64>>) -> Vec<RangeInclusive<u64>> {
    ranges.sort_by_key(|range| *range.start());

    ranges
        .into_iter()
        .coalesce(|prev, next| {
            if *next.start() <= prev.end() + 1 {
                Ok(*prev.start()..=*prev.end().max(next.end()))
            } else {
                Err((prev, next))
            }
        })
        .collect()
}

fn is_fresh(merged: &[RangeInclusive<u64>], id: u64) -> bool {
    merged
        .binary_search_by(|range| {
            if id < *range.start() {
                Ordering::Greater
            } else if id > *range.end() {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        })
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn part1() {
        let ranges = vec![3u64..=5, 10..=14, 16..=20, 12..=18];
        let ids = [1, 5, 8, 11, 17, 32];
        assert_eq!(analyze_fresh_ingredients(ranges, &ids).0, 3);
    }

    #[test]
    fn part2() {
        let ranges = vec![3u64..=5, 10..=14, 16..=20, 12..=18];
        assert_eq!(analyze_fresh_ingredients(ranges, &[]).1, 14);
    }
}
