use std::fs;

use anyhow::Result;

#[derive(Clone, Copy, Default)]
struct MaxPair {
    first: Option<(usize, u8)>,
    second: Option<(usize, u8)>,
}

impl MaxPair {
    fn joltage(&self) -> Option<u64> {
        let first = self.first?;
        let second = self.second?;
        Some(first.1 as u64 * 10 + second.1 as u64)
    }

    /// Tries to incorporate a digit into the pair. Called right-to-left, so:
    /// - First digit seen goes to `second` (rightmost position)
    /// - Second digit seen goes to `first` (leftmost position)
    /// - Subsequent digits update `first` if >= current, moving best to `second`
    fn try_set(self, index: usize, digit: u8) -> MaxPair {
        match (self.first, self.second) {
            // First digit seen: store as second (rightmost)
            (None, None) => MaxPair {
                second: Some((index, digit)),
                ..self
            },
            // Second digit seen: store as first (leftmost)
            (None, _) => MaxPair {
                first: Some((index, digit)),
                ..self
            },
            // Skip digits smaller than current first
            (Some((_, fst_digit)), _) if digit < fst_digit => self,
            // New first found: move best of old first/second into second
            // Order matters: on ties, max_by_key picks last, and first has smaller index
            (Some(first), Some(second)) => MaxPair {
                first: Some((index, digit)),
                second: [second, first].into_iter().max_by_key(|(_, digit)| *digit),
            },
            (Some(_), None) => unreachable!("second is always filled first"),
        }
    }
}

pub fn read_input(path: &str) -> Result<Vec<Vec<u8>>> {
    Ok(fs::read_to_string(path)?
        .lines()
        .map(|line| line.bytes().map(|b| b - b'0').collect())
        .collect())
}

pub fn find_total_joltage<I, B>(battery_banks: I, batteries_per_bank: usize) -> u64
where
    I: IntoIterator<Item = B>,
    B: AsRef<[u8]>,
{
    let alg = match batteries_per_bank {
        2 => |bank: B| find_max_joltage_2(bank.as_ref()).1,
        12 => |bank: B| find_max_joltage_12(bank.as_ref()).1,
        _ => panic!("only 2 or 12 batteries per bank allowed!"),
    };

    battery_banks.into_iter().map(alg).sum()
}

/// Finds the two digits that form the maximum two-digit number, preserving order.
/// Scans right-to-left so we can greedily pick the leftmost largest digit.
/// Returns (second_idx, joltage).
fn find_max_joltage_2(battery_bank: &[u8]) -> (usize, u64) {
    let pair = battery_bank
        .iter()
        .enumerate()
        .rev()
        .fold(MaxPair::default(), |pair, (index, &digit)| {
            pair.try_set(index, digit)
        });
    (
        pair.second
            .expect("battery bank must have at least 2 digits")
            .0,
        pair.joltage()
            .expect("battery bank must have at least 2 digits"),
    )
}

/// Selects 12 digits by repeatedly picking the best pair from a shrinking window.
/// Each iteration reserves room at the end for the remaining digits to be picked.
fn find_max_joltage_12(battery_bank: &[u8]) -> (usize, u64) {
    // i = remaining digits to pick after this iteration (10, 8, 6, 4, 2, 0)
    (0..=10)
        .rev()
        .step_by(2)
        .fold((0usize, 0u64), |(start, joltage), i| {
            // Slice ends at len-i to leave room for `i` more digits
            let (second_idx, pair_joltage) =
                find_max_joltage_2(&battery_bank[start..(battery_bank.len() - i)]);
            (start + second_idx + 1, joltage * 100 + pair_joltage)
        })
}

#[cfg(test)]
mod tests {
    use crate::find_total_joltage;

    const EXAMPLES: [[u8; 15]; 4] = [
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9],
        [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 7, 8],
        [8, 1, 8, 1, 8, 1, 9, 1, 1, 1, 1, 2, 1, 1, 1],
    ];

    #[test]
    fn part1() {
        let sum = find_total_joltage(EXAMPLES, 2);
        assert_eq!(sum, 357);
    }

    #[test]
    fn part2() {
        let sum = find_total_joltage(EXAMPLES, 12);
        assert_eq!(sum, 3121910778619);
    }
}
