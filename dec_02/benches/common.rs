use std::ops::RangeInclusive;

pub const EXAMPLE_RANGES: [RangeInclusive<u64>; 11] = [
    11..=22,
    95..=115,
    998..=1012,
    1188511880..=1188511890,
    222220..=222224,
    1698522..=1698528,
    446443..=446449,
    38593856..=38593862,
    565653..=565659,
    824824821..=824824827,
    2121212118..=2121212124,
];

pub fn generate_test_values() -> Vec<u64> {
    let mut seed: u64 = 12345;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        seed
    };

    (0..1_000_000).map(|_| next() % 100_000_000_000).collect()
}
