//! A small, dependency-free, seedable pseudo-random generator.
//!
//! `xoshiro256**` (Blackman and Vigna, 2018) seeded through `splitmix64`, which
//! is the reference seeding procedure. The crate uses it for every randomised
//! step so that a run is reproducible from its seed on every platform.

/// A `xoshiro256**` generator.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Create a generator from a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *v = z ^ (z >> 31);
        }
        Rng { s }
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// A uniform number in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// A uniform integer in `0..n` (`n > 0`).
    pub fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        (self.next_f64() * n as f64) as usize % n
    }

    /// Fisher-Yates shuffle in place.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproducible_and_uniform() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        assert_eq!(a.next_u64(), b.next_u64());
        let mut r = Rng::new(7);
        let m = 100_000;
        let mean: f64 = (0..m).map(|_| r.next_f64()).sum::<f64>() / m as f64;
        assert!((mean - 0.5).abs() < 0.01);
        let mut counts = [0usize; 5];
        for _ in 0..m {
            counts[r.below(5)] += 1;
        }
        for c in counts {
            assert!((c as f64 / m as f64 - 0.2).abs() < 0.01);
        }
    }
}
