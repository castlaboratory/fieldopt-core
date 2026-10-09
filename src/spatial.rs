//! Spatially ordered systematic sampling.
//!
//! * [`hilbert_order`]: the order of two-dimensional units along a Hilbert
//!   curve, so that consecutive units are close in space.
//! * [`systematic_replicates`]: systematic sampling with unequal
//!   probabilities along that order, drawn as `r` interpenetrating replicates
//!   (each a systematic sample of about `n / r` units with its own random
//!   start), which makes a design-based variance estimate possible under
//!   systematic selection.

use crate::rng::Rng;

/// Distance along a Hilbert curve of order `bits` for a point of a
/// `2^bits x 2^bits` grid.
fn hilbert_d(bits: u32, mut x: u64, mut y: u64) -> u64 {
    let n = 1u64 << bits;
    let mut d = 0u64;
    let mut s = n / 2;
    while s > 0 {
        let rx = u64::from((x & s) > 0);
        let ry = u64::from((y & s) > 0);
        d += s * s * ((3 * rx) ^ ry);
        // rotate the quadrant
        if ry == 0 {
            if rx == 1 {
                x = n - 1 - x;
                y = n - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
}

/// Order of the units of an `n x 2` row-major coordinate matrix along a
/// Hilbert curve on a grid of `2^bits` cells per side (16 bits is plenty).
/// Returns the permutation: `order[k]` is the index of the k-th unit along
/// the curve.
pub fn hilbert_order(coords: &[f64], bits: u32) -> Result<Vec<usize>, String> {
    if coords.len() % 2 != 0 || coords.is_empty() {
        return Err("coords must be an n x 2 matrix".into());
    }
    if !(1..=31).contains(&bits) {
        return Err("bits must lie between 1 and 31".into());
    }
    if coords.iter().any(|v| !v.is_finite()) {
        return Err("coordinates must be finite".into());
    }
    let n = coords.len() / 2;
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for i in 0..n {
        xmin = xmin.min(coords[2 * i]);
        xmax = xmax.max(coords[2 * i]);
        ymin = ymin.min(coords[2 * i + 1]);
        ymax = ymax.max(coords[2 * i + 1]);
    }
    let side = (xmax - xmin).max(ymax - ymin).max(f64::MIN_POSITIVE);
    let cells = ((1u64 << bits) - 1) as f64;
    let mut keyed: Vec<(u64, usize)> = (0..n)
        .map(|i| {
            let gx = ((coords[2 * i] - xmin) / side * cells).round() as u64;
            let gy = ((coords[2 * i + 1] - ymin) / side * cells).round() as u64;
            (hilbert_d(bits, gx, gy), i)
        })
        .collect();
    keyed.sort();
    Ok(keyed.into_iter().map(|(_, i)| i).collect())
}

/// Systematic sampling with inclusion probabilities `pi` (summing to the
/// sample size `n`, each at most one) along `order`, drawn as `replicates`
/// independent interpenetrating systematic samples. Replicate `k` is a
/// systematic PPS sample with interval `replicates` and its own random
/// start, so it holds about `n / replicates` units with inclusion
/// probabilities `pi / replicates`; replicates are drawn independently and
/// may share units. Returns the `n x replicates` row-major indicator matrix.
/// With one replicate this is plain systematic PPS sampling of exactly `n`
/// units with inclusion probabilities `pi`.
pub fn systematic_replicates(
    order: &[usize],
    pi: &[f64],
    replicates: usize,
    seed: u64,
) -> Result<Vec<bool>, String> {
    let n = pi.len();
    if order.len() != n {
        return Err("order and pi must have the same length".into());
    }
    if replicates == 0 {
        return Err("at least one replicate is needed".into());
    }
    if pi.iter().any(|&p| !(0.0..=1.0).contains(&p)) {
        return Err("inclusion probabilities must lie in [0, 1]".into());
    }
    let total: f64 = pi.iter().sum();
    if total < replicates as f64 - 1e-9 {
        return Err(format!(
            "the sample size {total:.2} is smaller than the number of replicates {replicates}"
        ));
    }
    let r = replicates as f64;
    let mut rng = Rng::new(seed);
    let starts: Vec<f64> = (0..replicates).map(|_| rng.next_f64() * r).collect();
    let mut out = vec![false; n * replicates];
    let mut cum = 0.0;
    for &i in order {
        let lo = cum;
        let hi = cum + pi[i];
        for (k, &u) in starts.iter().enumerate() {
            // first point of replicate k at or after lo: u + r * ceil((lo - u) / r)
            let j = ((lo - u) / r).ceil().max(0.0);
            let p = u + r * j;
            if p >= lo - 1e-12 && p < hi - 1e-12 {
                out[i * replicates + k] = true;
            }
        }
        cum = hi;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hilbert_neighbours_are_close() {
        // 8 x 8 grid: consecutive units along the curve are grid neighbours
        let coords: Vec<f64> = (0..64)
            .flat_map(|i| vec![(i % 8) as f64, (i / 8) as f64])
            .collect();
        let order = hilbert_order(&coords, 3).unwrap();
        assert_eq!(order.len(), 64);
        let mut seen = [false; 64];
        for w in order.windows(2) {
            let (a, b) = (w[0], w[1]);
            let d = (coords[2 * a] - coords[2 * b]).abs()
                + (coords[2 * a + 1] - coords[2 * b + 1]).abs();
            assert_eq!(d, 1.0, "units {a} and {b} are not neighbours");
            seen[a] = true;
        }
        seen[*order.last().unwrap()] = true;
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn systematic_replicates_have_the_right_size_and_probabilities() {
        let n = 100;
        let coords: Vec<f64> = (0..n)
            .flat_map(|i| vec![(i % 10) as f64, (i / 10) as f64])
            .collect();
        let order = hilbert_order(&coords, 4).unwrap();
        let pi: Vec<f64> = (0..n).map(|i| 0.1 + 0.2 * ((i % 3) as f64) / 2.0).collect();
        let total: f64 = pi.iter().sum();
        let reps = 2000;
        let r = 4;
        let mut count = vec![0.0; n];
        for s in 0..reps {
            let m = systematic_replicates(&order, &pi, r, s).unwrap();
            for k in 0..r {
                let sk = (0..n).filter(|&i| m[i * r + k]).count() as f64;
                assert!(
                    (sk - total / r as f64).abs() <= 1.0 + 1e-9,
                    "replicate size {sk}"
                );
                for i in 0..n {
                    if m[i * r + k] {
                        count[i] += 1.0;
                    }
                }
            }
        }
        for i in 0..n {
            let emp = count[i] / (reps as f64 * r as f64);
            assert!(
                (emp - pi[i] / r as f64).abs() < 0.01,
                "unit {i}: {emp} vs {}",
                pi[i] / 4.0
            );
        }
        // one replicate: exact size
        for s in 0..50 {
            let m = systematic_replicates(&order, &pi, 1, s).unwrap();
            let size = m.iter().filter(|&&v| v).count() as f64;
            assert!((size - total).abs() <= 1.0 + 1e-9);
        }
        assert!(systematic_replicates(&order, &pi, 0, 1).is_err());
    }
}
