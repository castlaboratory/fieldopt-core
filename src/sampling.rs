//! Probability sampling of field units.
//!
//! * [`inclusion_probabilities`]: probabilities proportional to a size measure
//!   with the usual iterative capping at one.
//! * [`local_pivotal`]: the local pivotal method 2 of Grafström, Lundström
//!   and Schelin (2012), which draws a spatially balanced sample with the
//!   given inclusion probabilities by pairwise updates of nearest neighbours.
//!   Spatial balance here is relative to the coordinates supplied, which may
//!   be geographic coordinates or any auxiliary space (including a travel-cost
//!   embedding).

use crate::rng::Rng;
use crate::spatial_index::GridIndex;

/// Inclusion probabilities proportional to `size`, summing to `n`, with
/// probabilities above one capped at one and the remainder redistributed.
pub fn inclusion_probabilities(size: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let big_n = size.len();
    if n == 0 || n > big_n {
        return Err(format!(
            "sample size {n} must be between 1 and the population size {big_n}"
        ));
    }
    if size.iter().any(|&s| !s.is_finite() || s < 0.0) {
        return Err("sizes must be finite and non-negative".into());
    }
    let positive = size.iter().filter(|&&s| s > 0.0).count();
    if positive < n {
        return Err(format!(
            "only {positive} units have a positive size, fewer than the sample size {n}"
        ));
    }
    let mut pi = vec![0.0; big_n];
    let mut fixed = vec![false; big_n];
    let mut remaining_n = n as f64;
    loop {
        let total: f64 = (0..big_n).filter(|&i| !fixed[i]).map(|i| size[i]).sum();
        let mut capped = false;
        for i in 0..big_n {
            if fixed[i] {
                continue;
            }
            pi[i] = if total > 0.0 {
                remaining_n * size[i] / total
            } else {
                0.0
            };
            if pi[i] >= 1.0 {
                pi[i] = 1.0;
                fixed[i] = true;
                remaining_n -= 1.0;
                capped = true;
            }
        }
        if !capped {
            break;
        }
        if remaining_n <= 0.0 {
            for i in 0..big_n {
                if !fixed[i] {
                    pi[i] = 0.0;
                }
            }
            break;
        }
    }
    Ok(pi)
}

/// Squared distance between rows `i` and `j` of a coordinate matrix with `d`
/// columns stored row-major.
fn sq_dist(coords: &[f64], d: usize, i: usize, j: usize) -> f64 {
    (0..d)
        .map(|k| (coords[i * d + k] - coords[j * d + k]).powi(2))
        .sum()
}

/// Local pivotal method 2. `coords` is an `n x d` row-major matrix; `pi` the
/// inclusion probabilities. Returns the sample as a vector of indicators.
pub fn local_pivotal(coords: &[f64], d: usize, pi: &[f64], seed: u64) -> Result<Vec<bool>, String> {
    let n = pi.len();
    if d == 0 || coords.len() != n * d {
        return Err(format!(
            "coords must be an {n} x {d} matrix (got {} values)",
            coords.len()
        ));
    }
    if pi
        .iter()
        .any(|&p| !p.is_finite() || !(0.0..=1.0).contains(&p))
    {
        return Err("inclusion probabilities must lie in [0, 1]".into());
    }
    if coords.iter().any(|v| !v.is_finite()) {
        return Err("coordinates must be finite".into());
    }
    let mut p = pi.to_vec();
    let mut rng = Rng::new(seed);
    let undecided = |p: &[f64], i: usize| p[i] > 1e-12 && p[i] < 1.0 - 1e-12;
    let mut active: Vec<usize> = (0..n).filter(|&i| undecided(&p, i)).collect();
    // position of each unit in `active`, for removals in constant time
    let mut where_ = vec![usize::MAX; n];
    for (k, &i) in active.iter().enumerate() {
        where_[i] = k;
    }
    let mut index = if d == 2 {
        Some(GridIndex::new(coords, &active))
    } else {
        None
    };
    let remove = |active: &mut Vec<usize>,
                  where_: &mut Vec<usize>,
                  index: &mut Option<GridIndex>,
                  i: usize| {
        let k = where_[i];
        if k == usize::MAX {
            return;
        }
        let last = *active.last().unwrap();
        active.swap_remove(k);
        if last != i {
            where_[last] = k;
        }
        where_[i] = usize::MAX;
        if let Some(ix) = index {
            ix.remove(i);
        }
    };
    while active.len() > 1 {
        let ai = rng.below(active.len());
        let i = active[ai];
        // nearest undecided neighbour, random tie-break
        let ties: Vec<usize> = if let Some(ix) = &index {
            ix.nearest(coords[2 * i], coords[2 * i + 1], i)
                .map(|(_, t)| t)
                .unwrap_or_default()
        } else {
            let mut best = f64::INFINITY;
            let mut ties: Vec<usize> = Vec::new();
            for &j in &active {
                if j == i {
                    continue;
                }
                let dist = sq_dist(coords, d, i, j);
                if dist < best - 1e-12 {
                    best = dist;
                    ties.clear();
                    ties.push(j);
                } else if (dist - best).abs() <= 1e-12 {
                    ties.push(j);
                }
            }
            ties.sort_unstable();
            ties
        };
        if ties.is_empty() {
            break;
        }
        let j = ties[rng.below(ties.len())];
        let (pi_i, pi_j) = (p[i], p[j]);
        let sum = pi_i + pi_j;
        if sum < 1.0 {
            if rng.next_f64() < pi_j / sum {
                p[i] = 0.0;
                p[j] = sum;
            } else {
                p[i] = sum;
                p[j] = 0.0;
            }
        } else if rng.next_f64() < (1.0 - pi_j) / (2.0 - sum) {
            p[i] = 1.0;
            p[j] = sum - 1.0;
        } else {
            p[i] = sum - 1.0;
            p[j] = 1.0;
        }
        for &k in &[i, j] {
            if !undecided(&p, k) {
                remove(&mut active, &mut where_, &mut index, k);
            }
        }
    }
    // a single undecided unit with probability p is included with probability p
    for i in 0..n {
        if undecided(&p, i) {
            p[i] = if rng.next_f64() < p[i] { 1.0 } else { 0.0 };
        }
    }
    Ok(p.iter().map(|&v| v > 0.5).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degenerate_inputs_are_refused() {
        assert!(inclusion_probabilities(&[1.0, 0.0, 0.0], 2).is_err());
        assert!(inclusion_probabilities(&[0.0, 0.0], 1).is_err());
        let pi = vec![0.5; 4];
        let coords = vec![0.0, 0.0, 1.0, 0.0, f64::NAN, 1.0, 1.0, 1.0];
        assert!(local_pivotal(&coords, 2, &pi, 1).is_err());
    }

    #[test]
    fn probabilities_sum_to_n_and_cap_at_one() {
        let pi = inclusion_probabilities(&[1.0, 1.0, 1.0, 1.0], 2).unwrap();
        assert!(pi.iter().all(|&p| (p - 0.5).abs() < 1e-12));
        let pi = inclusion_probabilities(&[100.0, 1.0, 1.0, 1.0, 1.0], 2).unwrap();
        assert_eq!(pi[0], 1.0);
        assert!((pi.iter().sum::<f64>() - 2.0).abs() < 1e-12);
        assert!(inclusion_probabilities(&[1.0, 1.0], 3).is_err());
    }

    #[test]
    fn lpm_has_the_right_size_and_inclusion_rates() {
        let n = 40;
        let coords: Vec<f64> = (0..n)
            .flat_map(|i| vec![(i % 8) as f64, (i / 8) as f64])
            .collect();
        let pi = vec![0.25; n];
        let mut counts = vec![0usize; n];
        let reps = 2000;
        for s in 0..reps {
            let sample = local_pivotal(&coords, 2, &pi, s as u64).unwrap();
            assert_eq!(sample.iter().filter(|&&b| b).count(), 10);
            for (c, &b) in counts.iter_mut().zip(&sample) {
                if b {
                    *c += 1;
                }
            }
        }
        for c in counts {
            let rate = c as f64 / reps as f64;
            assert!((rate - 0.25).abs() < 0.04, "rate {rate}");
        }
    }

    #[test]
    fn lpm_is_spatially_balanced() {
        // on a line, the sample of half the points should have small local
        // imbalance: neighbouring points are rarely both selected
        let n = 100;
        let coords: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let pi = vec![0.5; n];
        let mut adjacent_pairs = 0usize;
        let reps = 200;
        for s in 0..reps {
            let sample = local_pivotal(&coords, 1, &pi, s as u64).unwrap();
            adjacent_pairs += sample.windows(2).filter(|w| w[0] && w[1]).count();
        }
        // independent Bernoulli(0.5) would give 99 * 0.25 = 24.75 adjacent pairs per
        // replication; the local pivotal method roughly halves it
        let mean = adjacent_pairs as f64 / reps as f64;
        assert!(mean < 18.0, "mean adjacent pairs {mean}");
    }
}
