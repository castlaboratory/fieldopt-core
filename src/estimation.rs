//! Design-based estimation for the sample.
//!
//! * [`ht_total`]: the Horvitz-Thompson estimator of a population total.
//! * [`local_mean_variance`]: the local-mean variance estimator of Grafström
//!   and Schelin (2014) for spatially balanced samples, which does not need
//!   joint inclusion probabilities.
//! * [`srs_variance`]: the simple-random-sampling variance of the total, as a
//!   reference.

/// Horvitz-Thompson estimate of the total of `y` over the sampled units.
pub fn ht_total(y: &[f64], pi: &[f64], sampled: &[bool]) -> Result<f64, String> {
    check(y, pi, sampled)?;
    Ok((0..y.len())
        .filter(|&i| sampled[i])
        .map(|i| y[i] / pi[i])
        .sum())
}

fn check(y: &[f64], pi: &[f64], sampled: &[bool]) -> Result<(), String> {
    if y.len() != pi.len() || y.len() != sampled.len() {
        return Err("y, pi and sampled must have the same length".into());
    }
    for i in 0..y.len() {
        if sampled[i] {
            if !y[i].is_finite() {
                return Err(format!("y[{i}] of a sampled unit is not finite"));
            }
            if !(pi[i] > 0.0 && pi[i] <= 1.0) {
                return Err(format!("pi[{i}] of a sampled unit must lie in (0, 1]"));
            }
        }
    }
    Ok(())
}

fn sq_dist(coords: &[f64], d: usize, i: usize, j: usize) -> f64 {
    (0..d)
        .map(|k| (coords[i * d + k] - coords[j * d + k]).powi(2))
        .sum()
}

/// Local-mean variance estimator of the Horvitz-Thompson total for a
/// spatially balanced sample. For each sampled unit `i`, the neighbourhood is
/// `i` and its nearest sampled units, added until the sum of their inclusion
/// probabilities reaches one; the estimator is
/// `sum_i n_i / (n_i - 1) * (y_i / pi_i - mean_{j in N_i} y_j / pi_j)^2`.
/// `coords` is the `N x d` row-major coordinate matrix of the population.
pub fn local_mean_variance(
    coords: &[f64],
    d: usize,
    y: &[f64],
    pi: &[f64],
    sampled: &[bool],
) -> Result<f64, String> {
    check(y, pi, sampled)?;
    let n = y.len();
    if d == 0 || coords.len() != n * d {
        return Err(format!("coords must be an {n} x {d} matrix"));
    }
    let s: Vec<usize> = (0..n).filter(|&i| sampled[i]).collect();
    if s.len() < 2 {
        return Err("at least two sampled units are needed".into());
    }
    let mut total = 0.0;
    for &i in &s {
        let mut others: Vec<(f64, usize)> = s
            .iter()
            .filter(|&&j| j != i)
            .map(|&j| (sq_dist(coords, d, i, j), j))
            .collect();
        others.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut members = vec![i];
        let mut pi_sum = pi[i];
        for (_, j) in others {
            if pi_sum >= 1.0 {
                break;
            }
            members.push(j);
            pi_sum += pi[j];
        }
        if members.len() < 2 {
            // isolated unit with pi >= 1: take its nearest neighbour anyway
            let j = s
                .iter()
                .filter(|&&j| j != i)
                .min_by(|&&a, &&b| {
                    sq_dist(coords, d, i, a)
                        .partial_cmp(&sq_dist(coords, d, i, b))
                        .unwrap()
                })
                .copied()
                .unwrap();
            members.push(j);
        }
        let ni = members.len() as f64;
        let mean: f64 = members.iter().map(|&j| y[j] / pi[j]).sum::<f64>() / ni;
        total += ni / (ni - 1.0) * (y[i] / pi[i] - mean).powi(2);
    }
    Ok(total)
}

/// Variance of the estimated total under simple random sampling without
/// replacement of `n` units from `N`, from the sample variance of `y`.
pub fn srs_variance(y: &[f64], sampled: &[bool], big_n: usize) -> Result<f64, String> {
    if y.len() != sampled.len() {
        return Err("y and sampled must have the same length".into());
    }
    let ys: Vec<f64> = (0..y.len()).filter(|&i| sampled[i]).map(|i| y[i]).collect();
    let n = ys.len();
    if n < 2 {
        return Err("at least two sampled units are needed".into());
    }
    if big_n < n {
        return Err("the population size must be at least the sample size".into());
    }
    let mean = ys.iter().sum::<f64>() / n as f64;
    let s2 = ys.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let big_n = big_n as f64;
    Ok(big_n * big_n * (1.0 - n as f64 / big_n) * s2 / n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::local_pivotal;

    #[test]
    fn ht_is_unbiased_under_lpm() {
        let n = 60;
        let coords: Vec<f64> = (0..n)
            .flat_map(|i| vec![(i % 10) as f64, (i / 10) as f64])
            .collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + (i % 10) as f64 * 0.5).collect();
        let pi = vec![0.3; n];
        let truth: f64 = y.iter().sum();
        let reps = 3000;
        let mut sum = 0.0;
        let mut var_est = 0.0;
        for s in 0..reps {
            let sampled = local_pivotal(&coords, 2, &pi, s as u64).unwrap();
            sum += ht_total(&y, &pi, &sampled).unwrap();
            var_est += local_mean_variance(&coords, 2, &y, &pi, &sampled).unwrap();
        }
        let mean = sum / reps as f64;
        assert!(
            (mean - truth).abs() / truth < 0.01,
            "mean {mean} truth {truth}"
        );
        assert!(var_est / reps as f64 > 0.0);
    }

    #[test]
    fn srs_variance_formula() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let sampled = [true, true, true, true];
        // N = 4, n = 4: fpc is zero
        assert_eq!(srs_variance(&y, &sampled, 4).unwrap(), 0.0);
        let sampled = [true, false, true, false];
        // n = 2, s2 = 2, N = 4: 16 * 0.5 * 2 / 2 = 8
        assert!((srs_variance(&y, &sampled, 4).unwrap() - 8.0).abs() < 1e-12);
        assert!(ht_total(&y, &[0.0, 1.0, 1.0, 1.0], &sampled).is_err());
    }
}
