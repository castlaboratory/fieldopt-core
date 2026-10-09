//! Cost-aware allocation across strata (or frames treated as strata).
//!
//! Minimises the field cost `sum_h c_h n_h` subject to a target variance of
//! the estimated total under stratified sampling,
//! `sum_h N_h^2 S_h^2 / n_h - sum_h N_h S_h^2 = V0`, or minimises the variance
//! subject to a cost budget. The solution is the classical optimum
//! `n_h proportional to N_h S_h / sqrt(c_h)` (Cochran, 1977, Section 5.5),
//! scaled to meet the constraint, then bounded to `[2, N_h]`. Bounding makes
//! the result approximate when a bound is active; the function reports the
//! variance and cost actually attained.

/// One stratum of the allocation problem.
#[derive(Clone, Copy, Debug)]
pub struct Stratum {
    /// Number of population units.
    pub size: usize,
    /// Standard deviation of the study variable within the stratum.
    pub sd: f64,
    /// Cost of one unit (interview plus expected access cost).
    pub cost: f64,
}

/// An allocation and what it attains.
#[derive(Clone, Debug, PartialEq)]
pub struct Allocation {
    /// Sample size per stratum.
    pub n: Vec<usize>,
    /// Total cost `sum c_h n_h`.
    pub cost: f64,
    /// Variance of the estimated total attained.
    pub variance: f64,
    /// Whether a bound (`2 <= n_h <= N_h`) was active.
    pub bounded: bool,
}

fn validate(strata: &[Stratum]) -> Result<(), String> {
    if strata.is_empty() {
        return Err("at least one stratum is needed".into());
    }
    for (h, s) in strata.iter().enumerate() {
        if s.size < 2 {
            return Err(format!("stratum {h} has fewer than two units"));
        }
        if !s.sd.is_finite() || s.sd < 0.0 {
            return Err(format!("stratum {h} has an invalid standard deviation"));
        }
        if !s.cost.is_finite() || s.cost <= 0.0 {
            return Err(format!("stratum {h} must have a positive cost"));
        }
    }
    Ok(())
}

fn variance_of(strata: &[Stratum], n: &[usize]) -> f64 {
    strata
        .iter()
        .zip(n)
        .map(|(s, &nh)| {
            let big_n = s.size as f64;
            big_n * big_n * s.sd * s.sd / nh as f64 * (1.0 - nh as f64 / big_n)
        })
        .sum()
}

fn finish(strata: &[Stratum], raw: &[f64]) -> Allocation {
    let mut bounded = false;
    let n: Vec<usize> = strata
        .iter()
        .zip(raw)
        .map(|(s, &r)| {
            let v = r.ceil().max(2.0).min(s.size as f64);
            if (v - r.ceil()).abs() > 1e-9 {
                bounded = true;
            }
            v as usize
        })
        .collect();
    let cost = strata
        .iter()
        .zip(&n)
        .map(|(s, &nh)| s.cost * nh as f64)
        .sum();
    Allocation {
        variance: variance_of(strata, &n),
        cost,
        n,
        bounded,
    }
}

/// Allocation that minimises cost subject to variance at most `target_variance`.
pub fn allocate_for_variance(
    strata: &[Stratum],
    target_variance: f64,
) -> Result<Allocation, String> {
    validate(strata)?;
    if !target_variance.is_finite() || target_variance <= 0.0 {
        return Err("the target variance must be positive".into());
    }
    let a: f64 = strata
        .iter()
        .map(|s| s.size as f64 * s.sd * s.cost.sqrt())
        .sum();
    let b: f64 = strata.iter().map(|s| s.size as f64 * s.sd * s.sd).sum();
    let denom = target_variance + b;
    let raw: Vec<f64> = strata
        .iter()
        .map(|s| s.size as f64 * s.sd / s.cost.sqrt() * a / denom)
        .collect();
    Ok(finish(strata, &raw))
}

/// Allocation that minimises variance subject to cost at most `budget`.
pub fn allocate_for_budget(strata: &[Stratum], budget: f64) -> Result<Allocation, String> {
    validate(strata)?;
    if !budget.is_finite() || budget <= 0.0 {
        return Err("the budget must be positive".into());
    }
    let a: f64 = strata
        .iter()
        .map(|s| s.size as f64 * s.sd * s.cost.sqrt())
        .sum();
    if a <= 0.0 {
        return Err("every stratum has zero standard deviation; nothing to allocate".into());
    }
    let raw: Vec<f64> = strata
        .iter()
        .map(|s| budget * s.size as f64 * s.sd / s.cost.sqrt() / a)
        .collect();
    let mut a = finish(strata, &raw);
    repair_budget(strata, budget, &mut a);
    Ok(a)
}

/// Rounding can push the cost above the budget; drop units, one at a time,
/// from the stratum whose unit removal raises the variance least per unit of
/// cost saved, keeping `n_h >= 2`, until the cost fits. Then add units back
/// greedily while they still fit.
fn repair_budget(strata: &[Stratum], budget: f64, a: &mut Allocation) {
    let gain = |n: &[usize], h: usize, delta: i64| -> f64 {
        let mut m = n.to_vec();
        m[h] = (m[h] as i64 + delta) as usize;
        variance_of(strata, &m)
    };
    while a.cost > budget + 1e-9 {
        let mut best: Option<(usize, f64)> = None;
        for (h, s) in strata.iter().enumerate() {
            if a.n[h] > 2 {
                let rate = (gain(&a.n, h, -1) - a.variance) / s.cost;
                if best.map_or(true, |(_, r)| rate < r) {
                    best = Some((h, rate));
                }
            }
        }
        match best {
            Some((h, _)) => {
                a.n[h] -= 1;
                a.cost -= strata[h].cost;
                a.variance = variance_of(strata, &a.n);
                a.bounded = true;
            }
            None => break,
        }
    }
    loop {
        let mut best: Option<(usize, f64)> = None;
        for (h, s) in strata.iter().enumerate() {
            if a.n[h] < s.size && a.cost + s.cost <= budget + 1e-9 {
                let rate = (a.variance - gain(&a.n, h, 1)) / s.cost;
                if best.map_or(true, |(_, r)| rate > r) {
                    best = Some((h, rate));
                }
            }
        }
        match best {
            Some((h, _)) => {
                a.n[h] += 1;
                a.cost += strata[h].cost;
                a.variance = variance_of(strata, &a.n);
            }
            None => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neyman_with_cost() {
        let strata = vec![
            Stratum {
                size: 1000,
                sd: 10.0,
                cost: 1.0,
            },
            Stratum {
                size: 1000,
                sd: 10.0,
                cost: 4.0,
            },
        ];
        let a = allocate_for_variance(&strata, 1e6).unwrap();
        // equal N and S: n_h proportional to 1 / sqrt(c_h) => ratio 2:1
        assert!(
            (a.n[0] as f64 / a.n[1] as f64 - 2.0).abs() < 0.1,
            "{:?}",
            a.n
        );
        assert!(a.variance <= 1e6 * 1.05);
        let b = allocate_for_budget(&strata, a.cost).unwrap();
        assert!((b.variance - a.variance).abs() / a.variance < 0.05);
        assert!(
            b.cost <= a.cost + 1e-9,
            "cost {} exceeds budget {}",
            b.cost,
            a.cost
        );
        let c = allocate_for_budget(
            &[
                Stratum {
                    size: 2000,
                    sd: 12.0,
                    cost: 40.0,
                },
                Stratum {
                    size: 800,
                    sd: 30.0,
                    cost: 180.0,
                },
            ],
            20000.0,
        )
        .unwrap();
        assert!(c.cost <= 20000.0 && c.cost > 19600.0, "cost {}", c.cost);
        assert!(allocate_for_variance(&strata, -1.0).is_err());
        assert!(allocate_for_variance(
            &[Stratum {
                size: 1,
                sd: 1.0,
                cost: 1.0
            }],
            1.0
        )
        .is_err());
    }

    #[test]
    fn bounds_are_reported() {
        let strata = vec![Stratum {
            size: 10,
            sd: 100.0,
            cost: 1.0,
        }];
        // a huge target variance would allow fewer than two units: bounded below at 2
        let a = allocate_for_variance(&strata, 1e9).unwrap();
        assert_eq!(a.n[0], 2);
        assert!(a.bounded);
        // a tiny target variance asks for the whole stratum: n = N, variance zero
        let b = allocate_for_variance(&strata, 1e-6).unwrap();
        assert_eq!(b.n[0], 10);
        assert_eq!(b.variance, 0.0);
    }
}
