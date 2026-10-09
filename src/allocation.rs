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
            let up = (r - 1e-9).ceil();
            let v = up.max(2.0).min(s.size as f64);
            if r < 2.0 - 1e-9 || r > s.size as f64 + 1e-9 {
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
    // Strata whose unrounded optimum exceeds their size are taken completely
    // (they then contribute no variance) and the rest is re-solved for the
    // same target, until no new stratum hits its size.
    let h = strata.len();
    let mut full = vec![false; h];
    let mut raw = vec![0.0; h];
    loop {
        let a: f64 = (0..h)
            .filter(|&i| !full[i])
            .map(|i| strata[i].size as f64 * strata[i].sd * strata[i].cost.sqrt())
            .sum();
        let b: f64 = (0..h)
            .filter(|&i| !full[i])
            .map(|i| strata[i].size as f64 * strata[i].sd * strata[i].sd)
            .sum();
        let denom = target_variance + b;
        let mut new_full = false;
        for i in 0..h {
            if full[i] {
                raw[i] = strata[i].size as f64;
                continue;
            }
            raw[i] = strata[i].size as f64 * strata[i].sd / strata[i].cost.sqrt() * a / denom;
            if raw[i] >= strata[i].size as f64 {
                full[i] = true;
                new_full = true;
            }
        }
        if !new_full {
            break;
        }
    }
    let mut out = finish(strata, &raw);
    out.bounded = out.bounded || full.iter().any(|&f| f);
    Ok(out)
}

/// Allocation of minimum variance for a budget (cost-optimal allocation of
/// Cochran 1977, Section 5.5, with the cost function `sum c_h n_h`). Strata
/// whose optimum exceeds their size are taken completely and the rest is
/// re-solved; every stratum keeps at least two units (one when it has a
/// single unit), and the budget must cover that floor. Rounding is repaired
/// so that the cost stays within the budget.
pub fn allocate_for_budget(strata: &[Stratum], budget: f64) -> Result<Allocation, String> {
    validate(strata)?;
    if !budget.is_finite() || budget <= 0.0 {
        return Err("the budget must be positive".into());
    }
    let floor_cost: f64 = strata.iter().map(|s| s.cost * (s.size.min(2)) as f64).sum();
    if floor_cost > budget + 1e-9 {
        return Err(format!(
            "the budget {budget} does not cover two units per stratum (cost {floor_cost})"
        ));
    }
    let h = strata.len();
    let a_all: f64 = strata
        .iter()
        .map(|s| s.size as f64 * s.sd * s.cost.sqrt())
        .sum();
    if a_all <= 0.0 {
        return Err("every stratum has zero standard deviation; nothing to allocate".into());
    }
    // Strata whose optimum exceeds their size are taken completely; their
    // cost is removed from the budget and the rest is re-solved.
    let mut full = vec![false; h];
    let mut raw = vec![0.0; h];
    loop {
        let spent: f64 = (0..h)
            .filter(|&i| full[i])
            .map(|i| strata[i].cost * strata[i].size as f64)
            .sum();
        let left = budget - spent;
        if left <= 0.0 {
            return Err(
                "the budget does not cover the strata that must be taken completely".into(),
            );
        }
        let a: f64 = (0..h)
            .filter(|&i| !full[i])
            .map(|i| strata[i].size as f64 * strata[i].sd * strata[i].cost.sqrt())
            .sum();
        let mut new_full = false;
        for i in 0..h {
            if full[i] {
                raw[i] = strata[i].size as f64;
                continue;
            }
            raw[i] = if a > 0.0 {
                left * strata[i].size as f64 * strata[i].sd / strata[i].cost.sqrt() / a
            } else {
                2.0
            };
            if raw[i] >= strata[i].size as f64 {
                full[i] = true;
                new_full = true;
            }
        }
        if !new_full {
            break;
        }
    }
    let mut a = finish(strata, &raw);
    a.bounded = a.bounded || full.iter().any(|&f| f);
    repair_budget(strata, budget, &mut a);
    Ok(a)
}

/// Rounding can push the cost above the budget; drop units, one at a time,
/// (this is a rounding repair, not a bound: `bounded` is left as it is),
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

/// Multivariate allocation: minimum cost `sum c_h n_h` such that the variance
/// of the estimated total of every study variable `j` is at most its target,
/// `sum_h N_h^2 S_hj^2 / n_h - sum_h N_h S_hj^2 <= V0_j` (Bethel 1989;
/// Chromy 1987). `sd` is `H x J` row-major (one column per variable, in the
/// same units as `targets`). The Lagrangian solution `n_h = sqrt(sum_j mu_j
/// a_hj / c_h)` is found by multiplicative updates of the multipliers; strata
/// whose solution exceeds their size are taken completely and the rest
/// re-solved; every stratum keeps at least two units. Returns the allocation
/// and, in `attained`, the variance of each variable.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiAllocation {
    pub n: Vec<usize>,
    pub cost: f64,
    /// Variance attained for each variable.
    pub attained: Vec<f64>,
    /// Whether a bound (`2 <= n_h <= N_h`) moved the solution.
    pub bounded: bool,
    /// Multipliers of the constraints: zero for a constraint that is slack.
    pub multipliers: Vec<f64>,
}

pub fn allocate_multivariate(
    sizes: &[usize],
    costs: &[f64],
    sd: &[f64],
    targets: &[f64],
) -> Result<MultiAllocation, String> {
    let h = sizes.len();
    let j = targets.len();
    if h == 0 || j == 0 || costs.len() != h || sd.len() != h * j {
        return Err("sizes and costs need one entry per stratum, sd one row per stratum and one column per target".into());
    }
    if sizes.iter().any(|&n| n < 2) {
        return Err("every stratum needs at least two units".into());
    }
    if costs.iter().any(|c| !c.is_finite() || *c <= 0.0) {
        return Err("costs must be positive".into());
    }
    if sd.iter().any(|s| !s.is_finite() || *s < 0.0) {
        return Err("standard deviations must be finite and non-negative".into());
    }
    if targets.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("target variances must be positive".into());
    }
    let big_n: Vec<f64> = sizes.iter().map(|&n| n as f64).collect();
    // constraint j: sum_h a_hj / n_h <= b_j
    let a = |hh: usize, jj: usize| big_n[hh].powi(2) * sd[hh * j + jj].powi(2);
    let b: Vec<f64> = (0..j)
        .map(|jj| {
            targets[jj]
                + (0..h)
                    .map(|hh| big_n[hh] * sd[hh * j + jj].powi(2))
                    .sum::<f64>()
        })
        .collect();
    let mut full = vec![false; h];
    let mut raw = vec![0.0; h];
    let mut mu = vec![1.0; j];
    loop {
        // budget left for each constraint after the full strata
        let b_left: Vec<f64> = (0..j)
            .map(|jj| {
                b[jj]
                    - (0..h)
                        .filter(|&hh| full[hh])
                        .map(|hh| a(hh, jj) / big_n[hh])
                        .sum::<f64>()
            })
            .collect();
        if b_left.iter().any(|v| *v <= 0.0) {
            return Err("a target variance cannot be met even by observing every unit".into());
        }
        let free: Vec<usize> = (0..h).filter(|&hh| !full[hh]).collect();
        if free.is_empty() {
            break;
        }
        // binding set: constraints that cannot be satisfied are those with positive mu
        for _ in 0..20_000 {
            for &hh in &free {
                let num: f64 = (0..j).map(|jj| mu[jj] * a(hh, jj)).sum();
                raw[hh] = (num / costs[hh]).sqrt().max(1e-9);
            }
            let ratios: Vec<f64> = (0..j)
                .map(|jj| free.iter().map(|&hh| a(hh, jj) / raw[hh]).sum::<f64>() / b_left[jj])
                .collect();
            let mut done = true;
            for jj in 0..j {
                let r = ratios[jj];
                if r > 1.0 + 1e-9 || (mu[jj] > 1e-12 && r < 1.0 - 1e-9) {
                    done = false;
                }
                mu[jj] *= r.powf(0.5);
                if mu[jj] < 1e-14 {
                    mu[jj] = 0.0;
                }
            }
            if mu.iter().all(|v| *v == 0.0) {
                mu = vec![1e-6; j];
            }
            if done {
                break;
            }
        }
        // scale up so that no constraint is violated after the solve
        let worst = (0..j)
            .map(|jj| free.iter().map(|&hh| a(hh, jj) / raw[hh]).sum::<f64>() / b_left[jj])
            .fold(0.0, f64::max);
        if worst > 1.0 {
            for &hh in &free {
                raw[hh] *= worst;
            }
        }
        let mut new_full = false;
        for &hh in &free {
            if raw[hh] >= big_n[hh] - 1e-9 {
                full[hh] = true;
                raw[hh] = big_n[hh];
                new_full = true;
            }
        }
        if !new_full {
            break;
        }
    }
    let mut bounded = full.iter().any(|&f| f);
    let n: Vec<usize> = (0..h)
        .map(|hh| {
            let up = (raw[hh] - 1e-9).ceil();
            if up < 2.0 {
                bounded = true;
            }
            (up.max(2.0).min(big_n[hh])) as usize
        })
        .collect();
    let cost = (0..h).map(|hh| costs[hh] * n[hh] as f64).sum();
    let attained: Vec<f64> = (0..j)
        .map(|jj| {
            (0..h)
                .map(|hh| a(hh, jj) / n[hh] as f64 - big_n[hh] * sd[hh * j + jj].powi(2))
                .sum()
        })
        .collect();
    Ok(MultiAllocation {
        n,
        cost,
        attained,
        bounded,
        multipliers: mu,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multivariate_allocation_meets_every_target_at_minimum_cost() {
        // two strata, two variables; the second variable is the binding one in stratum 2
        let sizes = [1000usize, 500];
        let costs = [10.0, 30.0];
        let sd = [5.0, 2.0, 3.0, 9.0]; // H x J
                                       // targets from the univariate optimum of each variable at n = 120 total
        let s1 = [
            Stratum {
                size: 1000,
                sd: 5.0,
                cost: 10.0,
            },
            Stratum {
                size: 500,
                sd: 3.0,
                cost: 30.0,
            },
        ];
        let s2 = [
            Stratum {
                size: 1000,
                sd: 2.0,
                cost: 10.0,
            },
            Stratum {
                size: 500,
                sd: 9.0,
                cost: 30.0,
            },
        ];
        let v1 = allocate_for_budget(&s1, 2000.0).unwrap().variance;
        let v2 = allocate_for_budget(&s2, 2000.0).unwrap().variance;
        let m = allocate_multivariate(&sizes, &costs, &sd, &[v1, v2]).unwrap();
        assert!(
            m.attained[0] <= v1 * (1.0 + 1e-6),
            "{} vs {v1}",
            m.attained[0]
        );
        assert!(
            m.attained[1] <= v2 * (1.0 + 1e-6),
            "{} vs {v2}",
            m.attained[1]
        );
        // each univariate solution costs 2000; meeting both must cost at least that
        assert!(m.cost >= 2000.0 - costs[1]);
        // and no cheaper integer allocation meets both targets (grid search)
        let var = |n1: usize, n2: usize, jj: usize| -> f64 {
            let a = |hh: usize| (sizes[hh] as f64).powi(2) * sd[hh * 2 + jj].powi(2);
            let c = |hh: usize| sizes[hh] as f64 * sd[hh * 2 + jj].powi(2);
            a(0) / n1 as f64 + a(1) / n2 as f64 - c(0) - c(1)
        };
        for n1 in 2..400usize {
            for n2 in 2..200usize {
                let c = 10.0 * n1 as f64 + 30.0 * n2 as f64;
                if c < m.cost - 1e-9 && var(n1, n2, 0) <= v1 && var(n1, n2, 1) <= v2 {
                    panic!(
                        "cheaper feasible allocation ({n1}, {n2}) at cost {c} < {}",
                        m.cost
                    );
                }
            }
        }
        // a single variable reproduces the univariate allocation
        let uni = allocate_for_variance(&s1, v1).unwrap();
        let one = allocate_multivariate(&sizes, &costs, &[5.0, 3.0], &[v1]).unwrap();
        assert_eq!(one.n, uni.n);
        // a tiny target takes every unit
        let all = allocate_multivariate(&sizes, &costs, &sd, &[1e-9, v2]).unwrap();
        assert_eq!(all.n, vec![1000, 500]);
        assert!(all.bounded);
        assert!(allocate_multivariate(&sizes, &costs, &sd, &[v1]).is_err());
    }

    #[test]
    fn budget_below_the_floor_is_refused_and_small_optima_are_flagged() {
        let s = [Stratum {
            size: 10,
            sd: 1.0,
            cost: 10.0,
        }];
        assert!(allocate_for_budget(&s, 5.0).is_err());
        let a = allocate_for_budget(&s, 20.0).unwrap();
        assert_eq!(a.n, vec![2]);
        assert!(a.cost <= 20.0 + 1e-9);
        // raw optimum 1.5 rounds to 2 and is reported as bounded
        let one = [Stratum {
            size: 100,
            sd: 1.0,
            cost: 1.0,
        }];
        let v = 100.0f64.powi(2) / 1.5 * (1.0 - 1.5 / 100.0);
        let a = allocate_for_variance(&one, v).unwrap();
        assert_eq!(a.n, vec![2]);
        assert!(a.bounded);
    }

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
    fn full_strata_keep_the_target() {
        // frame B is tiny and cheap: taken completely; the target must still hold
        let strata = vec![
            Stratum {
                size: 680,
                sd: 24.0,
                cost: 103.0,
            },
            Stratum {
                size: 100,
                sd: 32.0,
                cost: 45.0,
            },
        ];
        let a = allocate_for_variance(&strata, 119_025.0).unwrap();
        assert_eq!(a.n[1], 100);
        assert!(
            a.variance <= 119_025.0 * (1.0 + 1e-9),
            "variance {}",
            a.variance
        );
        assert!(a.bounded);
        let b = allocate_for_budget(&strata, 20_000.0).unwrap();
        assert!(b.cost <= 20_000.0 + 1e-9);
        assert!(b.n[1] <= 100);
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
