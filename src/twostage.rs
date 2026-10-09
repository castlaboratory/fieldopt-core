//! Two-stage allocation with a cost function (Cochran 1977, Section 10.6).
//!
//! `n` primary units (segments, grid cells) are selected and `m` secondary
//! units (points, tracts, establishments) are observed in each, at a cost
//! `C = c1 * n + c2 * n * m`. With `S1^2` the variance among primary-unit
//! means and `S2^2` the variance within primary units, the variance of the
//! estimated mean is `(1 - n/N) S1^2 / n + (1 - m/M) S2^2 / (n m)` and the
//! cost-optimal number of secondary units per primary unit is
//! `m = sqrt(c1 S2^2 / (c2 (S1^2 - S2^2 / M)))`.

/// Inputs of a two-stage allocation.
#[derive(Clone, Copy, Debug)]
pub struct TwoStage {
    /// Number of primary units in the population.
    pub n_primary: usize,
    /// Number of secondary units per primary unit (an average when they
    /// differ; need not be a whole number).
    pub m_secondary: f64,
    /// Variance among primary-unit means and within primary units.
    pub s2_between: f64,
    pub s2_within: f64,
    /// Cost per primary unit visited and per secondary unit observed.
    pub c1: f64,
    pub c2: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TwoStageAllocation {
    pub n: usize,
    pub m: usize,
    /// Unrounded optimal `m`.
    pub m_optimal: f64,
    pub cost: f64,
    /// Variance of the estimated mean per secondary unit and of the total.
    pub variance_mean: f64,
    pub variance_total: f64,
    /// Whether a bound (at least two primary units, at most `N` primary or
    /// `M` secondary units) moved the allocation away from the unconstrained
    /// optimum.
    pub bounded: bool,
}

fn validate(t: &TwoStage) -> Result<(), String> {
    if t.n_primary < 2 || t.m_secondary.is_nan() || t.m_secondary < 1.0 || !t.m_secondary.is_finite() {
        return Err(
            "the population needs at least two primary units and one secondary unit".into(),
        );
    }
    if !(t.s2_between >= 0.0
        && t.s2_within >= 0.0
        && t.s2_between.is_finite()
        && t.s2_within.is_finite())
    {
        return Err("variance components must be finite and non-negative".into());
    }
    if !(t.c1 > 0.0 && t.c2 > 0.0 && t.c1.is_finite() && t.c2.is_finite()) {
        return Err("costs must be positive".into());
    }
    Ok(())
}

fn variance_mean(t: &TwoStage, n: usize, m: usize) -> f64 {
    let (nn, mm) = (n as f64, m as f64);
    let (big_n, big_m) = (t.n_primary as f64, t.m_secondary);
    (1.0 - nn / big_n) * t.s2_between / nn + (1.0 - mm / big_m).max(0.0) * t.s2_within / (nn * mm)
}

fn cost(t: &TwoStage, n: usize, m: usize) -> f64 {
    t.c1 * n as f64 + t.c2 * (n * m) as f64
}

fn finish(t: &TwoStage, n: usize, m: usize, m_opt: f64, bounded: bool) -> TwoStageAllocation {
    let vm = variance_mean(t, n, m);
    let scale = (t.n_primary as f64 * t.m_secondary).powi(2);
    TwoStageAllocation {
        n,
        m,
        m_optimal: m_opt,
        cost: cost(t, n, m),
        variance_mean: vm,
        variance_total: vm * scale,
        bounded,
    }
}

/// Largest whole number of secondary units per primary unit.
fn m_max(t: &TwoStage) -> usize {
    (t.m_secondary + 1e-9).floor().max(1.0) as usize
}

fn optimal_m(t: &TwoStage) -> f64 {
    let denom = t.s2_between - t.s2_within / t.m_secondary;
    if denom <= 0.0 {
        return t.m_secondary; // no gain from more primary units: observe them fully
    }
    (t.c1 * t.s2_within / (t.c2 * denom)).sqrt()
}

/// Smallest `n` (at least 2) whose variance of the mean is at most `v0`
/// at `m` secondary units, or `None` when even all `N` primary units do not
/// reach it.
fn n_for_variance(t: &TwoStage, m: usize, v0: f64) -> Option<usize> {
    // V = A / n - B  with A = S1^2 + (1 - m/M) S2^2 / m, B = S1^2 / N
    let a = t.s2_between + (1.0 - m as f64 / t.m_secondary).max(0.0) * t.s2_within / m as f64;
    let b = t.s2_between / t.n_primary as f64;
    let n_raw = a / (v0 + b);
    let n = ((n_raw - 1e-9).ceil().max(2.0)) as usize;
    if n > t.n_primary {
        None
    } else {
        Some(n)
    }
}

/// Minimum cost for a target variance of the estimated **mean** (divide a
/// target variance of the total by `(N M)^2`). Every whole `m` from 1 to `M`
/// is tried with the smallest `n` that meets the target, and the cheapest
/// pair is returned; `bounded` reports when the unconstrained optimum
/// `(n*, m*)` was not attainable (fewer than two or more than `N` primary
/// units, or `m*` above `M`). Returns an error when the target cannot be
/// met even by observing every unit.
pub fn two_stage_for_variance(
    t: &TwoStage,
    target_variance_mean: f64,
) -> Result<TwoStageAllocation, String> {
    validate(t)?;
    if !(target_variance_mean > 0.0 && target_variance_mean.is_finite()) {
        return Err("the target variance must be positive".into());
    }
    let m_opt = optimal_m(t);
    let mut best: Option<(usize, usize)> = None;
    for m in 1..=m_max(t) {
        if let Some(n) = n_for_variance(t, m, target_variance_mean) {
            let better = match best {
                None => true,
                Some((bn, bm)) => {
                    let (c, bc) = (cost(t, n, m), cost(t, bn, bm));
                    c < bc - 1e-9
                        || ((c - bc).abs() <= 1e-9
                            && variance_mean(t, n, m) < variance_mean(t, bn, bm))
                }
            };
            if better {
                best = Some((n, m));
            }
        }
    }
    let (n, m) = best.ok_or_else(|| {
        "the target variance cannot be reached even by observing every unit".to_string()
    })?;
    // was the unconstrained optimum attainable?
    let m_star = m_opt.round().max(1.0);
    let bounded = m_star > t.m_secondary + 1e-9
        || n_for_variance(t, m_star.min(t.m_secondary) as usize, target_variance_mean).map_or(
            true,
            |n0| {
                n0 == 2 && {
                    let a = t.s2_between
                        + (1.0 - m_star / t.m_secondary).max(0.0) * t.s2_within / m_star;
                    a / (target_variance_mean + t.s2_between / t.n_primary as f64) < 2.0 - 1e-9
                }
            },
        )
        || n == t.n_primary && {
            let a =
                t.s2_between + (1.0 - m as f64 / t.m_secondary).max(0.0) * t.s2_within / m as f64;
            a / (target_variance_mean + t.s2_between / t.n_primary as f64)
                > t.n_primary as f64 + 1e-9
        };
    Ok(finish(t, n, m, m_opt, bounded))
}

/// Minimum variance for a budget. Every whole `m` from 1 to `M` is tried
/// with the largest `n` (at most `N`) that the budget allows, and the pair
/// with the smallest variance is returned (ties go to the cheaper one), so
/// leftover budget is spent on more secondary units when all primary units
/// are already observed. Returns an error when the budget covers fewer than
/// two primary units with one secondary unit each.
pub fn two_stage_for_budget(t: &TwoStage, budget: f64) -> Result<TwoStageAllocation, String> {
    validate(t)?;
    if !(budget > 0.0 && budget.is_finite()) {
        return Err("the budget must be positive".into());
    }
    let m_opt = optimal_m(t);
    let mut best: Option<(usize, usize)> = None;
    for m in 1..=m_max(t) {
        let n_raw = (budget / (t.c1 + t.c2 * m as f64) + 1e-9).floor();
        if n_raw < 2.0 {
            continue;
        }
        let n = (n_raw as usize).min(t.n_primary);
        let better = match best {
            None => true,
            Some((bn, bm)) => {
                let (v, bv) = (variance_mean(t, n, m), variance_mean(t, bn, bm));
                v < bv - 1e-15 || ((v - bv).abs() <= 1e-15 && cost(t, n, m) < cost(t, bn, bm))
            }
        };
        if better {
            best = Some((n, m));
        }
    }
    let (n, m) = best.ok_or_else(|| {
        "the budget covers fewer than two primary units with one secondary unit each".to_string()
    })?;
    let m_star = m_opt.round().max(1.0);
    let n_star = (budget / (t.c1 + t.c2 * m_star.min(t.m_secondary))).floor();
    let bounded = m_star > t.m_secondary + 1e-9 || n_star > t.n_primary as f64 || n_star < 2.0;
    Ok(finish(t, n, m, m_opt, bounded))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case() -> TwoStage {
        TwoStage {
            n_primary: 5000,
            m_secondary: 20.0,
            s2_between: 4.0,
            s2_within: 25.0,
            c1: 300.0,
            c2: 20.0,
        }
    }

    #[test]
    fn cochran_optimum_m() {
        let t = case();
        // sqrt(300 * 25 / (20 * (4 - 25/20))) = sqrt(7500 / 55) = 11.68
        let a = two_stage_for_budget(&t, 100_000.0).unwrap();
        assert!((a.m_optimal - 11.677).abs() < 0.01, "{}", a.m_optimal);
        assert_eq!(a.m, 12);
        assert!(a.cost <= 100_000.0);
        assert!(a.cost > 100_000.0 - (300.0 + 20.0 * 12.0));
        // the optimum has lower variance than neighbouring choices at the same cost
        for m in [6usize, 20] {
            let n = (100_000.0 / (300.0 + 20.0 * m as f64)).floor() as usize;
            assert!(variance_mean(&t, n, m) >= a.variance_mean - 1e-12, "m {m}");
        }
    }

    #[test]
    fn variance_target_and_bounds() {
        let t = case();
        let a = two_stage_for_variance(&t, 0.01).unwrap();
        assert!(a.variance_mean <= 0.01);
        assert!(variance_mean(&t, a.n - 1, a.m) > 0.01, "n is minimal");
        assert!((a.variance_total - a.variance_mean * (5000.0f64 * 20.0).powi(2)).abs() < 1e-6);
        let huge = two_stage_for_variance(&t, 1e9).unwrap();
        assert_eq!(huge.n, 2);
        assert!(huge.bounded);
        assert!(two_stage_for_budget(&t, 10.0).is_err());
        let flat = TwoStage {
            s2_between: 0.0,
            ..t
        };
        assert_eq!(two_stage_for_budget(&flat, 100_000.0).unwrap().m, 20);
    }

    #[test]
    fn small_populations_meet_the_target_or_spend_the_budget() {
        let t = TwoStage {
            n_primary: 5,
            m_secondary: 20.0,
            s2_between: 4.0,
            s2_within: 25.0,
            c1: 300.0,
            c2: 20.0,
        };
        // the old code fixed m = 12, capped n at 5 and returned variance 0.167 > 0.05
        let a = two_stage_for_variance(&t, 0.05).unwrap();
        assert!(a.variance_mean <= 0.05 + 1e-12, "{}", a.variance_mean);
        assert!(a.bounded);
        for m in 1..=20usize {
            let n = ((t.s2_between + (1.0 - m as f64 / 20.0) * 25.0 / m as f64)
                / (0.05 + 4.0 / 5.0))
                .ceil()
                .max(2.0) as usize;
            if n <= 5 {
                assert!(
                    t.c1 * n as f64 + t.c2 * (n * m) as f64 >= a.cost - 1e-9,
                    "m {m}"
                );
            }
        }
        // a generous budget observes every primary unit and spends the rest on m
        let t10 = TwoStage { n_primary: 10, ..t };
        let b = two_stage_for_budget(&t10, 100_000.0).unwrap();
        assert_eq!((b.n, b.m), (10, 20));
        assert!(b.variance_mean.abs() < 1e-12);
        assert!(b.bounded);
        // a fractional M is used as given
        let tf = TwoStage {
            m_secondary: 12.7,
            ..case()
        };
        let ti = TwoStage {
            m_secondary: 12.0,
            ..case()
        };
        assert_ne!(
            two_stage_for_variance(&tf, 0.01).unwrap().m_optimal,
            two_stage_for_variance(&ti, 0.01).unwrap().m_optimal
        );
        assert!(two_stage_for_budget(&t, 500.0).is_err());
    }

    #[test]
    fn budget_optimum_is_the_best_integer_pair() {
        let t = TwoStage {
            n_primary: 5000,
            m_secondary: 20.0,
            s2_between: 1.0,
            s2_within: 4.0,
            c1: 1000.0,
            c2: 60.0,
        };
        let a = two_stage_for_budget(&t, 20_000.0).unwrap();
        for m in 1..=20usize {
            let n = ((20_000.0 / (1000.0 + 60.0 * m as f64)).floor() as usize).min(5000);
            if n >= 2 {
                assert!(variance_mean(&t, n, m) >= a.variance_mean - 1e-15, "m {m}");
            }
        }
    }
}
