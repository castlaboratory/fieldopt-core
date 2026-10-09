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
    /// Number of primary units in the population and of secondary units per
    /// primary unit (an average when they differ).
    pub n_primary: usize,
    pub m_secondary: usize,
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
    pub bounded: bool,
}

fn validate(t: &TwoStage) -> Result<(), String> {
    if t.n_primary < 2 || t.m_secondary < 1 {
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
    let (big_n, big_m) = (t.n_primary as f64, t.m_secondary as f64);
    (1.0 - nn / big_n) * t.s2_between / nn + (1.0 - mm / big_m) * t.s2_within / (nn * mm)
}

fn finish(t: &TwoStage, n: usize, m: usize, m_opt: f64, bounded: bool) -> TwoStageAllocation {
    let vm = variance_mean(t, n, m);
    let scale = (t.n_primary as f64 * t.m_secondary as f64).powi(2);
    TwoStageAllocation {
        n,
        m,
        m_optimal: m_opt,
        cost: t.c1 * n as f64 + t.c2 * (n * m) as f64,
        variance_mean: vm,
        variance_total: vm * scale,
        bounded,
    }
}

fn optimal_m(t: &TwoStage) -> f64 {
    let denom = t.s2_between - t.s2_within / t.m_secondary as f64;
    if denom <= 0.0 {
        return t.m_secondary as f64; // no gain from more primary units: observe them fully
    }
    (t.c1 * t.s2_within / (t.c2 * denom)).sqrt()
}

/// Minimum cost for a target variance of the estimated **mean** (divide a
/// target variance of the total by `(N M)^2`).
pub fn two_stage_for_variance(
    t: &TwoStage,
    target_variance_mean: f64,
) -> Result<TwoStageAllocation, String> {
    validate(t)?;
    if !(target_variance_mean > 0.0 && target_variance_mean.is_finite()) {
        return Err("the target variance must be positive".into());
    }
    let m_opt = optimal_m(t);
    let m = (m_opt.round().max(1.0) as usize).min(t.m_secondary);
    let mut bounded = (m as f64 - m_opt.round()).abs() > 1e-9;
    // V = A / n - B  with A = S1^2 + (1 - m/M) S2^2 / m, B = S1^2 / N
    let a = t.s2_between + (1.0 - m as f64 / t.m_secondary as f64) * t.s2_within / m as f64;
    let b = t.s2_between / t.n_primary as f64;
    let denom = target_variance_mean + b;
    let n_raw = a / denom;
    let up = (n_raw - 1e-9).ceil();
    let n = (up.max(2.0) as usize).min(t.n_primary);
    if (n as f64 - up).abs() > 1e-9 {
        bounded = true;
    }
    Ok(finish(t, n, m, m_opt, bounded))
}

/// Minimum variance for a budget.
pub fn two_stage_for_budget(t: &TwoStage, budget: f64) -> Result<TwoStageAllocation, String> {
    validate(t)?;
    if !(budget > 0.0 && budget.is_finite()) {
        return Err("the budget must be positive".into());
    }
    let m_opt = optimal_m(t);
    let m = (m_opt.round().max(1.0) as usize).min(t.m_secondary);
    let mut bounded = (m as f64 - m_opt.round()).abs() > 1e-9;
    let per_primary = t.c1 + t.c2 * m as f64;
    let n_raw = (budget / per_primary).floor();
    if n_raw < 2.0 {
        return Err(format!(
            "the budget covers fewer than two primary units at m = {m}"
        ));
    }
    let n = (n_raw as usize).min(t.n_primary);
    if n < n_raw as usize {
        bounded = true;
    }
    Ok(finish(t, n, m, m_opt, bounded))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case() -> TwoStage {
        TwoStage {
            n_primary: 5000,
            m_secondary: 20,
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
}
