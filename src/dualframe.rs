//! Dual-frame allocation (Hartley 1962, 1974).
//!
//! Two frames cover the population: frame A (say the area frame) and frame B
//! (the list), with an overlap. The population splits into the domains `a`
//! (A only), `ab` (both) and `b` (B only). The Hartley estimator of the total
//! is `Y_a + theta * Y_ab(A) + (1 - theta) * Y_ab(B) + Y_b`, where the
//! overlap is estimated from both samples and mixed with weight `theta`.
//!
//! Under simple random sampling in each frame (with optional design effects
//! for other designs), the variance is the sum of two stratum-like terms,
//! so for a fixed `theta` the cost-optimal sizes `n_A`, `n_B` follow the
//! cost-weighted Neyman rule; the optimal `theta` is found by a grid search
//! with golden-section refinement. `theta = 0` is the screening design in
//! which overlap units found in the area sample contribute nothing.

use crate::allocation::{allocate_for_budget, allocate_for_variance, Stratum};

/// Moments of the study variable in one domain.
#[derive(Clone, Copy, Debug)]
pub struct Domain {
    pub size: usize,
    pub mean: f64,
    pub sd: f64,
}

/// Inputs of a dual-frame allocation.
#[derive(Clone, Copy, Debug)]
pub struct DualFrame {
    pub a: Domain,
    pub ab: Domain,
    pub b: Domain,
    /// Cost per sampled unit of frame A and B.
    pub cost_a: f64,
    pub cost_b: f64,
    /// Design effects of the two samples relative to simple random sampling.
    pub deff_a: f64,
    pub deff_b: f64,
}

/// What the allocation targets.
#[derive(Clone, Copy, Debug)]
pub enum Target {
    /// Minimum cost for this variance of the estimated total.
    Variance(f64),
    /// Minimum variance for this budget.
    Budget(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct DualFrameAllocation {
    pub n_a: usize,
    pub n_b: usize,
    pub theta: f64,
    pub cost: f64,
    pub variance: f64,
    /// Variance contributed by each frame.
    pub variance_a: f64,
    pub variance_b: f64,
    /// Expected overlap units in each sample.
    pub overlap_a: f64,
    pub overlap_b: f64,
    pub bounded: bool,
}

fn validate(f: &DualFrame) -> Result<(), String> {
    for (name, d) in [("a", f.a), ("ab", f.ab), ("b", f.b)] {
        if !d.mean.is_finite() || !d.sd.is_finite() || d.sd < 0.0 {
            return Err(format!("domain {name} has an invalid mean or sd"));
        }
    }
    if f.a.size + f.ab.size < 2 || f.b.size + f.ab.size < 2 {
        return Err("each frame needs at least two units".into());
    }
    if !(f.cost_a > 0.0 && f.cost_b > 0.0 && f.cost_a.is_finite() && f.cost_b.is_finite()) {
        return Err("costs per unit must be positive".into());
    }
    if !(f.deff_a > 0.0 && f.deff_b > 0.0) {
        return Err("design effects must be positive".into());
    }
    Ok(())
}

/// Population standard deviation (divisor N) of `y` times `w` over a frame
/// made of two domains, where `w` is 1 in the first domain and `theta` in the
/// overlap.
fn frame_sd(only: Domain, overlap: Domain, theta: f64) -> f64 {
    let n = (only.size + overlap.size) as f64;
    let m1 = (only.size as f64 * only.mean + overlap.size as f64 * theta * overlap.mean) / n;
    let m2 = (only.size as f64 * (only.sd * only.sd + only.mean * only.mean)
        + overlap.size as f64
            * theta
            * theta
            * (overlap.sd * overlap.sd + overlap.mean * overlap.mean))
        / n;
    (m2 - m1 * m1).max(0.0).sqrt()
}

fn strata_for(f: &DualFrame, theta: f64) -> [Stratum; 2] {
    [
        Stratum {
            size: f.a.size + f.ab.size,
            sd: frame_sd(f.a, f.ab, theta) * f.deff_a.sqrt(),
            cost: f.cost_a,
        },
        Stratum {
            size: f.b.size + f.ab.size,
            sd: frame_sd(f.b, f.ab, 1.0 - theta) * f.deff_b.sqrt(),
            cost: f.cost_b,
        },
    ]
}

fn solve(f: &DualFrame, theta: f64, target: Target) -> Result<DualFrameAllocation, String> {
    let strata = strata_for(f, theta);
    let a = match target {
        Target::Variance(v) => allocate_for_variance(&strata, v)?,
        Target::Budget(c) => allocate_for_budget(&strata, c)?,
    };
    let var_of = |h: usize| {
        let s = strata[h];
        let big_n = s.size as f64;
        let nh = a.n[h] as f64;
        big_n * big_n * s.sd * s.sd / nh * (1.0 - nh / big_n)
    };
    let (va, vb) = (var_of(0), var_of(1));
    Ok(DualFrameAllocation {
        n_a: a.n[0],
        n_b: a.n[1],
        theta,
        cost: a.cost,
        variance: va + vb,
        variance_a: va,
        variance_b: vb,
        overlap_a: a.n[0] as f64 * f.ab.size as f64 / (f.a.size + f.ab.size) as f64,
        overlap_b: a.n[1] as f64 * f.ab.size as f64 / (f.b.size + f.ab.size) as f64,
        bounded: a.bounded,
    })
}

/// Objective to minimise for the search over `theta`.
fn objective(x: &DualFrameAllocation, target: Target) -> f64 {
    match target {
        Target::Variance(_) => x.cost,
        Target::Budget(_) => x.variance,
    }
}

/// Dual-frame allocation. With `theta = Some(t)` the mixing weight is fixed
/// (0 is the screening design); with `None` it is optimised.
pub fn dual_frame_allocation(
    f: &DualFrame,
    theta: Option<f64>,
    target: Target,
) -> Result<DualFrameAllocation, String> {
    validate(f)?;
    match target {
        Target::Variance(v) if !(v > 0.0 && v.is_finite()) => {
            return Err("the target variance must be positive".into())
        }
        Target::Budget(c) if !(c > 0.0 && c.is_finite()) => {
            return Err("the budget must be positive".into())
        }
        _ => {}
    }
    if let Some(t) = theta {
        if !(0.0..=1.0).contains(&t) {
            return Err("theta must lie in [0, 1]".into());
        }
        return solve(f, t, target);
    }
    // grid, then golden-section refinement around the best grid point
    let grid = 101;
    let mut best: Option<DualFrameAllocation> = None;
    for k in 0..grid {
        let t = k as f64 / (grid - 1) as f64;
        let x = solve(f, t, target)?;
        if best
            .as_ref()
            .map_or(true, |b| objective(&x, target) < objective(b, target))
        {
            best = Some(x);
        }
    }
    let b = best.unwrap();
    let (mut lo, mut hi) = ((b.theta - 0.01).max(0.0), (b.theta + 0.01).min(1.0));
    let phi = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..40 {
        let t1 = hi - phi * (hi - lo);
        let t2 = lo + phi * (hi - lo);
        let f1 = objective(&solve(f, t1, target)?, target);
        let f2 = objective(&solve(f, t2, target)?, target);
        if f1 <= f2 {
            hi = t2;
        } else {
            lo = t1;
        }
    }
    let t = (lo + hi) / 2.0;
    let x = solve(f, t, target)?;
    Ok(if objective(&x, target) <= objective(&b, target) {
        x
    } else {
        b
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> DualFrame {
        DualFrame {
            a: Domain {
                size: 8000,
                mean: 5.0,
                sd: 6.0,
            },
            ab: Domain {
                size: 1500,
                mean: 40.0,
                sd: 30.0,
            },
            b: Domain {
                size: 500,
                mean: 60.0,
                sd: 50.0,
            },
            cost_a: 200.0,
            cost_b: 40.0,
            deff_a: 1.0,
            deff_b: 1.0,
        }
    }

    #[test]
    fn optimal_theta_beats_fixed_choices() {
        let f = frame();
        let opt = dual_frame_allocation(&f, None, Target::Budget(100_000.0)).unwrap();
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let fixed = dual_frame_allocation(&f, Some(t), Target::Budget(100_000.0)).unwrap();
            assert!(opt.variance <= fixed.variance + 1e-6, "theta {t}");
            assert!(fixed.cost <= 100_000.0 + 1e-6);
        }
        // the cheap list should carry the overlap: theta small
        assert!(opt.theta < 0.5, "theta {}", opt.theta);
        assert!(opt.n_a >= 2 && opt.n_b >= 2);
        assert!((opt.overlap_a - opt.n_a as f64 * 1500.0 / 9500.0).abs() < 1e-9);
    }

    #[test]
    fn variance_target_is_met_at_minimum_cost() {
        let f = frame();
        let x = dual_frame_allocation(&f, None, Target::Variance(1e8)).unwrap();
        assert!(x.variance <= 1e8 * 1.001, "variance {}", x.variance);
        for t in [0.0, 0.5, 1.0] {
            let y = dual_frame_allocation(&f, Some(t), Target::Variance(1e8)).unwrap();
            assert!(x.cost <= y.cost + 1e-6);
        }
        assert!(dual_frame_allocation(&f, Some(1.5), Target::Budget(1.0)).is_err());
        assert!(dual_frame_allocation(&f, None, Target::Budget(-1.0)).is_err());
    }

    #[test]
    fn frame_sd_matches_direct_computation() {
        // domain a: values {1, 3}; overlap: values {10, 20}; theta 0.5 -> {5, 10}
        let a = Domain {
            size: 2,
            mean: 2.0,
            sd: 1.0,
        };
        let ab = Domain {
            size: 2,
            mean: 15.0,
            sd: 5.0,
        };
        let vals = [1.0, 3.0, 5.0, 10.0];
        let m = vals.iter().sum::<f64>() / 4.0;
        let sd = (vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / 4.0).sqrt();
        assert!((frame_sd(a, ab, 0.5) - sd).abs() < 1e-12);
    }
}
