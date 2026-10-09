//! Balanced sampling and the measure of spatial balance.
//!
//! * [`spatial_balance`]: the Voronoi measure of Stevens and Olsen (2004):
//!   each population unit is assigned to its nearest sampled unit, the
//!   inclusion probabilities assigned to sampled unit `i` are summed to
//!   `v_i`, and the balance is `sum (v_i - 1)^2 / n`. Zero is perfect
//!   balance; simple random sampling gives values around one.
//! * [`cube`]: the cube method of Deville and Tillé (2004), which draws a
//!   sample with the given inclusion probabilities whose Horvitz-Thompson
//!   estimates of the balancing variables match their population totals
//!   (exactly when possible). The flight phase is the fast version of
//!   Chauvet and Tillé (2006), which works on `p + 1` units at a time; the
//!   landing is by suppression of balancing variables, last first, so the
//!   first variable (normally the inclusion probabilities, which fix the
//!   sample size) is kept longest.

use crate::rng::Rng;

/// Voronoi measure of spatial balance of a sample (see the module
/// documentation). `coords` is `n x d` row-major.
pub fn spatial_balance(
    coords: &[f64],
    d: usize,
    pi: &[f64],
    sampled: &[bool],
) -> Result<f64, String> {
    let n = pi.len();
    if d == 0 || coords.len() != n * d || sampled.len() != n {
        return Err("coords must be n x d, pi and sampled of length n".into());
    }
    if coords.iter().any(|v| !v.is_finite()) {
        return Err("coordinates must be finite".into());
    }
    if pi.iter().any(|&p| !(0.0..=1.0).contains(&p)) {
        return Err("inclusion probabilities must lie in [0, 1]".into());
    }
    let s: Vec<usize> = (0..n).filter(|&i| sampled[i]).collect();
    if s.is_empty() {
        return Err("the sample is empty".into());
    }
    let mut v = vec![0.0; s.len()];
    for k in 0..n {
        let mut best = (f64::INFINITY, 0usize);
        for (idx, &i) in s.iter().enumerate() {
            let dist: f64 = (0..d)
                .map(|c| (coords[k * d + c] - coords[i * d + c]).powi(2))
                .sum();
            if dist < best.0 {
                best = (dist, idx);
            }
        }
        v[best.1] += pi[k];
    }
    Ok(v.iter().map(|x| (x - 1.0).powi(2)).sum::<f64>() / s.len() as f64)
}

/// A non-zero vector of the kernel of the `rows x cols` row-major matrix
/// `a`, or `None` when the kernel is trivial.
fn kernel_vector(a: &[f64], rows: usize, cols: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut pivot_col = vec![usize::MAX; rows];
    let mut r = 0;
    let mut is_pivot = vec![false; cols];
    for c in 0..cols {
        if r >= rows {
            break;
        }
        // partial pivoting
        let (mut best, mut best_row) = (0.0, r);
        for i in r..rows {
            if m[i * cols + c].abs() > best {
                best = m[i * cols + c].abs();
                best_row = i;
            }
        }
        if best <= 1e-12 {
            continue;
        }
        if best_row != r {
            for k in 0..cols {
                m.swap(r * cols + k, best_row * cols + k);
            }
        }
        let p = m[r * cols + c];
        for k in 0..cols {
            m[r * cols + k] /= p;
        }
        for i in 0..rows {
            if i != r {
                let f = m[i * cols + c];
                if f != 0.0 {
                    for k in 0..cols {
                        m[i * cols + k] -= f * m[r * cols + k];
                    }
                }
            }
        }
        pivot_col[r] = c;
        is_pivot[c] = true;
        r += 1;
    }
    let free = (0..cols).find(|&c| !is_pivot[c])?;
    let mut u = vec![0.0; cols];
    u[free] = 1.0;
    for i in 0..r {
        u[pivot_col[i]] = -m[i * cols + free];
    }
    Some(u)
}

/// One flight phase on the undecided units with the first `p` balancing
/// variables; `x` is `n x p_all` row-major. Updates `pi` in place.
fn flight(x: &[f64], p_all: usize, p: usize, pi0: &[f64], pi: &mut [f64], rng: &mut Rng) {
    let n = pi.len();
    let undecided = |v: f64| v > 1e-9 && v < 1.0 - 1e-9;
    let mut pool: Vec<usize> = (0..n).filter(|&i| undecided(pi[i])).collect();
    pool.reverse(); // pop from the end gives the original order
    let mut active: Vec<usize> = Vec::with_capacity(p + 1);
    loop {
        while active.len() < p + 1 {
            match pool.pop() {
                Some(i) => active.push(i),
                None => break,
            }
        }
        if active.is_empty() {
            break;
        }
        let q = active.len();
        // A: p x q with columns x_k / pi_k (the original probabilities)
        let mut a = vec![0.0; p * q];
        for (col, &k) in active.iter().enumerate() {
            for j in 0..p {
                a[j * q + col] = x[k * p_all + j] / pi0[k];
            }
        }
        let u = match kernel_vector(&a, p, q) {
            Some(u) => u,
            None => break,
        };
        let (mut l1, mut l2) = (f64::INFINITY, f64::INFINITY);
        for (col, &k) in active.iter().enumerate() {
            let uk = u[col];
            if uk > 1e-15 {
                l1 = l1.min((1.0 - pi[k]) / uk);
                l2 = l2.min(pi[k] / uk);
            } else if uk < -1e-15 {
                l1 = l1.min(-pi[k] / uk);
                l2 = l2.min((pi[k] - 1.0) / uk);
            }
        }
        if !l1.is_finite() || !l2.is_finite() {
            break;
        }
        let up = rng.next_f64() < l2 / (l1 + l2);
        for (col, &k) in active.iter().enumerate() {
            pi[k] += if up { l1 * u[col] } else { -l2 * u[col] };
            if pi[k] < 1e-9 {
                pi[k] = 0.0;
            } else if pi[k] > 1.0 - 1e-9 {
                pi[k] = 1.0;
            }
        }
        active.retain(|&k| undecided(pi[k]));
        if pool.is_empty() && active.len() <= p {
            break;
        }
    }
}

/// Cube sampling with inclusion probabilities `pi` balanced on the `n x p`
/// row-major matrix `x` (see the module documentation). Returns the sample
/// indicators.
pub fn cube(x: &[f64], p: usize, pi: &[f64], seed: u64) -> Result<Vec<bool>, String> {
    let n = pi.len();
    if p == 0 || x.len() != n * p {
        return Err("x must be n x p with p at least 1".into());
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err("balancing variables must be finite".into());
    }
    if pi.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
        return Err("inclusion probabilities must lie in [0, 1]".into());
    }
    let mut rng = Rng::new(seed);
    let mut cur = pi.to_vec();
    // random order of the units, as in the reference implementations
    let mut order: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut order);
    let xs: Vec<f64> = order
        .iter()
        .flat_map(|&i| x[i * p..(i + 1) * p].to_vec())
        .collect();
    let mut ps: Vec<f64> = order.iter().map(|&i| cur[i]).collect();
    let ps0 = ps.clone();
    let mut keep = p;
    loop {
        flight(&xs, p, keep, &ps0, &mut ps, &mut rng);
        if ps.iter().all(|&v| v <= 1e-9 || v >= 1.0 - 1e-9) || keep == 0 {
            break;
        }
        keep -= 1; // landing by suppression of the last balancing variable
    }
    // whatever is still undecided (only when all variables were dropped)
    for v in ps.iter_mut() {
        if *v > 1e-9 && *v < 1.0 - 1e-9 {
            *v = if rng.next_f64() < *v { 1.0 } else { 0.0 };
        }
    }
    for (k, &i) in order.iter().enumerate() {
        cur[i] = ps[k];
    }
    Ok(cur.iter().map(|&v| v >= 0.5).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n_side: usize) -> Vec<f64> {
        (0..n_side * n_side)
            .flat_map(|i| vec![(i % n_side) as f64, (i / n_side) as f64])
            .collect()
    }

    #[test]
    fn kernel_vectors_are_in_the_kernel() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 x 3
        let u = kernel_vector(&a, 2, 3).unwrap();
        for i in 0..2 {
            let dot: f64 = (0..3).map(|j| a[i * 3 + j] * u[j]).sum();
            assert!(dot.abs() < 1e-9);
        }
        assert!(u.iter().any(|v| v.abs() > 1e-9));
        assert!(kernel_vector(&[1.0, 0.0, 0.0, 1.0], 2, 2).is_none());
    }

    #[test]
    fn cube_balances_on_pi_and_a_covariate() {
        let n = 100;
        let z: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();
        let pi = vec![0.2; n];
        // balance on pi (fixed size) and on z
        let x: Vec<f64> = (0..n).flat_map(|i| vec![pi[i], z[i]]).collect();
        let total_z: f64 = z.iter().sum();
        let mut sizes_ok = 0;
        let (mut err_cube, mut err_srs) = (0.0, 0.0);
        for seed in 0..100u64 {
            let s = cube(&x, 2, &pi, seed).unwrap();
            if s.iter().filter(|&&b| b).count() == 20 {
                sizes_ok += 1;
            }
            let ht: f64 = (0..n).filter(|&i| s[i]).map(|i| z[i] / pi[i]).sum();
            err_cube += (ht - total_z).abs();
            // simple random sample of 20 for comparison
            let mut rng = Rng::new(1000 + seed);
            let mut idx: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut idx);
            let ht_srs: f64 = idx[..20].iter().map(|&i| z[i] / pi[i]).sum();
            err_srs += (ht_srs - total_z).abs();
        }
        assert_eq!(sizes_ok, 100, "fixed size in {sizes_ok} of 100 draws");
        assert!(err_cube < 0.5 * err_srs, "cube {err_cube} vs srs {err_srs}");
        // inclusion probabilities are respected on average
        let mut hits = vec![0usize; n];
        for seed in 0..400u64 {
            let s = cube(&x, 2, &pi, seed).unwrap();
            for i in 0..n {
                if s[i] {
                    hits[i] += 1;
                }
            }
        }
        let mean_rate = hits.iter().sum::<usize>() as f64 / (400.0 * n as f64);
        assert!((mean_rate - 0.2).abs() < 0.01);
        assert!(hits.iter().all(|&h| h > 40 && h < 120), "{:?}", hits);
    }

    #[test]
    fn spatial_balance_prefers_spread_samples() {
        let coords = grid(10);
        let pi = vec![0.1; 100];
        // a regular lattice of 10 units vs a clump in one corner
        let mut lattice = vec![false; 100];
        for &i in &[0usize, 3, 6, 9, 30, 33, 36, 39, 60, 63] {
            lattice[i] = true;
        }
        let mut corner = vec![false; 100];
        for &i in &[0usize, 1, 2, 10, 11, 12, 20, 21, 22, 30] {
            corner[i] = true;
        }
        let bl = spatial_balance(&coords, 2, &pi, &lattice).unwrap();
        let bc = spatial_balance(&coords, 2, &pi, &corner).unwrap();
        assert!(bl < bc, "{bl} vs {bc}");
        assert!(spatial_balance(&coords, 2, &pi, &[false; 100]).is_err());
    }
}
