//! Timing of the local pivotal method and the spatial balance on large frames.
//!
//! ```text
//! cargo run --release --example lpm_scale -- 100000
//! ```
use fieldopt_core::balance::spatial_balance;
use fieldopt_core::rng::Rng;
use fieldopt_core::sampling::{inclusion_probabilities, local_pivotal};
use std::time::Instant;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);
    let mut rng = Rng::new(1);
    let coords: Vec<f64> = (0..2 * n).map(|_| rng.next_f64() * 1000.0).collect();
    let pi = inclusion_probabilities(&vec![1.0; n], n / 100).unwrap();
    let t = Instant::now();
    let s = local_pivotal(&coords, 2, &pi, 1).unwrap();
    let k = s.iter().filter(|&&b| b).count();
    println!(
        "lpm: n = {n}, sample {k}, {:.2} s",
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let b = spatial_balance(&coords, 2, &pi, &s).unwrap();
    println!(
        "spatial balance {b:.3} in {:.2} s",
        t.elapsed().as_secs_f64()
    );
}
