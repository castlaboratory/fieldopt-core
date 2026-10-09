//! # fieldopt-core
//!
//! Field-survey design under real field cost. The crate provides the
//! computational pieces behind the R package `fieldopt`: travel matrices,
//! GRASP routing of field work, spatially balanced sampling with the local
//! pivotal method, Horvitz-Thompson estimation with the local-mean variance
//! estimator, and cost-aware allocation across strata or frames. It has no
//! dependencies and every randomised step is reproducible from a seed.
//!
//! ```
//! use fieldopt_core::distance::euclidean_matrix;
//! use fieldopt_core::routing::{grasp_routes, GraspOptions, RouteLimits};
//! use fieldopt_core::sampling::{inclusion_probabilities, local_pivotal};
//!
//! // a 6 x 6 grid of segments plus a depot at the origin
//! let mut x = vec![0.0]; let mut y = vec![0.0];
//! for i in 0..36 { x.push(1.0 + (i % 6) as f64); y.push(1.0 + (i / 6) as f64); }
//! let m = euclidean_matrix(&x, &y).unwrap();
//!
//! // select 12 segments, spatially balanced, with equal probabilities
//! let coords: Vec<f64> = (1..37).flat_map(|i| vec![x[i], y[i]]).collect();
//! let pi = inclusion_probabilities(&vec![1.0; 36], 12).unwrap();
//! let sampled = local_pivotal(&coords, 2, &pi, 1).unwrap();
//! let units: Vec<usize> = (0..36).filter(|&i| sampled[i]).map(|i| i + 1).collect();
//!
//! // route the field work from the depot with at most 5 segments per day
//! let limits = RouteLimits { max_length: f64::INFINITY, max_stops: 5 };
//! let sol = grasp_routes(&m, 0, &units, limits, GraspOptions::default()).unwrap();
//! assert_eq!(sol.routes.iter().map(|r| r.len()).sum::<usize>(), 12);
//! assert!(sol.total >= sol.lower_bound);
//! ```

pub mod allocation;
pub mod distance;
pub mod estimation;
pub mod rng;
pub mod routing;
pub mod sampling;

pub use allocation::{allocate_for_budget, allocate_for_variance, Allocation, Stratum};
pub use distance::{euclidean_matrix, haversine_matrix, Matrix};
pub use estimation::{ht_total, local_mean_variance, srs_variance};
pub use routing::{grasp_routes, route_length, GraspOptions, RouteLimits, Solution};
pub use sampling::{inclusion_probabilities, local_pivotal};
