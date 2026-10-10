//! # fieldopt-core
//!
//! Field-survey design under real field cost. The crate provides the
//! computational pieces behind the R package `fieldopt`: travel matrices,
//! routing of field work (exact for small instances, hybrid genetic search
//! otherwise, with service times, stops and capacity limits),
//! spatially balanced sampling with the local pivotal method, balanced
//! sampling with the cube method and the Voronoi measure of spatial balance,
//! Horvitz-Thompson estimation with the local-mean variance estimator, and
//! cost-aware allocation across strata or frames, for one or several study
//! variables, in dual frames and in two stages. It has no dependencies and
//! every randomised step is reproducible from a seed.
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
//! let limits = RouteLimits { max_length: f64::INFINITY, max_stops: 5, capacity: f64::INFINITY };
//! let sol = grasp_routes(&m, 0, &units, limits, GraspOptions::default()).unwrap();
//! assert_eq!(sol.routes.iter().map(|r| r.len()).sum::<usize>(), 12);
//! assert!(sol.total >= sol.lower_bound);
//! ```

/// Version of this crate, for reports.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod allocation;
pub mod balance;
pub mod certify;
pub mod distance;
pub mod dualframe;
pub mod estimation;
pub mod rng;
pub mod routing;
pub mod sampling;
pub mod spatial;
pub(crate) mod spatial_index;
pub mod twostage;
pub mod vrp;

pub use allocation::{
    allocate_for_budget, allocate_for_variance, allocate_multivariate, Allocation, MultiAllocation,
    Stratum,
};
pub use balance::{cube, spatial_balance};
pub use certify::{certify_tour, Certificate};
pub use distance::{euclidean_matrix, haversine_matrix, Matrix};
pub use dualframe::{dual_frame_allocation, Domain, DualFrame, DualFrameAllocation, Target};
pub use estimation::{ht_total, local_mean_variance, srs_variance};
pub use routing::{
    grasp_routes, grasp_routes_with_service, route_duration, route_length, GraspOptions,
    RouteLimits, Solution,
};
pub use sampling::{inclusion_probabilities, local_pivotal};
pub use spatial::{hilbert_order, systematic_replicates};
pub use twostage::{two_stage_for_budget, two_stage_for_variance, TwoStage, TwoStageAllocation};
