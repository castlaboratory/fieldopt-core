//! Routing of field work: one team (a travelling-salesman tour from a depot)
//! or several teams or days (vehicle routes with a maximum length and a
//! maximum number of stops per route).
//!
//! The solver is GRASP (Feo and Resende, 1995): each iteration builds a tour
//! with a randomised greedy construction (restricted candidate list of the
//! nearest unvisited units), improves it with 2-opt and Or-opt local search,
//! and keeps the best. For several routes the improved giant tour is split
//! optimally into routes that respect the limits by the dynamic programme of
//! Prins (2004), and routes are improved again individually. Heuristics do not
//! certify optimality; [`Solution::gap_to_bound`] reports the gap to a cheap
//! lower bound (the one-tree bound is not computed; the bound used is the sum
//! of each unit's two shortest edges divided by two, a classical TSP bound).

use crate::distance::Matrix;
use crate::rng::Rng;

/// Limits of each route.
#[derive(Clone, Copy, Debug)]
pub struct RouteLimits {
    /// Maximum travel cost of one route (including the return to the depot);
    /// `f64::INFINITY` for none.
    pub max_length: f64,
    /// Maximum number of units in one route; `usize::MAX` for none.
    pub max_stops: usize,
}

impl Default for RouteLimits {
    fn default() -> Self {
        RouteLimits {
            max_length: f64::INFINITY,
            max_stops: usize::MAX,
        }
    }
}

/// Options of the GRASP solver.
#[derive(Clone, Copy, Debug)]
pub struct GraspOptions {
    /// Number of GRASP iterations.
    pub iterations: usize,
    /// Greediness of the construction in `[0, 1]`: 0 is pure greedy, 1 is
    /// uniformly random among all candidates.
    pub alpha: f64,
    /// Seed of the random generator.
    pub seed: u64,
}

impl Default for GraspOptions {
    fn default() -> Self {
        GraspOptions {
            iterations: 200,
            alpha: 0.3,
            seed: 1,
        }
    }
}

/// A routing solution.
#[derive(Clone, Debug)]
pub struct Solution {
    /// The routes, each a sequence of unit indices (depot excluded).
    pub routes: Vec<Vec<usize>>,
    /// Travel cost of each route, depot to depot.
    pub lengths: Vec<f64>,
    /// Total travel cost.
    pub total: f64,
    /// A lower bound on the total cost of any solution visiting the same units
    /// with one route (half the sum of the two cheapest edges at each node,
    /// depot included).
    pub lower_bound: f64,
    /// Iteration at which the best solution was found.
    pub best_iteration: usize,
}

impl Solution {
    /// Relative gap of the total cost to the lower bound.
    pub fn gap_to_bound(&self) -> f64 {
        if self.lower_bound <= 0.0 {
            return f64::NAN;
        }
        (self.total - self.lower_bound) / self.lower_bound
    }
}

fn validate(m: &Matrix, depot: usize, units: &[usize]) -> Result<(), String> {
    let n = m.len();
    if depot >= n {
        return Err(format!("depot index {depot} is out of range for {n} units"));
    }
    if units.is_empty() {
        return Err("no units to route".into());
    }
    let mut seen = vec![false; n];
    for &u in units {
        if u >= n {
            return Err(format!("unit index {u} is out of range for {n} units"));
        }
        if u == depot {
            return Err("the depot cannot be one of the units to visit".into());
        }
        if seen[u] {
            return Err(format!("unit {u} appears twice"));
        }
        seen[u] = true;
    }
    Ok(())
}

/// Cost of a route: depot -> units in order -> depot.
pub fn route_length(m: &Matrix, depot: usize, route: &[usize]) -> f64 {
    if route.is_empty() {
        return 0.0;
    }
    let mut total = m.get(depot, route[0]);
    for w in route.windows(2) {
        total += m.get(w[0], w[1]);
    }
    total + m.get(route[route.len() - 1], depot)
}

fn two_edge_bound(m: &Matrix, depot: usize, units: &[usize]) -> f64 {
    let mut nodes: Vec<usize> = Vec::with_capacity(units.len() + 1);
    nodes.push(depot);
    nodes.extend_from_slice(units);
    if nodes.len() < 3 {
        return if nodes.len() == 2 {
            2.0 * m.get(nodes[0], nodes[1])
        } else {
            0.0
        };
    }
    let mut sum = 0.0;
    for &i in &nodes {
        let mut best = [f64::INFINITY; 2];
        for &j in &nodes {
            if i == j {
                continue;
            }
            let d = m.get(i, j);
            if d < best[0] {
                best[1] = best[0];
                best[0] = d;
            } else if d < best[1] {
                best[1] = d;
            }
        }
        sum += best[0] + best[1];
    }
    sum / 2.0
}

/// Randomised nearest-neighbour construction from the depot.
fn construct(m: &Matrix, depot: usize, units: &[usize], alpha: f64, rng: &mut Rng) -> Vec<usize> {
    let mut remaining: Vec<usize> = units.to_vec();
    let mut tour = Vec::with_capacity(units.len());
    let mut current = depot;
    while !remaining.is_empty() {
        let costs: Vec<f64> = remaining.iter().map(|&u| m.get(current, u)).collect();
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &c in &costs {
            lo = lo.min(c);
            hi = hi.max(c);
        }
        let threshold = lo + alpha * (hi - lo);
        let candidates: Vec<usize> = (0..remaining.len())
            .filter(|&k| costs[k] <= threshold)
            .collect();
        let pick = candidates[rng.below(candidates.len())];
        current = remaining.swap_remove(pick);
        tour.push(current);
    }
    tour
}

/// 2-opt improvement of an open tour anchored at the depot on both ends.
fn two_opt(m: &Matrix, depot: usize, tour: &mut [usize]) -> bool {
    let n = tour.len();
    if n < 3 {
        return false;
    }
    let node = |tour: &[usize], k: isize| -> usize {
        if k < 0 || k as usize >= tour.len() {
            depot
        } else {
            tour[k as usize]
        }
    };
    let mut improved_any = false;
    let mut improved = true;
    while improved {
        improved = false;
        for i in 0..n - 1 {
            for j in (i + 1)..n {
                let a = node(tour, i as isize - 1);
                let b = tour[i];
                let c = tour[j];
                let d = node(tour, j as isize + 1);
                let delta = m.get(a, c) + m.get(b, d) - m.get(a, b) - m.get(c, d);
                if delta < -1e-12 {
                    tour[i..=j].reverse();
                    improved = true;
                    improved_any = true;
                }
            }
        }
    }
    improved_any
}

/// Or-opt: move segments of 1 to 3 consecutive units elsewhere in the tour.
fn or_opt(m: &Matrix, depot: usize, tour: &mut Vec<usize>) -> bool {
    let mut improved_any = false;
    let mut improved = true;
    while improved {
        improved = false;
        let n = tour.len();
        'outer: for seg in 1..=3.min(n.saturating_sub(1)) {
            for i in 0..=(n - seg) {
                let prev = if i == 0 { depot } else { tour[i - 1] };
                let next = if i + seg >= n { depot } else { tour[i + seg] };
                let first = tour[i];
                let last = tour[i + seg - 1];
                let removal_gain = m.get(prev, first) + m.get(last, next) - m.get(prev, next);
                // candidate insertion positions in the tour without the segment
                let rest: Vec<usize> = tour
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k < i || *k >= i + seg)
                    .map(|(_, &u)| u)
                    .collect();
                for p in 0..=rest.len() {
                    let a = if p == 0 { depot } else { rest[p - 1] };
                    let b = if p >= rest.len() { depot } else { rest[p] };
                    for &reversed in &[false, true] {
                        let (f, l) = if reversed {
                            (last, first)
                        } else {
                            (first, last)
                        };
                        let insert_cost = m.get(a, f) + m.get(l, b) - m.get(a, b);
                        if insert_cost < removal_gain - 1e-12 {
                            let mut segment: Vec<usize> = tour[i..i + seg].to_vec();
                            if reversed {
                                segment.reverse();
                            }
                            let mut new_tour = rest.clone();
                            for (k, u) in segment.into_iter().enumerate() {
                                new_tour.insert(p + k, u);
                            }
                            *tour = new_tour;
                            improved = true;
                            improved_any = true;
                            continue 'outer;
                        }
                    }
                }
            }
        }
    }
    improved_any
}

fn local_search(m: &Matrix, depot: usize, tour: &mut Vec<usize>) {
    loop {
        let a = two_opt(m, depot, tour);
        let b = or_opt(m, depot, tour);
        if !a && !b {
            break;
        }
    }
}

/// Optimal split of a giant tour into routes respecting the limits (Prins 2004).
/// Returns `None` if some unit cannot be served within the limits alone.
fn split(
    m: &Matrix,
    depot: usize,
    tour: &[usize],
    limits: &RouteLimits,
) -> Option<Vec<Vec<usize>>> {
    let n = tour.len();
    let mut cost = vec![f64::INFINITY; n + 1];
    let mut pred = vec![usize::MAX; n + 1];
    cost[0] = 0.0;
    for i in 0..n {
        if !cost[i].is_finite() {
            continue;
        }
        let mut len = 0.0;
        for j in i..n {
            if j - i + 1 > limits.max_stops {
                break;
            }
            if j == i {
                len = m.get(depot, tour[i]);
            } else {
                len += m.get(tour[j - 1], tour[j]);
            }
            let route_cost = len + m.get(tour[j], depot);
            if route_cost > limits.max_length {
                break;
            }
            if cost[i] + route_cost < cost[j + 1] {
                cost[j + 1] = cost[i] + route_cost;
                pred[j + 1] = i;
            }
        }
    }
    if !cost[n].is_finite() {
        return None;
    }
    let mut routes = Vec::new();
    let mut j = n;
    while j > 0 {
        let i = pred[j];
        routes.push(tour[i..j].to_vec());
        j = i;
    }
    routes.reverse();
    Some(routes)
}

/// Solve the routing of `units` from `depot` under `limits` with GRASP.
///
/// With default limits the result is a single tour (TSP). Returns an error
/// for invalid indices or when a unit cannot be served within the limits.
pub fn grasp_routes(
    m: &Matrix,
    depot: usize,
    units: &[usize],
    limits: RouteLimits,
    options: GraspOptions,
) -> Result<Solution, String> {
    validate(m, depot, units)?;
    if !(0.0..=1.0).contains(&options.alpha) {
        return Err("alpha must be in [0, 1]".into());
    }
    if options.iterations == 0 {
        return Err("iterations must be at least 1".into());
    }
    if limits.max_stops == 0 || limits.max_length.is_nan() || limits.max_length <= 0.0 {
        return Err("route limits must allow at least one stop".into());
    }
    let mut rng = Rng::new(options.seed);
    let lower_bound = two_edge_bound(m, depot, units);
    let mut best: Option<Solution> = None;
    for it in 0..options.iterations {
        let mut tour = construct(m, depot, units, options.alpha, &mut rng);
        local_search(m, depot, &mut tour);
        let mut routes = match split(m, depot, &tour, &limits) {
            Some(r) => r,
            None => return Err("some unit cannot be served within the route limits".into()),
        };
        if routes.len() > 1 {
            for r in routes.iter_mut() {
                local_search(m, depot, r);
            }
        }
        let lengths: Vec<f64> = routes.iter().map(|r| route_length(m, depot, r)).collect();
        let total: f64 = lengths.iter().sum();
        if best.as_ref().map_or(true, |b| total < b.total - 1e-12) {
            best = Some(Solution {
                routes,
                lengths,
                total,
                lower_bound,
                best_iteration: it + 1,
            });
        }
    }
    Ok(best.expect("at least one iteration"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::euclidean_matrix;

    fn square_points() -> Matrix {
        // depot at the centre, 8 points on a square
        let x = [0.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0];
        let y = [0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0];
        euclidean_matrix(&x, &y).unwrap()
    }

    #[test]
    fn tsp_on_a_square_is_optimal() {
        let m = square_points();
        let units: Vec<usize> = (1..9).collect();
        let s = grasp_routes(
            &m,
            0,
            &units,
            RouteLimits::default(),
            GraspOptions {
                iterations: 50,
                alpha: 0.3,
                seed: 3,
            },
        )
        .unwrap();
        assert_eq!(s.routes.len(), 1);
        // seven unit edges of the square plus the depot legs to the two ends of the
        // open path, which are adjacent: one midpoint (1) and one corner (sqrt 2)
        let optimum = 7.0 + 1.0 + 2f64.sqrt();
        assert!((s.total - optimum).abs() < 1e-9, "total {}", s.total);
        assert!(s.lower_bound <= s.total + 1e-9);
        assert_eq!(route_length(&m, 0, &s.routes[0]), s.total);
    }

    #[test]
    fn split_respects_limits() {
        let m = square_points();
        let units: Vec<usize> = (1..9).collect();
        let limits = RouteLimits {
            max_length: f64::INFINITY,
            max_stops: 3,
        };
        let s = grasp_routes(&m, 0, &units, limits, GraspOptions::default()).unwrap();
        assert!(s.routes.len() >= 3);
        assert!(s.routes.iter().all(|r| r.len() <= 3));
        let mut all: Vec<usize> = s.routes.concat();
        all.sort();
        assert_eq!(all, units);
        // a corner alone needs 2 * sqrt(2) = 2.83, so 3.0 is feasible and 2.5 is not
        let limits = RouteLimits {
            max_length: 3.0,
            max_stops: usize::MAX,
        };
        let s = grasp_routes(&m, 0, &units, limits, GraspOptions::default()).unwrap();
        assert!(s.lengths.iter().all(|&l| l <= 3.0 + 1e-9));
        let limits = RouteLimits {
            max_length: 2.5,
            max_stops: usize::MAX,
        };
        assert!(grasp_routes(&m, 0, &units, limits, GraspOptions::default()).is_err());
        let limits = RouteLimits {
            max_length: 1.0,
            max_stops: usize::MAX,
        };
        assert!(grasp_routes(&m, 0, &units, limits, GraspOptions::default()).is_err());
    }

    #[test]
    fn invalid_input_is_refused() {
        let m = square_points();
        assert!(grasp_routes(
            &m,
            0,
            &[0, 1],
            RouteLimits::default(),
            GraspOptions::default()
        )
        .is_err());
        assert!(grasp_routes(
            &m,
            0,
            &[1, 1],
            RouteLimits::default(),
            GraspOptions::default()
        )
        .is_err());
        assert!(
            grasp_routes(&m, 9, &[1], RouteLimits::default(), GraspOptions::default()).is_err()
        );
        assert!(grasp_routes(&m, 0, &[], RouteLimits::default(), GraspOptions::default()).is_err());
    }

    #[test]
    fn deterministic_for_a_seed() {
        let m = square_points();
        let units: Vec<usize> = (1..9).collect();
        let o = GraspOptions {
            iterations: 20,
            alpha: 0.5,
            seed: 11,
        };
        let a = grasp_routes(&m, 0, &units, RouteLimits::default(), o).unwrap();
        let b = grasp_routes(&m, 0, &units, RouteLimits::default(), o).unwrap();
        assert_eq!(a.routes, b.routes);
    }
}
