//! Routing of field work: one team (a tour from a depot) or several teams
//! or days (routes with a maximum duration, a maximum number of stops and a
//! capacity).
//!
//! [`solve_routes`] solves instances of up to [`crate::vrp::EXACT_MAX_UNITS`]
//! units exactly by dynamic programming over subsets (Held and Karp 1962 for
//! the tours, then an optimal partition into feasible routes). Larger
//! instances go to a hybrid genetic search (Vidal 2022): a population of
//! giant tours, order crossover, the optimal split of Prins (2004) under the
//! limits, and a local search with granular neighbourhoods (Toth and Vigo
//! 2003) made of 2-opt, Or-opt, relocate, swap and 2-opt* moves; the
//! population is managed with a biased fitness that rewards diversity. The
//! randomised nearest-neighbour construction of GRASP (Feo and Resende 1995)
//! seeds the population.
//!
//! [`Solution::lower_bound`] is the optimum when the instance was solved
//! exactly; otherwise, for a symmetric matrix and limits that allow a single
//! tour, the Held-Karp 1-tree bound (Held and Karp 1971; typically within a
//! few percent of the optimum, and valid as a bound on the routing problem
//! under the triangle inequality); otherwise half the sum, over the depot
//! and the units, of the cheapest incoming and outgoing edge at each node,
//! valid for any number of routes and asymmetric matrices but loose.
//!
//! Travel matrices may be asymmetric (road networks with one-way streets or
//! different durations by direction): every move is priced in the direction
//! travelled.

use crate::distance::Matrix;
use crate::rng::Rng;
use crate::vrp::{self, Instance, EXACT_MAX_UNITS};

/// Limits of each route.
#[derive(Clone, Copy, Debug)]
pub struct RouteLimits {
    /// Maximum duration of one route: travel including the return to the
    /// depot plus the service time of its units; `f64::INFINITY` for none.
    pub max_length: f64,
    /// Maximum number of units in one route; `usize::MAX` for none.
    pub max_stops: usize,
    /// Maximum total demand of one route; `f64::INFINITY` for none.
    pub capacity: f64,
}

impl Default for RouteLimits {
    fn default() -> Self {
        RouteLimits {
            max_length: f64::INFINITY,
            max_stops: usize::MAX,
            capacity: f64::INFINITY,
        }
    }
}

/// Options of the solver.
#[derive(Clone, Copy, Debug)]
pub struct GraspOptions {
    /// Number of offspring of the genetic search (ignored when the instance
    /// is solved exactly, or when `time_limit` is finite).
    pub iterations: usize,
    /// Wall-clock limit of the genetic search in seconds; `f64::INFINITY`
    /// to run `iterations` offspring instead.
    pub time_limit: f64,
    /// Seconds allowed to the branch-and-bound that tries to prove a single
    /// tour optimal (symmetric matrices, one depot, limits that allow one
    /// route); 0 switches it off.
    pub certify_time: f64,
    /// Greediness of the constructions that seed the population, in
    /// `[0, 1]`: 0 is pure greedy, 1 is uniformly random among all
    /// candidates.
    pub alpha: f64,
    /// Seed of the random generator.
    pub seed: u64,
}

impl Default for GraspOptions {
    fn default() -> Self {
        GraspOptions {
            iterations: 200,
            time_limit: f64::INFINITY,
            certify_time: 1.0,
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
    /// The depot (node index) each route leaves from and returns to.
    pub route_depots: Vec<usize>,
    /// Travel cost of each route, depot to depot.
    pub lengths: Vec<f64>,
    /// Duration of each route: travel plus the service time of its units
    /// (equal to `lengths` when no service time is given).
    pub durations: Vec<f64>,
    /// Total demand of each route (zero without demands).
    pub loads: Vec<f64>,
    /// Total travel cost.
    pub total: f64,
    /// A lower bound on the total cost of any solution visiting the same units
    /// under the same limits (see the module documentation).
    pub lower_bound: f64,
    /// Whether the solution is proven optimal (solved exactly, or the bound
    /// meets the cost).
    pub optimal: bool,
    /// Iteration at which the best solution was found (0 when exact).
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
///
/// # Panics
///
/// Panics when `depot` or an element of `route` is not an index of `m`.
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

/// Whether `m(i, j) == m(j, i)` for every pair.
fn is_symmetric(m: &Matrix) -> bool {
    let n = m.len();
    (0..n).all(|i| (i + 1..n).all(|j| (m.get(i, j) - m.get(j, i)).abs() <= 1e-12))
}

/// Half the sum of the cheapest incoming and cheapest outgoing edge at each
/// node: every node of a solution has one incoming and one outgoing edge
/// (the depot at least one of each), so this bounds any number of routes.
fn in_out_bound(m: &Matrix, depot: usize, units: &[usize]) -> f64 {
    let mut nodes: Vec<usize> = Vec::with_capacity(units.len() + 1);
    nodes.push(depot);
    nodes.extend_from_slice(units);
    let mut sum = 0.0;
    for &i in &nodes {
        let (mut min_out, mut min_in) = (f64::INFINITY, f64::INFINITY);
        for &j in &nodes {
            if i != j {
                min_out = min_out.min(m.get(i, j));
                min_in = min_in.min(m.get(j, i));
            }
        }
        sum += min_out + min_in;
    }
    sum / 2.0
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
pub(crate) fn construct(
    m: &Matrix,
    depot: usize,
    units: &[usize],
    alpha: f64,
    rng: &mut Rng,
) -> Vec<usize> {
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

/// Cumulative cost of the tour's internal edges forwards (`fwd[k]`: edges
/// before position `k`) and backwards (`bwd[k]`: the same edges reversed).
fn prefix_costs(m: &Matrix, tour: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let n = tour.len();
    let mut fwd = vec![0.0; n];
    let mut bwd = vec![0.0; n];
    for k in 1..n {
        fwd[k] = fwd[k - 1] + m.get(tour[k - 1], tour[k]);
        bwd[k] = bwd[k - 1] + m.get(tour[k], tour[k - 1]);
    }
    (fwd, bwd)
}

/// 2-opt improvement of an open tour anchored at the depot on both ends.
/// Reversing `tour[i..=j]` also reverses its internal edges, which matters
/// on asymmetric matrices: that cost is included through prefix sums.
pub(crate) fn two_opt(m: &Matrix, depot: usize, tour: &mut [usize], symmetric: bool) -> bool {
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
        let (fwd, bwd) = if symmetric {
            (Vec::new(), Vec::new())
        } else {
            prefix_costs(m, tour)
        };
        'scan: for i in 0..n - 1 {
            for j in (i + 1)..n {
                let a = node(tour, i as isize - 1);
                let b = tour[i];
                let c = tour[j];
                let d = node(tour, j as isize + 1);
                let mut delta = m.get(a, c) + m.get(b, d) - m.get(a, b) - m.get(c, d);
                if !symmetric {
                    delta += (bwd[j] - bwd[i]) - (fwd[j] - fwd[i]);
                }
                if delta < -1e-12 {
                    tour[i..=j].reverse();
                    improved = true;
                    improved_any = true;
                    if !symmetric {
                        break 'scan; // prefix sums are stale: rescan
                    }
                }
            }
        }
    }
    improved_any
}

/// Or-opt: move segments of 1 to 3 consecutive units elsewhere in the tour,
/// possibly reversed. On asymmetric matrices a reversed segment also changes
/// the cost of its internal edges, which is priced in.
pub(crate) fn or_opt(m: &Matrix, depot: usize, tour: &mut Vec<usize>, symmetric: bool) -> bool {
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
                        let mut insert_cost = m.get(a, f) + m.get(l, b) - m.get(a, b);
                        if reversed && !symmetric {
                            for w in tour[i..i + seg].windows(2) {
                                insert_cost += m.get(w[1], w[0]) - m.get(w[0], w[1]);
                            }
                        }
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

/// Lower bound used in the solution (see the module documentation).
fn lower_bound(inst: &Instance, units: &[usize], upper: f64) -> (f64, bool) {
    let general = in_out_bound(inst.m, inst.depot, units);
    if inst.symmetric && inst.single_route_allowed(units) {
        let (hk, _) = vrp::held_karp_bound(inst.m, inst.depot, units, upper);
        let b = general
            .max(two_edge_bound(inst.m, inst.depot, units))
            .max(hk);
        (b.min(upper), upper - b <= 1e-9 * upper.max(1.0))
    } else {
        (general.min(upper), false)
    }
}

/// Solve the routing of `units` from `depot` under `limits` (see the module
/// documentation). With default limits the result is a single tour.
pub fn grasp_routes(
    m: &Matrix,
    depot: usize,
    units: &[usize],
    limits: RouteLimits,
    options: GraspOptions,
) -> Result<Solution, String> {
    solve_routes(m, depot, units, &[], &[], limits, options)
}

/// Duration of a route: its travel plus the service time of its units.
pub fn route_duration(m: &Matrix, depot: usize, route: &[usize], service: &[f64]) -> f64 {
    route_length(m, depot, route)
        + route
            .iter()
            .map(|&u| service.get(u).copied().unwrap_or(0.0))
            .sum::<f64>()
}

/// As [`grasp_routes`], with a service time per node of the matrix.
pub fn grasp_routes_with_service(
    m: &Matrix,
    depot: usize,
    units: &[usize],
    service: &[f64],
    limits: RouteLimits,
    options: GraspOptions,
) -> Result<Solution, String> {
    solve_routes(m, depot, units, service, &[], limits, options)
}

/// Depots of a routing problem: node indices, the number of routes each may
/// send (`usize::MAX` for no limit) and the cost of opening one route, in
/// travel units (it enters the objective as `fixed_cost * routes`).
#[derive(Clone, Debug)]
pub struct Fleet {
    pub depots: Vec<usize>,
    pub routes_available: Vec<usize>,
    pub fixed_cost: f64,
}

impl Fleet {
    /// One depot, unlimited routes, no fixed cost.
    pub fn single(depot: usize) -> Self {
        Fleet {
            depots: vec![depot],
            routes_available: vec![usize::MAX],
            fixed_cost: 0.0,
        }
    }
}

/// Solve the routing of `units` from `depot`: `service` is the time spent at
/// each node and `demand` the load each node adds to its route (both per
/// node of the matrix, empty for none); `max_length` limits the duration of
/// a route (travel plus service), `max_stops` its number of units and
/// `capacity` its load. The objective is the total travel. Returns an error
/// for invalid indices or when a unit cannot be served alone within the
/// limits.
pub fn solve_routes(
    m: &Matrix,
    depot: usize,
    units: &[usize],
    service: &[f64],
    demand: &[f64],
    limits: RouteLimits,
    options: GraspOptions,
) -> Result<Solution, String> {
    solve_fleet(
        m,
        &Fleet::single(depot),
        units,
        service,
        demand,
        limits,
        options,
    )
}

/// As [`solve_routes`], with several depots: every route leaves from and
/// returns to one depot, chosen by the solver; each depot sends at most its
/// number of routes available; the objective is the total travel plus the
/// fixed cost of the routes opened. Instances of up to
/// [`EXACT_MAX_UNITS`] units are solved exactly when no depot's route limit
/// can bind.
pub fn solve_fleet(
    m: &Matrix,
    fleet: &Fleet,
    units: &[usize],
    service: &[f64],
    demand: &[f64],
    limits: RouteLimits,
    options: GraspOptions,
) -> Result<Solution, String> {
    if fleet.depots.is_empty() || fleet.depots.len() != fleet.routes_available.len() {
        return Err(
            "the fleet needs at least one depot, with a number of routes available for each".into(),
        );
    }
    if !fleet.fixed_cost.is_finite() || fleet.fixed_cost < 0.0 {
        return Err("the fixed cost per route must be finite and non-negative".into());
    }
    for &d in &fleet.depots {
        validate(m, d, units)?;
    }
    {
        let mut seen = vec![false; m.len()];
        for &d in &fleet.depots {
            if seen[d] {
                return Err("depots must be distinct".into());
            }
            seen[d] = true;
        }
    }
    if fleet.routes_available.contains(&0) {
        return Err("every depot needs at least one route available".into());
    }
    if !(0.0..=1.0).contains(&options.alpha) {
        return Err("alpha must be in [0, 1]".into());
    }
    if options.iterations == 0 && !options.time_limit.is_finite() {
        return Err("iterations must be at least 1".into());
    }
    if options.time_limit.is_nan() || options.time_limit <= 0.0 {
        return Err("time_limit must be positive".into());
    }
    if limits.max_stops == 0
        || limits.max_length.is_nan()
        || limits.max_length <= 0.0
        || limits.capacity.is_nan()
        || limits.capacity < 0.0
    {
        return Err("route limits must allow at least one stop".into());
    }
    let n = m.len();
    let check = |v: &[f64], what: &str| -> Result<Vec<f64>, String> {
        if !v.is_empty() && v.len() != n {
            return Err(format!("{what} must have one entry per node of the matrix"));
        }
        if v.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(format!("{what} must be finite and non-negative"));
        }
        Ok(if v.is_empty() {
            vec![0.0; n]
        } else {
            v.to_vec()
        })
    };
    let service = check(service, "service times")?;
    let demand = check(demand, "demands")?;
    let symmetric = is_symmetric(m);
    let inst = Instance::with_depots(
        m,
        &fleet.depots,
        &fleet.routes_available,
        fleet.fixed_cost,
        units,
        service.clone(),
        demand.clone(),
        limits,
        symmetric,
    );
    // every unit must be servable alone from some depot
    for &u in units {
        let ok = fleet.depots.iter().any(|&d| {
            let travel = m.get(d, u) + m.get(u, d);
            travel + service[u] <= limits.max_length + 1e-9 && demand[u] <= limits.capacity + 1e-9
        });
        if !ok {
            return Err("some unit cannot be served within the route limits".into());
        }
    }
    let total_routes_available: f64 = fleet
        .routes_available
        .iter()
        .map(|&k| {
            if k == usize::MAX {
                f64::INFINITY
            } else {
                k as f64
            }
        })
        .sum();
    let fleet_binds = total_routes_available.is_finite();
    let mut rng = Rng::new(options.seed);
    let (routes, depot_idx, objective, best_iteration, optimal) = if units.len() <= EXACT_MAX_UNITS
        && !fleet_binds
    {
        let (r, d, c) = vrp::exact(&inst, units)
            .ok_or_else(|| "some unit cannot be served within the route limits".to_string())?;
        (r, d, c, 0, true)
    } else {
        let (r, d, c, it) = vrp::hgs(
            &inst,
            units,
            options.iterations,
            options.alpha,
            options.time_limit,
            &mut rng,
        )
        .ok_or_else(|| {
            "no solution respects the route limits and the routes available per depot".to_string()
        })?;
        (r, d, c, it, false)
    };
    let mut routes = routes;
    let route_depots: Vec<usize> = depot_idx.iter().map(|&k| fleet.depots[k]).collect();
    let mut lengths: Vec<f64> = routes
        .iter()
        .zip(&route_depots)
        .map(|(r, &d)| route_length(m, d, r))
        .collect();
    let mut total: f64 = lengths.iter().sum();
    let (mut lower_bound, mut proven) = if optimal {
        (total, true)
    } else if fleet.depots.len() == 1 {
        lower_bound(&inst, units, total)
    } else {
        (
            in_out_bound_multi(m, &fleet.depots, units).min(total),
            false,
        )
    };
    // branch-and-bound certificate for a single tour
    if !proven
        && !optimal
        && options.certify_time > 0.0
        && symmetric
        && fleet.depots.len() == 1
        && routes.len() == 1
        && inst.single_route_allowed(units)
    {
        let cert = crate::certify::certify_tour(
            m,
            fleet.depots[0],
            units,
            &routes[0],
            total,
            usize::MAX,
            options.certify_time,
            1e-7,
        );
        if let Some(t) = cert.tour {
            if cert.upper_bound < total - 1e-9 {
                routes = vec![t];
                lengths = vec![cert.upper_bound];
                total = cert.upper_bound;
            }
        }
        lower_bound = lower_bound.max(cert.lower_bound).min(total);
        proven = cert.optimal;
    }
    let durations: Vec<f64> = routes
        .iter()
        .zip(&route_depots)
        .map(|(r, &d)| route_duration(m, d, r, &service))
        .collect();
    let loads: Vec<f64> = routes
        .iter()
        .map(|r| r.iter().map(|&u| demand[u]).sum())
        .collect();
    let _ = objective;
    Ok(Solution {
        routes,
        route_depots,
        lengths,
        durations,
        loads,
        total,
        lower_bound,
        optimal: optimal || proven,
        best_iteration,
    })
}

/// Half the sum over the units of the cheapest incoming and outgoing edge
/// (from units or depots): a bound on the travel of any multi-depot solution.
fn in_out_bound_multi(m: &Matrix, depots: &[usize], units: &[usize]) -> f64 {
    let nodes: Vec<usize> = depots.iter().chain(units.iter()).copied().collect();
    let mut sum = 0.0;
    for &i in units {
        let (mut min_out, mut min_in) = (f64::INFINITY, f64::INFINITY);
        for &j in &nodes {
            if i != j {
                min_out = min_out.min(m.get(i, j));
                min_in = min_in.min(m.get(j, i));
            }
        }
        sum += min_out + min_in;
    }
    sum / 2.0
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
                time_limit: f64::INFINITY,
                certify_time: 1.0,
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
            capacity: f64::INFINITY,
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
            capacity: f64::INFINITY,
        };
        let s = grasp_routes(&m, 0, &units, limits, GraspOptions::default()).unwrap();
        assert!(s.lengths.iter().all(|&l| l <= 3.0 + 1e-9));
        let limits = RouteLimits {
            max_length: 2.5,
            max_stops: usize::MAX,
            capacity: f64::INFINITY,
        };
        assert!(grasp_routes(&m, 0, &units, limits, GraspOptions::default()).is_err());
        let limits = RouteLimits {
            max_length: 1.0,
            max_stops: usize::MAX,
            capacity: f64::INFINITY,
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
    fn asymmetric_matrices_terminate_and_respect_the_bound() {
        // random asymmetric costs: the old symmetric deltas cycled forever here
        let n = 9;
        let mut rng = Rng::new(0);
        let data: Vec<f64> = (0..n * n)
            .map(|k| {
                if k % (n + 1) == 0 {
                    0.0
                } else {
                    1.0 + 9.0 * rng.next_f64()
                }
            })
            .collect();
        let m = Matrix::from_vec(n, data).unwrap();
        let units: Vec<usize> = (1..n).collect();
        let o = GraspOptions {
            iterations: 30,
            time_limit: f64::INFINITY,
            certify_time: 1.0,
            alpha: 0.3,
            seed: 1,
        };
        let s = grasp_routes(&m, 0, &units, RouteLimits::default(), o).unwrap();
        assert!(s.total >= s.lower_bound - 1e-9);
        assert!((route_length(&m, 0, &s.routes[0]) - s.total).abs() < 1e-9);
        // brute force over all 8! tours for the optimum
        fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
            if items.len() <= 1 {
                return vec![items.to_vec()];
            }
            let mut out = Vec::new();
            for (k, &x) in items.iter().enumerate() {
                let mut rest = items.to_vec();
                rest.remove(k);
                for mut p in permutations(&rest) {
                    p.insert(0, x);
                    out.push(p);
                }
            }
            out
        }
        let opt = permutations(&units)
            .iter()
            .map(|p| route_length(&m, 0, p))
            .fold(f64::INFINITY, f64::min);
        assert!(
            opt >= s.lower_bound - 1e-9,
            "bound {} above optimum {opt}",
            s.lower_bound
        );
        assert!(s.total <= opt * 1.15, "grasp {} vs optimum {opt}", s.total);
        // a directed 3-cycle of cost 3 with expensive reverse edges
        let data = vec![0.0, 1.0, 100.0, 100.0, 0.0, 1.0, 1.0, 100.0, 0.0];
        let m = Matrix::from_vec(3, data).unwrap();
        let s = grasp_routes(&m, 0, &[1, 2], RouteLimits::default(), o).unwrap();
        assert!((s.total - 3.0).abs() < 1e-9);
        assert!(s.lower_bound <= 3.0 + 1e-9);
    }

    #[test]
    fn bound_is_valid_with_several_routes_without_triangle_inequality() {
        // depot legs cost 0, unit-unit legs 100: one-stop routes cost 0 in total
        let n = 4;
        let data: Vec<f64> = (0..n * n)
            .map(|k| {
                let (i, j) = (k / n, k % n);
                if i == j || i == 0 || j == 0 {
                    0.0
                } else {
                    100.0
                }
            })
            .collect();
        let m = Matrix::from_vec(n, data).unwrap();
        let limits = RouteLimits {
            max_length: f64::INFINITY,
            max_stops: 1,
            capacity: f64::INFINITY,
        };
        let s = grasp_routes(&m, 0, &[1, 2, 3], limits, GraspOptions::default()).unwrap();
        assert!(s.total.abs() < 1e-12);
        assert!(s.lower_bound <= s.total + 1e-12);
    }

    #[test]
    fn service_time_counts_towards_the_route_limit() {
        let m = square_points();
        let units: Vec<usize> = (1..9).collect();
        let service = vec![0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let limits = RouteLimits {
            max_length: 5.0,
            max_stops: usize::MAX,
            capacity: f64::INFINITY,
        };
        let o = GraspOptions {
            iterations: 20,
            time_limit: f64::INFINITY,
            certify_time: 1.0,
            alpha: 0.3,
            seed: 1,
        };
        let no_service = grasp_routes(&m, 0, &units, limits, o).unwrap();
        let with = grasp_routes_with_service(&m, 0, &units, &service, limits, o).unwrap();
        assert!(with.routes.len() > no_service.routes.len());
        assert!(with.durations.iter().all(|&d| d <= 5.0 + 1e-9));
        for (k, r) in with.routes.iter().enumerate() {
            assert!((with.durations[k] - with.lengths[k] - r.len() as f64).abs() < 1e-9);
        }
        // a unit whose service alone exceeds the limit cannot be served
        let mut heavy = service.clone();
        heavy[3] = 10.0;
        assert!(grasp_routes_with_service(&m, 0, &units, &heavy, limits, o).is_err());
        assert!(grasp_routes_with_service(&m, 0, &units, &[1.0], limits, o).is_err());
    }

    #[test]
    fn deterministic_for_a_seed() {
        let m = square_points();
        let units: Vec<usize> = (1..9).collect();
        let o = GraspOptions {
            iterations: 20,
            time_limit: f64::INFINITY,
            certify_time: 1.0,
            alpha: 0.5,
            seed: 11,
        };
        let a = grasp_routes(&m, 0, &units, RouteLimits::default(), o).unwrap();
        let b = grasp_routes(&m, 0, &units, RouteLimits::default(), o).unwrap();
        assert_eq!(a.routes, b.routes);
    }
}
