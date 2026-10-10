//! The routing solver behind [`crate::routing::solve_routes`]: a hybrid
//! genetic search in the spirit of Vidal (2022) with an optimal split of
//! giant tours (Prins 2004), a local search with granular neighbourhoods
//! (Toth and Vigo 2003) made of intra-route 2-opt and Or-opt and inter-route
//! relocate, swap, SWAP* (Vidal 2022: exchange two units, each inserted at
//! its best position in the other route) and 2-opt* moves, an exact dynamic
//! programme for small instances, and the Held-Karp 1-tree bound for single
//! tours. The genetic search keeps feasible and infeasible subpopulations
//! with adaptive penalties on the excess load, duration and stops, prices
//! moves from route totals and prefix sums in constant time, restarts its
//! population after a long run without improvement and can stop on a time
//! limit.

use crate::distance::Matrix;
use crate::rng::Rng;
use crate::routing::{construct, or_opt, route_length, two_opt, RouteLimits};

/// Largest number of units solved exactly by dynamic programming.
pub const EXACT_MAX_UNITS: usize = 13;
/// Size of the granular neighbourhoods of the local search.
const GRANULARITY: usize = 20;

/// Everything the solver needs about one instance.
pub(crate) struct Instance<'a> {
    pub m: &'a Matrix,
    /// The first depot (the only one for single-depot problems).
    pub depot: usize,
    /// All depots, and the number of routes each may send (`usize::MAX` for
    /// no limit).
    pub depots: Vec<usize>,
    pub fleet: Vec<usize>,
    /// Cost of opening one route, in travel units.
    pub fixed_cost: f64,
    pub service: Vec<f64>,
    pub demand: Vec<f64>,
    pub limits: RouteLimits,
    pub symmetric: bool,
    /// Nearest units of every node (units only), by travel cost.
    neighbours: Vec<Vec<usize>>,
}

impl<'a> Instance<'a> {
    #[cfg(test)]
    pub fn new(
        m: &'a Matrix,
        depot: usize,
        units: &[usize],
        service: Vec<f64>,
        demand: Vec<f64>,
        limits: RouteLimits,
        symmetric: bool,
    ) -> Self {
        Self::with_depots(
            m,
            &[depot],
            &[usize::MAX],
            0.0,
            units,
            service,
            demand,
            limits,
            symmetric,
        )
    }

    /// An instance with several depots, a route limit per depot and a fixed
    /// cost per route.
    #[allow(clippy::too_many_arguments)]
    pub fn with_depots(
        m: &'a Matrix,
        depots: &[usize],
        fleet: &[usize],
        fixed_cost: f64,
        units: &[usize],
        service: Vec<f64>,
        demand: Vec<f64>,
        limits: RouteLimits,
        symmetric: bool,
    ) -> Self {
        let n = m.len();
        let depot = depots[0];
        let mut neighbours = vec![Vec::new(); n];
        for &u in units.iter().chain(depots.iter()) {
            let mut others: Vec<(f64, usize)> = units
                .iter()
                .filter(|&&v| v != u)
                .map(|&v| (m.get(u, v).min(m.get(v, u)), v))
                .collect();
            others.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            others.truncate(GRANULARITY);
            neighbours[u] = others.into_iter().map(|(_, v)| v).collect();
        }
        Instance {
            m,
            depot,
            depots: depots.to_vec(),
            fleet: fleet.to_vec(),
            fixed_cost,
            service,
            demand,
            limits,
            symmetric,
            neighbours,
        }
    }

    #[inline]
    fn c(&self, i: usize, j: usize) -> f64 {
        self.m.get(i, j)
    }

    /// Travel, duration and load of a route.
    pub fn evaluate(&self, route: &[usize]) -> (f64, f64, f64) {
        let travel = route_length(self.m, self.depot, route);
        let served: f64 = route.iter().map(|&u| self.service[u]).sum();
        let load: f64 = route.iter().map(|&u| self.demand[u]).sum();
        (travel, travel + served, load)
    }

    pub fn feasible(&self, route: &[usize]) -> bool {
        if route.len() > self.limits.max_stops {
            return false;
        }
        let (_, duration, load) = self.evaluate(route);
        duration <= self.limits.max_length + 1e-9 && load <= self.limits.capacity + 1e-9
    }

    /// Whether a single route through every unit is allowed by the limits.
    pub fn single_route_allowed(&self, units: &[usize]) -> bool {
        self.limits.max_length.is_infinite()
            && self.limits.max_stops >= units.len()
            && units.iter().map(|&u| self.demand[u]).sum::<f64>() <= self.limits.capacity + 1e-9
    }
}

/// Penalty weights on the excess load, duration and stops of a route and on
/// the excess routes of a depot; the search walks through infeasible
/// solutions at this price (Vidal 2022).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Penalties {
    pub load: f64,
    pub duration: f64,
    pub stops: f64,
    pub fleet: f64,
}

impl Penalties {
    fn scaled(self, k: f64) -> Penalties {
        Penalties {
            load: self.load * k,
            duration: self.duration * k,
            stops: self.stops * k,
            fleet: self.fleet * k,
        }
    }
}

/// Totals and prefix sums of one route, so that moves are priced in O(1).
#[derive(Clone, Debug, Default)]
struct RouteInfo {
    travel: f64,
    load: f64,
    service: f64,
    /// `pre_t[k]`: travel from the depot to `route[k]`; `suf_t[k]`: travel from
    /// `route[k]` to the depot through the rest of the route.
    pre_t: Vec<f64>,
    suf_t: Vec<f64>,
    pre_l: Vec<f64>,
    pre_s: Vec<f64>,
}

impl RouteInfo {
    fn new(inst: &Instance, route: &[usize], d: usize) -> Self {
        let n = route.len();
        let mut info = RouteInfo::default();
        if n == 0 {
            return info;
        }
        info.pre_t = vec![0.0; n];
        info.suf_t = vec![0.0; n];
        info.pre_l = vec![0.0; n];
        info.pre_s = vec![0.0; n];
        info.pre_t[0] = inst.c(d, route[0]);
        info.pre_l[0] = inst.demand[route[0]];
        info.pre_s[0] = inst.service[route[0]];
        for k in 1..n {
            info.pre_t[k] = info.pre_t[k - 1] + inst.c(route[k - 1], route[k]);
            info.pre_l[k] = info.pre_l[k - 1] + inst.demand[route[k]];
            info.pre_s[k] = info.pre_s[k - 1] + inst.service[route[k]];
        }
        info.suf_t[n - 1] = inst.c(route[n - 1], d);
        for k in (0..n - 1).rev() {
            info.suf_t[k] = inst.c(route[k], route[k + 1]) + info.suf_t[k + 1];
        }
        info.travel = info.pre_t[n - 1] + inst.c(route[n - 1], d);
        info.load = info.pre_l[n - 1];
        info.service = info.pre_s[n - 1];
        info
    }
}

#[inline]
fn excess(x: f64, limit: f64) -> f64 {
    if x > limit {
        x - limit
    } else {
        0.0
    }
}

impl<'a> Instance<'a> {
    /// Penalised cost of a route given its totals (fixed cost included).
    #[inline]
    fn route_cost(
        &self,
        travel: f64,
        load: f64,
        service: f64,
        stops: usize,
        pen: &Penalties,
    ) -> f64 {
        let l = &self.limits;
        travel
            + self.fixed_cost
            + pen.load * excess(load, l.capacity)
            + pen.duration * excess(travel + service, l.max_length)
            + pen.stops
                * excess(
                    stops as f64,
                    if l.max_stops == usize::MAX {
                        f64::INFINITY
                    } else {
                        l.max_stops as f64
                    },
                )
    }

    #[inline]
    fn info_cost(&self, info: &RouteInfo, stops: usize, pen: &Penalties) -> f64 {
        self.route_cost(info.travel, info.load, info.service, stops, pen)
    }

    /// Penalty for `count` routes at depot `k` (index into `depots`).
    #[inline]
    fn fleet_cost(&self, k: usize, count: usize, pen: &Penalties) -> f64 {
        let cap = self.fleet[k];
        if cap == usize::MAX || count <= cap {
            0.0
        } else {
            pen.fleet * (count - cap) as f64
        }
    }

    /// Cheapest depot for a route from `first` to `last` (index into `depots`).
    fn best_depot(&self, first: usize, last: usize) -> (usize, f64) {
        let mut best = (0usize, f64::INFINITY);
        for (k, &d) in self.depots.iter().enumerate() {
            let c = self.c(d, first) + self.c(last, d);
            if c < best.1 {
                best = (k, c);
            }
        }
        best
    }
}

/// A set of routes, each with its depot (index into `inst.depots`), with
/// their totals and the number of routes per depot.
#[derive(Clone, Debug)]
pub(crate) struct Sol {
    routes: Vec<Vec<usize>>,
    depot_of: Vec<usize>,
    infos: Vec<RouteInfo>,
    count: Vec<usize>,
}

impl Sol {
    fn new(inst: &Instance, routes: Vec<Vec<usize>>, depot_of: Vec<usize>) -> Self {
        let infos = routes
            .iter()
            .zip(&depot_of)
            .map(|(r, &k)| RouteInfo::new(inst, r, inst.depots[k]))
            .collect();
        let mut count = vec![0usize; inst.depots.len()];
        for &k in &depot_of {
            count[k] += 1;
        }
        Sol {
            routes,
            depot_of,
            infos,
            count,
        }
    }

    fn travel(&self) -> f64 {
        self.infos.iter().map(|i| i.travel).sum()
    }

    fn cost(&self, inst: &Instance, pen: &Penalties) -> f64 {
        let routes: f64 = self
            .routes
            .iter()
            .zip(&self.infos)
            .map(|(r, i)| inst.info_cost(i, r.len(), pen))
            .sum();
        let fleet: f64 = (0..inst.depots.len())
            .map(|k| inst.fleet_cost(k, self.count[k], pen))
            .sum();
        routes + fleet
    }

    fn feasible(&self, inst: &Instance) -> bool {
        self.routes.iter().all(|r| inst.feasible(r))
            && (0..inst.depots.len())
                .all(|k| inst.fleet[k] == usize::MAX || self.count[k] <= inst.fleet[k])
    }

    fn refresh(&mut self, inst: &Instance, k: usize) {
        self.infos[k] = RouteInfo::new(inst, &self.routes[k], inst.depots[self.depot_of[k]]);
    }

    fn drop_empty(&mut self) {
        let keep: Vec<bool> = self.routes.iter().map(|r| !r.is_empty()).collect();
        for (k, &kept) in keep.iter().enumerate() {
            if !kept {
                self.count[self.depot_of[k]] -= 1;
            }
        }
        let mut k = 0;
        self.routes.retain(|_| {
            k += 1;
            keep[k - 1]
        });
        k = 0;
        self.depot_of.retain(|_| {
            k += 1;
            keep[k - 1]
        });
        k = 0;
        self.infos.retain(|_| {
            k += 1;
            keep[k - 1]
        });
    }
}

/// Optimal split of a giant tour into routes (Prins 2004) under the
/// penalised cost, each route leaving from its cheapest depot: routes may
/// exceed the limits by up to half, at the penalty's price. `None` when no
/// unit can start a route.
fn split_penalised(
    inst: &Instance,
    tour: &[usize],
    pen: &Penalties,
) -> Option<(Vec<Vec<usize>>, Vec<usize>)> {
    let n = tour.len();
    let l = &inst.limits;
    let max_load = if l.capacity.is_finite() {
        1.5 * l.capacity
    } else {
        f64::INFINITY
    };
    let max_dur = if l.max_length.is_finite() {
        1.5 * l.max_length
    } else {
        f64::INFINITY
    };
    let max_stops = if l.max_stops == usize::MAX {
        usize::MAX
    } else {
        (3 * l.max_stops + 1) / 2
    };
    let mut cost = vec![f64::INFINITY; n + 1];
    let mut pred = vec![usize::MAX; n + 1];
    let mut dep = vec![0usize; n + 1];
    cost[0] = 0.0;
    for i in 0..n {
        if !cost[i].is_finite() {
            continue;
        }
        let (mut inner, mut served, mut load) = (0.0, 0.0, 0.0);
        for j in i..n {
            let stops = j - i + 1;
            if stops > max_stops {
                break;
            }
            if j > i {
                inner += inst.c(tour[j - 1], tour[j]);
            }
            served += inst.service[tour[j]];
            load += inst.demand[tour[j]];
            if load > max_load + 1e-9 {
                break;
            }
            let (k, ends) = inst.best_depot(tour[i], tour[j]);
            let travel = inner + ends;
            if travel + served > max_dur + 1e-9 {
                break;
            }
            let route_cost = inst.route_cost(travel, load, served, stops, pen);
            if cost[i] + route_cost < cost[j + 1] {
                cost[j + 1] = cost[i] + route_cost;
                pred[j + 1] = i;
                dep[j + 1] = k;
            }
        }
    }
    if !cost[n].is_finite() {
        return None;
    }
    let mut routes = Vec::new();
    let mut depots = Vec::new();
    let mut j = n;
    while j > 0 {
        let i = pred[j];
        routes.push(tour[i..j].to_vec());
        depots.push(dep[j]);
        j = i;
    }
    routes.reverse();
    depots.reverse();
    Some((routes, depots))
}

/// Feasible split of a giant tour (hard limits, single depot), used in tests.
#[cfg(test)]
pub(crate) fn split(inst: &Instance, tour: &[usize]) -> Option<Vec<Vec<usize>>> {
    let hard = Penalties {
        load: 1e12,
        duration: 1e12,
        stops: 1e12,
        fleet: 1e12,
    };
    let (routes, _) = split_penalised(inst, tour, &hard)?;
    if routes.iter().all(|r| inst.feasible(r)) {
        Some(routes)
    } else {
        None
    }
}

#[cfg(test)]
fn total_travel(inst: &Instance, routes: &[Vec<usize>]) -> f64 {
    routes
        .iter()
        .map(|r| route_length(inst.m, inst.depot, r))
        .sum()
}

/// Cheapest insertion of `u` into `route` from depot `d` (travel increase
/// and position).
fn best_insertion(inst: &Instance, route: &[usize], u: usize, d: usize) -> (f64, usize) {
    let mut best = (f64::INFINITY, 0usize);
    for p in 0..=route.len() {
        let x = if p == 0 { d } else { route[p - 1] };
        let y = if p == route.len() { d } else { route[p] };
        let c = inst.c(x, u) + inst.c(u, y) - inst.c(x, y);
        if c < best.0 {
            best = (c, p);
        }
    }
    best
}

/// Local search over a solution under penalised costs: intra-route 2-opt,
/// Or-opt and swap; inter-route relocate (also into a new route from the
/// cheapest depot), swap, SWAP* and 2-opt*, restricted to granular
/// neighbourhoods; and a change of depot for a whole route. Moves are
/// priced from the route totals in constant time (SWAP* in linear time) and
/// accepted when they lower the penalised cost.
fn local_search_pen(inst: &Instance, sol: &mut Sol, pen: &Penalties, rng: &mut Rng) {
    let n = inst.m.len();
    let mut guard = 0usize;
    loop {
        guard += 1;
        let mut improved = false;
        for k in 0..sol.routes.len() {
            let d = inst.depots[sol.depot_of[k]];
            let mut changed = false;
            if two_opt(inst.m, d, &mut sol.routes[k], inst.symmetric) {
                changed = true;
            }
            if or_opt(inst.m, d, &mut sol.routes[k], inst.symmetric) {
                changed = true;
            }
            if changed {
                sol.refresh(inst, k);
                improved = true;
            }
        }
        sol.drop_empty();
        // change of depot of a whole route
        if inst.depots.len() > 1 {
            for a in 0..sol.routes.len() {
                let ka = sol.depot_of[a];
                let da = inst.depots[ka];
                let (first, last) = (sol.routes[a][0], *sol.routes[a].last().unwrap());
                let ia = &sol.infos[a];
                let base_travel = ia.travel - inst.c(da, first) - inst.c(last, da);
                let cur = inst.info_cost(ia, sol.routes[a].len(), pen)
                    + inst.fleet_cost(ka, sol.count[ka], pen);
                let mut best: Option<(usize, f64)> = None;
                for (k, &d) in inst.depots.iter().enumerate() {
                    if k == ka {
                        continue;
                    }
                    let travel = base_travel + inst.c(d, first) + inst.c(last, d);
                    let new =
                        inst.route_cost(travel, ia.load, ia.service, sol.routes[a].len(), pen)
                            + inst.fleet_cost(k, sol.count[k] + 1, pen)
                            + inst.fleet_cost(ka, sol.count[ka] - 1, pen)
                            - inst.fleet_cost(ka, sol.count[ka], pen);
                    if new < cur - 1e-9 && best.map_or(true, |b| new < b.1) {
                        best = Some((k, new));
                    }
                }
                if let Some((k, _)) = best {
                    sol.count[ka] -= 1;
                    sol.count[k] += 1;
                    sol.depot_of[a] = k;
                    sol.refresh(inst, a);
                    improved = true;
                }
            }
        }
        let mut pos = vec![(usize::MAX, usize::MAX); n];
        for (a, r) in sol.routes.iter().enumerate() {
            for (i, &u) in r.iter().enumerate() {
                pos[u] = (a, i);
            }
        }
        let mut order: Vec<usize> = sol.routes.iter().flatten().copied().collect();
        rng.shuffle(&mut order);
        let prev = |r: &[usize], i: usize, d: usize| if i == 0 { d } else { r[i - 1] };
        let next = |r: &[usize], i: usize, d: usize| if i + 1 >= r.len() { d } else { r[i + 1] };
        let cost_of = |travel: f64, load: f64, service: f64, stops: usize| {
            inst.route_cost(travel, load, service, stops, pen)
        };
        'scan: for &u in &order {
            let (a, i) = pos[u];
            let ra = &sol.routes[a];
            let ia = &sol.infos[a];
            let ka = sol.depot_of[a];
            let da = inst.depots[ka];
            let (pa, na) = (prev(ra, i, da), next(ra, i, da));
            let (du, su) = (inst.demand[u], inst.service[u]);
            let remove_gain = inst.c(pa, u) + inst.c(u, na) - inst.c(pa, na);
            let cost_a = inst.info_cost(ia, ra.len(), pen);
            // cost of a without u: when u is alone the route disappears (fixed cost and fleet count go with it)
            let (cost_a_without, fleet_a_after) = if ra.len() > 1 {
                (
                    cost_of(
                        ia.travel - remove_gain,
                        ia.load - du,
                        ia.service - su,
                        ra.len() - 1,
                    ),
                    0.0,
                )
            } else {
                (
                    0.0,
                    inst.fleet_cost(ka, sol.count[ka] - 1, pen)
                        - inst.fleet_cost(ka, sol.count[ka], pen),
                )
            };
            // relocate u into a route of its own from its cheapest depot
            if ra.len() > 1 {
                let (k, ends) = inst.best_depot(u, u);
                let alone = cost_of(ends, du, su, 1) + inst.fleet_cost(k, sol.count[k] + 1, pen)
                    - inst.fleet_cost(k, sol.count[k], pen);
                if cost_a_without + alone < cost_a - 1e-9 {
                    sol.routes[a].remove(i);
                    sol.routes.push(vec![u]);
                    sol.depot_of.push(k);
                    sol.count[k] += 1;
                    sol.infos.push(RouteInfo::default());
                    let last = sol.routes.len() - 1;
                    sol.refresh(inst, a);
                    sol.refresh(inst, last);
                    improved = true;
                    break 'scan;
                }
            }
            // intra-route swap with a neighbour
            for &v in &inst.neighbours[u] {
                let (b, j) = pos[v];
                if b != a || j == i {
                    continue;
                }
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                let (x, y) = (ra[lo], ra[hi]);
                let (px, ny) = (prev(ra, lo, da), next(ra, hi, da));
                let delta = if hi == lo + 1 {
                    inst.c(px, y) + inst.c(y, x) + inst.c(x, ny)
                        - inst.c(px, x)
                        - inst.c(x, y)
                        - inst.c(y, ny)
                } else {
                    let (nx, py) = (ra[lo + 1], ra[hi - 1]);
                    inst.c(px, y) + inst.c(y, nx) + inst.c(py, x) + inst.c(x, ny)
                        - inst.c(px, x)
                        - inst.c(x, nx)
                        - inst.c(py, y)
                        - inst.c(y, ny)
                };
                if delta < -1e-9 {
                    sol.routes[a].swap(lo, hi);
                    sol.refresh(inst, a);
                    improved = true;
                    break 'scan;
                }
            }
            // relocate u at the start of every other route
            for b in 0..sol.routes.len() {
                if b == a {
                    continue;
                }
                let rb = &sol.routes[b];
                let ib = &sol.infos[b];
                let db = inst.depots[sol.depot_of[b]];
                let first = rb[0];
                let ins = inst.c(db, u) + inst.c(u, first) - inst.c(db, first);
                let cost_b = inst.info_cost(ib, rb.len(), pen);
                let new_b = cost_of(ib.travel + ins, ib.load + du, ib.service + su, rb.len() + 1);
                if cost_a_without + fleet_a_after + new_b < cost_a + cost_b - 1e-9 {
                    sol.routes[a].remove(i);
                    sol.routes[b].insert(0, u);
                    sol.refresh(inst, a);
                    sol.refresh(inst, b);
                    improved = true;
                    break 'scan;
                }
            }
            for &v in &inst.neighbours[u] {
                let (b, j) = pos[v];
                if b == usize::MAX || b == a {
                    continue;
                }
                let rb = &sol.routes[b];
                let ib = &sol.infos[b];
                let db = inst.depots[sol.depot_of[b]];
                let (pb, nb) = (prev(rb, j, db), next(rb, j, db));
                let (dv, sv) = (inst.demand[v], inst.service[v]);
                let cost_b = inst.info_cost(ib, rb.len(), pen);
                let base = cost_a + cost_b;
                // relocate u after v
                let ins = inst.c(v, u) + inst.c(u, nb) - inst.c(v, nb);
                let new_b = cost_of(ib.travel + ins, ib.load + du, ib.service + su, rb.len() + 1);
                if cost_a_without + fleet_a_after + new_b < base - 1e-9 {
                    sol.routes[a].remove(i);
                    sol.routes[b].insert(j + 1, u);
                    sol.refresh(inst, a);
                    sol.refresh(inst, b);
                    improved = true;
                    break 'scan;
                }
                // relocate u before v
                let ins = inst.c(pb, u) + inst.c(u, v) - inst.c(pb, v);
                let new_b = cost_of(ib.travel + ins, ib.load + du, ib.service + su, rb.len() + 1);
                if cost_a_without + fleet_a_after + new_b < base - 1e-9 {
                    sol.routes[a].remove(i);
                    sol.routes[b].insert(j, u);
                    sol.refresh(inst, a);
                    sol.refresh(inst, b);
                    improved = true;
                    break 'scan;
                }
                // swap u and v in place
                let ta = ia.travel - inst.c(pa, u) - inst.c(u, na) + inst.c(pa, v) + inst.c(v, na);
                let tb = ib.travel - inst.c(pb, v) - inst.c(v, nb) + inst.c(pb, u) + inst.c(u, nb);
                let new_a = cost_of(ta, ia.load - du + dv, ia.service - su + sv, ra.len());
                let new_b = cost_of(tb, ib.load - dv + du, ib.service - sv + su, rb.len());
                if new_a + new_b < base - 1e-9 {
                    sol.routes[a][i] = v;
                    sol.routes[b][j] = u;
                    sol.refresh(inst, a);
                    sol.refresh(inst, b);
                    improved = true;
                    break 'scan;
                }
                // 2-opt*: a[..=i] + b[j+1..] (from a's depot) and b[..=j] + a[i+1..] (from b's depot)
                let cross_a = if j + 1 < rb.len() {
                    inst.c(u, rb[j + 1]) + ib.suf_t[j + 1] - inst.c(*rb.last().unwrap(), db)
                        + inst.c(*rb.last().unwrap(), da)
                } else {
                    inst.c(u, da)
                };
                let cross_b = if i + 1 < ra.len() {
                    inst.c(v, ra[i + 1]) + ia.suf_t[i + 1] - inst.c(*ra.last().unwrap(), da)
                        + inst.c(*ra.last().unwrap(), db)
                } else {
                    inst.c(v, db)
                };
                let ta = ia.pre_t[i] + cross_a;
                let tb = ib.pre_t[j] + cross_b;
                let la = ia.pre_l[i] + (ib.load - ib.pre_l[j]);
                let lb = ib.pre_l[j] + (ia.load - ia.pre_l[i]);
                let sa = ia.pre_s[i] + (ib.service - ib.pre_s[j]);
                let sb = ib.pre_s[j] + (ia.service - ia.pre_s[i]);
                let na_stops = i + 1 + (rb.len() - j - 1);
                let nb_stops = j + 1 + (ra.len() - i - 1);
                let new_a = cost_of(ta, la, sa, na_stops);
                let new_b = cost_of(tb, lb, sb, nb_stops);
                if new_a + new_b < base - 1e-9 {
                    let mut new_ra: Vec<usize> = ra[..=i].to_vec();
                    new_ra.extend_from_slice(&rb[j + 1..]);
                    let mut new_rb: Vec<usize> = rb[..=j].to_vec();
                    new_rb.extend_from_slice(&ra[i + 1..]);
                    sol.routes[a] = new_ra;
                    sol.routes[b] = new_rb;
                    sol.refresh(inst, a);
                    sol.refresh(inst, b);
                    improved = true;
                    break 'scan;
                }
                // SWAP*: u to its best position in b without v, v to its best position in a without u
                let remove_gain_v = inst.c(pb, v) + inst.c(v, nb) - inst.c(pb, nb);
                let mut ra2 = ra.clone();
                ra2.remove(i);
                let mut rb2 = rb.clone();
                rb2.remove(j);
                let (ins_u, pos_u) = best_insertion(inst, &rb2, u, db);
                let (ins_v, pos_v) = best_insertion(inst, &ra2, v, da);
                let new_a = cost_of(
                    ia.travel - remove_gain + ins_v,
                    ia.load - du + dv,
                    ia.service - su + sv,
                    ra.len(),
                );
                let new_b = cost_of(
                    ib.travel - remove_gain_v + ins_u,
                    ib.load - dv + du,
                    ib.service - sv + su,
                    rb.len(),
                );
                if new_a + new_b < base - 1e-9 {
                    rb2.insert(pos_u, u);
                    ra2.insert(pos_v, v);
                    sol.routes[a] = ra2;
                    sol.routes[b] = rb2;
                    sol.refresh(inst, a);
                    sol.refresh(inst, b);
                    improved = true;
                    break 'scan;
                }
            }
        }
        sol.drop_empty();
        if !improved || guard > 20_000 {
            break;
        }
    }
}

/// Local search with hard limits (feasible in, feasible out), single depot, used in tests.
#[cfg(test)]
pub(crate) fn local_search(inst: &Instance, routes: &mut Vec<Vec<usize>>, rng: &mut Rng) {
    let hard = Penalties {
        load: 1e12,
        duration: 1e12,
        stops: 1e12,
        fleet: 1e12,
    };
    let k = routes.len();
    let mut sol = Sol::new(inst, std::mem::take(routes), vec![0; k]);
    local_search_pen(inst, &mut sol, &hard, rng);
    *routes = sol.routes;
}

struct Indiv {
    tour: Vec<usize>,
    routes: Vec<Vec<usize>>,
    depot_of: Vec<usize>,
    /// Penalised cost at the time of insertion.
    cost: f64,
}

/// Broken-pairs distance between two giant tours (fraction of successor
/// pairs of `a`, depot included, not present in `b`).
fn broken_pairs(a: &[usize], b: &[usize], n: usize, depot: usize) -> f64 {
    let mut succ = vec![usize::MAX; n];
    let mut prevb = depot;
    for &u in b {
        succ[prevb] = u;
        prevb = u;
    }
    succ[prevb] = depot;
    let mut prev = depot;
    let mut broken = 0usize;
    for &u in a {
        if succ[prev] != u {
            broken += 1;
        }
        prev = u;
    }
    if succ[prev] != depot {
        broken += 1;
    }
    broken as f64 / (a.len() + 1) as f64
}

/// Order crossover of two permutations.
fn order_crossover(p1: &[usize], p2: &[usize], rng: &mut Rng) -> Vec<usize> {
    let n = p1.len();
    let mut i = rng.below(n);
    let mut j = rng.below(n);
    if i > j {
        std::mem::swap(&mut i, &mut j);
    }
    let mut child = vec![usize::MAX; n];
    let mut used = vec![false; p1.iter().max().copied().unwrap_or(0) + 1];
    for k in i..=j {
        child[k] = p1[k];
        used[p1[k]] = true;
    }
    let mut k = (j + 1) % n;
    for t in 0..n {
        let v = p2[(j + 1 + t) % n];
        if !used[v] {
            child[k] = v;
            used[v] = true;
            k = (k + 1) % n;
        }
    }
    child
}

/// Biased fitness ranks (lower is better) from cost and diversity within a
/// subpopulation.
fn biased_fitness(
    pop: &[Indiv],
    n: usize,
    depot: usize,
    nb_elite: usize,
    nb_close: usize,
) -> Vec<f64> {
    let len = pop.len();
    if len <= 1 {
        return vec![0.0; len];
    }
    let mut cost_rank: Vec<usize> = (0..len).collect();
    cost_rank.sort_by(|&a, &b| {
        pop[a]
            .cost
            .partial_cmp(&pop[b].cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rank_cost = vec![0.0; len];
    for (r, &i) in cost_rank.iter().enumerate() {
        rank_cost[i] = r as f64;
    }
    let diversity: Vec<f64> = (0..len)
        .map(|i| {
            let mut ds: Vec<f64> = (0..len)
                .filter(|&j| j != i)
                .map(|j| broken_pairs(&pop[i].tour, &pop[j].tour, n, depot))
                .collect();
            ds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let k = nb_close.min(ds.len()).max(1);
            ds[..k].iter().sum::<f64>() / k as f64
        })
        .collect();
    let mut div_rank: Vec<usize> = (0..len).collect();
    div_rank.sort_by(|&a, &b| {
        diversity[b]
            .partial_cmp(&diversity[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rank_div = vec![0.0; len];
    for (r, &i) in div_rank.iter().enumerate() {
        rank_div[i] = r as f64;
    }
    let w = 1.0 - nb_elite as f64 / len as f64;
    (0..len)
        .map(|i| (rank_cost[i] + w * rank_div[i]) / (len - 1) as f64)
        .collect()
}

/// Survivor selection of one subpopulation: clones first, then the worst
/// biased fitness, never the best individual.
fn survivors(
    pop: &mut Vec<Indiv>,
    mu: usize,
    n: usize,
    depot: usize,
    nb_elite: usize,
    nb_close: usize,
) {
    while pop.len() > mu {
        let fit = biased_fitness(pop, n, depot, nb_elite, nb_close);
        let best_idx = (0..pop.len())
            .min_by(|&a, &b| {
                pop[a]
                    .cost
                    .partial_cmp(&pop[b].cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let clone = (0..pop.len()).find(|&i| {
            i != best_idx && (0..pop.len()).any(|j| j != i && pop[i].tour == pop[j].tour)
        });
        let drop = clone.unwrap_or_else(|| {
            (0..pop.len())
                .filter(|&i| i != best_idx)
                .max_by(|&a, &b| {
                    fit[a]
                        .partial_cmp(&fit[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap()
        });
        pop.swap_remove(drop);
    }
}

/// Initial penalties from the scale of the instance (Vidal 2022).
fn initial_penalties(inst: &Instance, units: &[usize]) -> Penalties {
    let mut max_dist: f64 = 0.0;
    for &u in units {
        for &d in &inst.depots {
            max_dist = max_dist.max(inst.c(d, u)).max(inst.c(u, d));
        }
        for &v in &inst.neighbours[u] {
            max_dist = max_dist.max(inst.c(u, v));
        }
    }
    let max_dist = max_dist.max(1e-9);
    let max_demand = units
        .iter()
        .map(|&u| inst.demand[u])
        .fold(0.0, f64::max)
        .max(1e-9);
    Penalties {
        load: (max_dist / max_demand).clamp(0.1, 1e4),
        duration: 1.0,
        stops: max_dist,
        fleet: 2.0 * max_dist + inst.fixed_cost,
    }
}

/// Result of the genetic search: routes, their depots (indices into
/// `inst.depots`), total travel and the offspring at which the best was found.
pub(crate) type Found = (Vec<Vec<usize>>, Vec<usize>, f64, usize);

/// Hybrid genetic search with feasible and infeasible subpopulations and
/// adaptive penalties. Returns the best feasible solution.
pub(crate) fn hgs(
    inst: &Instance,
    units: &[usize],
    iterations: usize,
    alpha: f64,
    time_limit: f64,
    rng: &mut Rng,
) -> Option<Found> {
    let n = inst.m.len();
    let (mu, lambda, nb_elite, nb_close) = (10usize, 20usize, 4usize, 3usize);
    let target_feasible = 0.2;
    let start = std::time::Instant::now();
    let mut pen = initial_penalties(inst, units);
    let mut feas: Vec<Indiv> = Vec::new();
    let mut infeas: Vec<Indiv> = Vec::new();
    let mut best: Option<Found> = None;
    let mut last_improvement = 0usize;
    let patience = (iterations / 4).max(500);
    let mut hist_load: Vec<bool> = Vec::new();
    let mut hist_dur: Vec<bool> = Vec::new();
    let mut hist_stops: Vec<bool> = Vec::new();
    let mut hist_fleet: Vec<bool> = Vec::new();
    let depot0 = inst.depots[0];

    let educate = |tour: Vec<usize>, pen: &Penalties, rng: &mut Rng| -> Option<Sol> {
        let (routes, depots) = split_penalised(inst, &tour, pen)?;
        let mut sol = Sol::new(inst, routes, depots);
        local_search_pen(inst, &mut sol, pen, rng);
        Some(sol)
    };
    // objective of a feasible solution: travel plus fixed costs
    let objective = |sol: &Sol| sol.travel() + inst.fixed_cost * sol.routes.len() as f64;
    let add = |sol: Sol,
               it: usize,
               feas: &mut Vec<Indiv>,
               infeas: &mut Vec<Indiv>,
               best: &mut Option<Found>,
               pen: &Penalties|
     -> bool {
        let feasible = sol.feasible(inst);
        let obj = objective(&sol);
        let cost = sol.cost(inst, pen);
        let tour: Vec<usize> = sol.routes.iter().flatten().copied().collect();
        if feasible && best.as_ref().map_or(true, |b| obj < b.2 - 1e-9) {
            *best = Some((sol.routes.clone(), sol.depot_of.clone(), obj, it));
        }
        let ind = Indiv {
            tour,
            routes: sol.routes,
            depot_of: sol.depot_of,
            cost,
        };
        if feasible {
            feas.push(ind);
        } else {
            infeas.push(ind);
        }
        feasible
    };
    for k in 0..mu {
        let a = if k == 0 { 0.0 } else { alpha.max(0.2) };
        let tour = construct(inst.m, depot0, units, a, rng);
        let sol = educate(tour, &pen.scaled(1e9), rng)?;
        add(sol, 0, &mut feas, &mut infeas, &mut best, &pen);
    }
    if best.is_none() {
        // no feasible seed (tight fleet): keep searching with the penalties, the loop may still find one
        if feas.is_empty() && infeas.is_empty() {
            return None;
        }
    }
    let mut it = 0usize;
    loop {
        it += 1;
        let out_of_iterations = time_limit.is_infinite() && it > iterations;
        let out_of_time = time_limit.is_finite() && start.elapsed().as_secs_f64() >= time_limit;
        if out_of_iterations || out_of_time {
            break;
        }
        if let Some(b) = &best {
            if it - b.3.max(last_improvement) > patience {
                feas.clear();
                infeas.clear();
                let seed_sol = Sol::new(inst, b.0.clone(), b.1.clone());
                add(seed_sol, it, &mut feas, &mut infeas, &mut best, &pen);
                for _ in 1..mu {
                    let tour = construct(inst.m, depot0, units, alpha.max(0.3), rng);
                    if let Some(sol) = educate(tour, &pen.scaled(1e9), rng) {
                        add(sol, it, &mut feas, &mut infeas, &mut best, &pen);
                    }
                }
                last_improvement = it;
            }
        }
        let fit_f = biased_fitness(&feas, n, depot0, nb_elite, nb_close);
        let fit_i = biased_fitness(&infeas, n, depot0, nb_elite, nb_close);
        let total = feas.len() + infeas.len();
        let pick = |rng: &mut Rng| -> (bool, usize) {
            let draw = |rng: &mut Rng| {
                let k = rng.below(total);
                if k < feas.len() {
                    (true, k)
                } else {
                    (false, k - feas.len())
                }
            };
            let (a, b) = (draw(rng), draw(rng));
            let f = |x: (bool, usize)| if x.0 { fit_f[x.1] } else { fit_i[x.1] };
            if f(a) <= f(b) {
                a
            } else {
                b
            }
        };
        let (p1, p2) = (pick(rng), pick(rng));
        let t1 = if p1.0 {
            &feas[p1.1].tour
        } else {
            &infeas[p1.1].tour
        };
        let t2 = if p2.0 {
            &feas[p2.1].tour
        } else {
            &infeas[p2.1].tour
        };
        let child = order_crossover(t1, t2, rng);
        if let Some(mut sol) = educate(child, &pen, rng) {
            let l = &inst.limits;
            let (mut v_load, mut v_dur, mut v_stops) = (false, false, false);
            for (r, i) in sol.routes.iter().zip(&sol.infos) {
                if i.load > l.capacity + 1e-9 {
                    v_load = true;
                }
                if i.travel + i.service > l.max_length + 1e-9 {
                    v_dur = true;
                }
                if r.len() > l.max_stops {
                    v_stops = true;
                }
            }
            let v_fleet = (0..inst.depots.len())
                .any(|k| inst.fleet[k] != usize::MAX && sol.count[k] > inst.fleet[k]);
            hist_load.push(!v_load);
            hist_dur.push(!v_dur);
            hist_stops.push(!v_stops);
            hist_fleet.push(!v_fleet);
            let feasible = !(v_load || v_dur || v_stops || v_fleet);
            if !feasible && rng.next_f64() < 0.5 {
                let mut repaired = sol.clone();
                local_search_pen(inst, &mut repaired, &pen.scaled(10.0), rng);
                if repaired.feasible(inst) {
                    sol = repaired;
                }
            }
            let was_feasible = add(sol, it, &mut feas, &mut infeas, &mut best, &pen);
            if was_feasible {
                if let Some(b) = &best {
                    if b.3 == it {
                        last_improvement = it;
                    }
                }
            }
        }
        if feas.len() >= mu + lambda {
            survivors(&mut feas, mu, n, depot0, nb_elite, nb_close);
        }
        if infeas.len() >= mu + lambda {
            survivors(&mut infeas, mu, n, depot0, nb_elite, nb_close);
        }
        if it % 100 == 0 {
            let adjust = |hist: &mut Vec<bool>, p: &mut f64| {
                if hist.is_empty() {
                    return;
                }
                let frac = hist.iter().filter(|&&f| f).count() as f64 / hist.len() as f64;
                if frac < target_feasible - 0.05 {
                    *p = (*p * 1.2).min(1e6);
                } else if frac > target_feasible + 0.05 {
                    *p = (*p * 0.85).max(1e-3);
                }
                hist.clear();
            };
            adjust(&mut hist_load, &mut pen.load);
            adjust(&mut hist_dur, &mut pen.duration);
            adjust(&mut hist_stops, &mut pen.stops);
            adjust(&mut hist_fleet, &mut pen.fleet);
            for ind in feas.iter_mut().chain(infeas.iter_mut()) {
                let s = Sol::new(inst, ind.routes.clone(), ind.depot_of.clone());
                ind.cost = s.cost(inst, &pen);
            }
        }
    }
    best
}

/// Exact solution by dynamic programming over subsets: optimal tours of
/// every subset from its best depot (Held and Karp 1962) and an optimal
/// partition into feasible routes, each paying the fixed cost. For at most
/// [`EXACT_MAX_UNITS`] units; route limits per depot are ignored (the caller
/// uses it only when they do not bind). Returns `None` when the limits
/// cannot be met.
pub(crate) fn exact(
    inst: &Instance,
    units: &[usize],
) -> Option<(Vec<Vec<usize>>, Vec<usize>, f64)> {
    let n = units.len();
    if n == 0 || n > EXACT_MAX_UNITS {
        return None;
    }
    let full = (1usize << n) - 1;
    let size = 1usize << n;
    let inf = f64::INFINITY;
    // best tour of every subset over all depots: cost, depot, last unit, and the
    // predecessor table of that depot
    let mut tour = vec![inf; size];
    let mut tour_dep = vec![0usize; size];
    let mut last = vec![usize::MAX; size];
    let mut pars: Vec<Vec<usize>> = Vec::with_capacity(inst.depots.len());
    for (kd, &d) in inst.depots.iter().enumerate() {
        let mut dp = vec![inf; size * n];
        let mut par = vec![usize::MAX; size * n];
        for j in 0..n {
            dp[(1 << j) * n + j] = inst.c(d, units[j]);
        }
        for s in 1..size {
            for j in 0..n {
                let cur = dp[s * n + j];
                if !cur.is_finite() || s & (1 << j) == 0 {
                    continue;
                }
                for k in 0..n {
                    if s & (1 << k) != 0 {
                        continue;
                    }
                    let t = s | (1 << k);
                    let v = cur + inst.c(units[j], units[k]);
                    if v < dp[t * n + k] {
                        dp[t * n + k] = v;
                        par[t * n + k] = j;
                    }
                }
            }
        }
        for s in 1..size {
            for j in 0..n {
                if s & (1 << j) != 0 {
                    let v = dp[s * n + j] + inst.c(units[j], d);
                    if v < tour[s] {
                        tour[s] = v;
                        tour_dep[s] = kd;
                        last[s] = j;
                    }
                }
            }
        }
        pars.push(par);
    }
    let mut ok = vec![false; size];
    for s in 1..size {
        let mut served = 0.0;
        let mut load = 0.0;
        let mut stops = 0usize;
        for (j, &u) in units.iter().enumerate() {
            if s & (1 << j) != 0 {
                served += inst.service[u];
                load += inst.demand[u];
                stops += 1;
            }
        }
        ok[s] = stops <= inst.limits.max_stops
            && load <= inst.limits.capacity + 1e-9
            && tour[s] + served <= inst.limits.max_length + 1e-9;
    }
    // optimal partition into feasible routes, each paying the fixed cost
    let mut g = vec![inf; size];
    let mut gpart = vec![0usize; size];
    g[0] = 0.0;
    for s in 1..size {
        let low = s & s.wrapping_neg();
        let mut r = s;
        while r > 0 {
            if r & low != 0 && ok[r] && g[s ^ r].is_finite() {
                let v = tour[r] + inst.fixed_cost + g[s ^ r];
                if v < g[s] {
                    g[s] = v;
                    gpart[s] = r;
                }
            }
            r = (r - 1) & s;
        }
    }
    if !g[full].is_finite() {
        return None;
    }
    let mut routes = Vec::new();
    let mut depots = Vec::new();
    let mut s = full;
    while s > 0 {
        let r = gpart[s];
        let kd = tour_dep[r];
        let par = &pars[kd];
        let mut seq = Vec::new();
        let mut cur = r;
        let mut j = last[r];
        while j != usize::MAX {
            seq.push(units[j]);
            let pj = par[cur * n + j];
            cur ^= 1 << j;
            j = pj;
        }
        seq.reverse();
        routes.push(seq);
        depots.push(kd);
        s ^= r;
    }
    Some((routes, depots, g[full]))
}

/// Held-Karp 1-tree lower bound on a single tour through the depot and the
/// units of a symmetric matrix (subgradient ascent, Held and Karp 1971).
/// Returns the bound and whether the final 1-tree is itself a tour (then
/// the bound is the optimum).
pub(crate) fn held_karp_bound(
    m: &Matrix,
    depot: usize,
    units: &[usize],
    upper: f64,
) -> (f64, bool) {
    let nodes: Vec<usize> = std::iter::once(depot)
        .chain(units.iter().copied())
        .collect();
    let k = nodes.len();
    if k < 3 {
        return (
            if k == 2 {
                2.0 * m.get(nodes[0], nodes[1])
            } else {
                0.0
            },
            true,
        );
    }
    let mut pi = vec![0.0; k];
    let mut best = f64::NEG_INFINITY;
    let mut lambda = 2.0;
    let mut since = 0;
    let mut is_tour = false;
    let cost = |i: usize, j: usize, pi: &[f64]| m.get(nodes[i], nodes[j]) + pi[i] + pi[j];
    for _ in 0..300 {
        // MST on nodes 1..k with Prim
        let mut in_tree = vec![false; k];
        let mut key = vec![f64::INFINITY; k];
        let mut parent = vec![usize::MAX; k];
        let mut deg = vec![0usize; k];
        in_tree[1] = true;
        for j in 2..k {
            key[j] = cost(1, j, &pi);
            parent[j] = 1;
        }
        let mut w = 0.0;
        for _ in 2..k {
            let mut bj = usize::MAX;
            for j in 2..k {
                if !in_tree[j] && (bj == usize::MAX || key[j] < key[bj]) {
                    bj = j;
                }
            }
            in_tree[bj] = true;
            w += key[bj];
            deg[bj] += 1;
            deg[parent[bj]] += 1;
            for j in 2..k {
                if !in_tree[j] {
                    let c = cost(bj, j, &pi);
                    if c < key[j] {
                        key[j] = c;
                        parent[j] = bj;
                    }
                }
            }
        }
        // two cheapest edges from the depot (node 0)
        let mut e: Vec<(f64, usize)> = (1..k).map(|j| (cost(0, j, &pi), j)).collect();
        e.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        w += e[0].0 + e[1].0;
        deg[0] = 2;
        deg[e[0].1] += 1;
        deg[e[1].1] += 1;
        w -= 2.0 * pi.iter().sum::<f64>();
        if w > best + 1e-9 {
            best = w;
            since = 0;
        } else {
            since += 1;
            if since >= 20 {
                lambda /= 2.0;
                since = 0;
            }
        }
        let norm: f64 = deg.iter().map(|&d| (d as f64 - 2.0).powi(2)).sum();
        if norm == 0.0 {
            is_tour = true;
            break;
        }
        if lambda < 1e-4 || upper - w <= 1e-9 {
            break;
        }
        let t = lambda * (upper - w) / norm;
        for i in 0..k {
            pi[i] += t * (deg[i] as f64 - 2.0);
        }
    }
    (best.min(upper), is_tour)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::euclidean_matrix;

    fn random_instance(n: usize, seed: u64) -> Matrix {
        let mut rng = Rng::new(seed);
        let x: Vec<f64> = (0..=n).map(|_| rng.next_f64() * 100.0).collect();
        let y: Vec<f64> = (0..=n).map(|_| rng.next_f64() * 100.0).collect();
        euclidean_matrix(&x, &y).unwrap()
    }

    #[test]
    fn exact_matches_brute_force_and_hgs_finds_it() {
        for seed in 0..5u64 {
            let m = random_instance(8, seed);
            let units: Vec<usize> = (1..9).collect();
            let limits = RouteLimits {
                max_length: f64::INFINITY,
                max_stops: 3,
                capacity: f64::INFINITY,
            };
            let inst = Instance::new(&m, 0, &units, vec![0.0; 9], vec![1.0; 9], limits, true);
            let (routes, _, cost) = exact(&inst, &units).unwrap();
            assert!(routes.iter().all(|r| r.len() <= 3));
            assert!((total_travel(&inst, &routes) - cost).abs() < 1e-9);
            // brute force over partitions via all permutations + split
            fn perms(items: &[usize]) -> Vec<Vec<usize>> {
                if items.len() <= 1 {
                    return vec![items.to_vec()];
                }
                let mut out = Vec::new();
                for (k, &x) in items.iter().enumerate() {
                    let mut rest = items.to_vec();
                    rest.remove(k);
                    for mut p in perms(&rest) {
                        p.insert(0, x);
                        out.push(p);
                    }
                }
                out
            }
            let brute = perms(&units)
                .iter()
                .filter_map(|p| split(&inst, p).map(|r| total_travel(&inst, &r)))
                .fold(f64::INFINITY, f64::min);
            assert!(
                (brute - cost).abs() < 1e-9,
                "seed {seed}: brute {brute} exact {cost}"
            );
            let mut rng = Rng::new(seed);
            let (_, _, hcost, _) = hgs(&inst, &units, 100, 0.3, f64::INFINITY, &mut rng).unwrap();
            assert!(
                hcost <= cost + 1e-9,
                "seed {seed}: hgs {hcost} > exact {cost}"
            );
        }
    }

    #[test]
    fn several_depots_are_chosen_well_and_fleet_limits_hold() {
        for seed in 0..4u64 {
            let m = random_instance(9, seed); // nodes 0..=9: depots 0 and 9, units 1..=8
            let units: Vec<usize> = (1..9).collect();
            let limits = RouteLimits {
                max_length: f64::INFINITY,
                max_stops: 3,
                capacity: f64::INFINITY,
            };
            let both = Instance::with_depots(
                &m,
                &[0, 9],
                &[usize::MAX, usize::MAX],
                0.0,
                &units,
                vec![0.0; 10],
                vec![0.0; 10],
                limits,
                true,
            );
            let (routes, depots, cost) = exact(&both, &units).unwrap();
            assert_eq!(routes.len(), depots.len());
            let direct: f64 = routes
                .iter()
                .zip(&depots)
                .map(|(r, &k)| route_length(&m, both.depots[k], r))
                .sum();
            assert!((direct - cost).abs() < 1e-9);
            // never worse than using either depot alone
            for d in [0usize, 9] {
                let one = Instance::with_depots(
                    &m,
                    &[d],
                    &[usize::MAX],
                    0.0,
                    &units,
                    vec![0.0; 10],
                    vec![0.0; 10],
                    limits,
                    true,
                );
                let (_, _, c1) = exact(&one, &units).unwrap();
                assert!(
                    cost <= c1 + 1e-9,
                    "seed {seed}: two depots {cost} > depot {d} alone {c1}"
                );
            }
            // the genetic search matches the exact optimum on these small instances
            let mut rng = Rng::new(seed);
            let (_, _, hcost, _) = hgs(&both, &units, 300, 0.3, f64::INFINITY, &mut rng).unwrap();
            assert!(
                hcost <= cost + 1e-6,
                "seed {seed}: hgs {hcost} > exact {cost}"
            );
            // a fixed cost per route reduces the number of routes
            let fixed = Instance::with_depots(
                &m,
                &[0, 9],
                &[usize::MAX, usize::MAX],
                50.0,
                &units,
                vec![0.0; 10],
                vec![0.0; 10],
                limits,
                true,
            );
            let (r_fixed, _, _) = exact(&fixed, &units).unwrap();
            assert!(r_fixed.len() <= routes.len());
            assert_eq!(r_fixed.len(), 3); // 8 units, at most 3 per route: three routes at the least
        }
        // fleet limit: depot 0 may send one route only
        let m = random_instance(20, 5); // depots 0 and 20
        let units: Vec<usize> = (1..20).collect();
        let limits = RouteLimits {
            max_length: f64::INFINITY,
            max_stops: 4,
            capacity: f64::INFINITY,
        };
        let inst = Instance::with_depots(
            &m,
            &[0, 20],
            &[1, usize::MAX],
            0.0,
            &units,
            vec![0.0; 21],
            vec![0.0; 21],
            limits,
            true,
        );
        let mut rng = Rng::new(1);
        let (routes, depots, _, _) = hgs(&inst, &units, 500, 0.3, f64::INFINITY, &mut rng).unwrap();
        assert!(depots.iter().filter(|&&k| k == 0).count() <= 1);
        assert!(routes.iter().all(|r| r.len() <= 4));
        let mut all: Vec<usize> = routes.concat();
        all.sort();
        assert_eq!(all, units);
    }

    #[test]
    fn held_karp_is_a_valid_bound_and_certifies_small_tours() {
        for seed in 0..5u64 {
            let m = random_instance(10, seed);
            let units: Vec<usize> = (1..11).collect();
            let inst = Instance::new(
                &m,
                0,
                &units,
                vec![0.0; 11],
                vec![0.0; 11],
                RouteLimits::default(),
                true,
            );
            let (_, _, opt) = exact(&inst, &units).unwrap();
            let (lb, _) = held_karp_bound(&m, 0, &units, opt);
            assert!(
                lb <= opt + 1e-9,
                "seed {seed}: bound {lb} above optimum {opt}"
            );
            assert!(
                lb >= 0.9 * opt,
                "seed {seed}: bound {lb} far below optimum {opt}"
            );
        }
    }

    #[test]
    fn local_search_respects_capacity_and_improves() {
        let m = random_instance(40, 3);
        let units: Vec<usize> = (1..41).collect();
        let limits = RouteLimits {
            max_length: f64::INFINITY,
            max_stops: usize::MAX,
            capacity: 10.0,
        };
        let demand: Vec<f64> = (0..41)
            .map(|i| if i == 0 { 0.0 } else { 1.0 + (i % 3) as f64 })
            .collect();
        let inst = Instance::new(&m, 0, &units, vec![0.0; 41], demand.clone(), limits, true);
        let mut rng = Rng::new(1);
        let tour = construct(&m, 0, &units, 0.5, &mut rng);
        let mut routes = split(&inst, &tour).unwrap();
        let before = total_travel(&inst, &routes);
        local_search(&inst, &mut routes, &mut rng);
        let after = total_travel(&inst, &routes);
        assert!(after < before);
        for r in &routes {
            assert!(r.iter().map(|&u| demand[u]).sum::<f64>() <= 10.0 + 1e-9);
        }
        let mut all: Vec<usize> = routes.concat();
        all.sort();
        assert_eq!(all, units);
    }
}
