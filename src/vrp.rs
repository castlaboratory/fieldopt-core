//! The routing solver behind [`crate::routing::solve_routes`]: a hybrid
//! genetic search in the spirit of Vidal (2022) with an optimal split of
//! giant tours (Prins 2004), a local search with granular neighbourhoods
//! (Toth and Vigo 2003) made of intra-route 2-opt and Or-opt and inter-route
//! relocate, swap, SWAP* (Vidal 2022: exchange two units, each inserted at
//! its best position in the other route) and 2-opt* moves, an exact dynamic
//! programme for small instances, and the Held-Karp 1-tree bound for single
//! tours. The genetic search restarts its population after a long run
//! without improvement and can stop on a time limit.

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
    pub depot: usize,
    pub service: Vec<f64>,
    pub demand: Vec<f64>,
    pub limits: RouteLimits,
    pub symmetric: bool,
    /// Nearest units of every node (units only), by travel cost.
    neighbours: Vec<Vec<usize>>,
}

impl<'a> Instance<'a> {
    pub fn new(
        m: &'a Matrix,
        depot: usize,
        units: &[usize],
        service: Vec<f64>,
        demand: Vec<f64>,
        limits: RouteLimits,
        symmetric: bool,
    ) -> Self {
        let n = m.len();
        let mut neighbours = vec![Vec::new(); n];
        for &u in units.iter().chain(std::iter::once(&depot)) {
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

/// Optimal split of a giant tour into feasible routes (Prins 2004); `None`
/// when a unit cannot be served alone.
pub(crate) fn split(inst: &Instance, tour: &[usize]) -> Option<Vec<Vec<usize>>> {
    let n = tour.len();
    let d = inst.depot;
    let mut cost = vec![f64::INFINITY; n + 1];
    let mut pred = vec![usize::MAX; n + 1];
    cost[0] = 0.0;
    for i in 0..n {
        if !cost[i].is_finite() {
            continue;
        }
        let (mut len, mut served, mut load) = (0.0, 0.0, 0.0);
        for j in i..n {
            if j - i + 1 > inst.limits.max_stops {
                break;
            }
            len += if j == i {
                inst.c(d, tour[i])
            } else {
                inst.c(tour[j - 1], tour[j])
            };
            served += inst.service[tour[j]];
            load += inst.demand[tour[j]];
            if load > inst.limits.capacity + 1e-9 {
                break;
            }
            let route_cost = len + inst.c(tour[j], d);
            if route_cost + served > inst.limits.max_length + 1e-9 {
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

fn total_travel(inst: &Instance, routes: &[Vec<usize>]) -> f64 {
    routes
        .iter()
        .map(|r| route_length(inst.m, inst.depot, r))
        .sum()
}

/// Local search over a set of routes: intra-route 2-opt and Or-opt, and
/// inter-route relocate, swap and 2-opt* restricted to granular
/// neighbourhoods. Every accepted move is feasible and strictly improving.
pub(crate) fn local_search(inst: &Instance, routes: &mut Vec<Vec<usize>>, rng: &mut Rng) {
    let d = inst.depot;
    let n = inst.m.len();
    let mut guard = 0usize;
    loop {
        guard += 1;
        let mut improved = false;
        for r in routes.iter_mut() {
            if two_opt(inst.m, d, r, inst.symmetric) {
                improved = true;
            }
            if or_opt(inst.m, d, r, inst.symmetric) {
                improved = true;
            }
        }
        routes.retain(|r| !r.is_empty());
        // positions
        let mut pos = vec![(usize::MAX, usize::MAX); n];
        for (a, r) in routes.iter().enumerate() {
            for (i, &u) in r.iter().enumerate() {
                pos[u] = (a, i);
            }
        }
        let mut order: Vec<usize> = routes.iter().flatten().copied().collect();
        rng.shuffle(&mut order);
        let prev = |r: &[usize], i: usize| if i == 0 { d } else { r[i - 1] };
        let next = |r: &[usize], i: usize| if i + 1 >= r.len() { d } else { r[i + 1] };
        'scan: for &u in &order {
            let (a, i) = pos[u];
            let (pa, na) = (prev(&routes[a], i), next(&routes[a], i));
            let remove_gain = inst.c(pa, u) + inst.c(u, na) - inst.c(pa, na);
            // intra-route swap with a neighbour (travel can only decrease: no feasibility check needed)
            for &v in &inst.neighbours[u] {
                let (b, j) = pos[v];
                if b != a || j == i {
                    continue;
                }
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                let (x, y) = (routes[a][lo], routes[a][hi]);
                let (px, ny) = (prev(&routes[a], lo), next(&routes[a], hi));
                let delta = if hi == lo + 1 {
                    inst.c(px, y) + inst.c(y, x) + inst.c(x, ny)
                        - inst.c(px, x)
                        - inst.c(x, y)
                        - inst.c(y, ny)
                } else {
                    let (nx, py) = (routes[a][lo + 1], routes[a][hi - 1]);
                    inst.c(px, y) + inst.c(y, nx) + inst.c(py, x) + inst.c(x, ny)
                        - inst.c(px, x)
                        - inst.c(x, nx)
                        - inst.c(py, y)
                        - inst.c(y, ny)
                };
                if delta < -1e-9 {
                    routes[a].swap(lo, hi);
                    improved = true;
                    break 'scan;
                }
            }
            // relocate u at the start of every other route
            for b in 0..routes.len() {
                if b == a {
                    continue;
                }
                let first = routes[b][0];
                let delta = inst.c(d, u) + inst.c(u, first) - inst.c(d, first) - remove_gain;
                if delta < -1e-9 {
                    let mut ra = routes[a].clone();
                    ra.remove(i);
                    let mut rb = routes[b].clone();
                    rb.insert(0, u);
                    if (ra.is_empty() || inst.feasible(&ra)) && inst.feasible(&rb) {
                        routes[a] = ra;
                        routes[b] = rb;
                        improved = true;
                        break 'scan;
                    }
                }
            }
            for &v in &inst.neighbours[u] {
                let (b, j) = pos[v];
                if b == usize::MAX || b == a {
                    continue;
                }
                let (pb, nb) = (prev(&routes[b], j), next(&routes[b], j));
                // relocate u after v
                let delta = inst.c(v, u) + inst.c(u, nb) - inst.c(v, nb) - remove_gain;
                if delta < -1e-9 {
                    let mut ra = routes[a].clone();
                    ra.remove(i);
                    let mut rb = routes[b].clone();
                    rb.insert(j + 1, u);
                    if (ra.is_empty() || inst.feasible(&ra)) && inst.feasible(&rb) {
                        routes[a] = ra;
                        routes[b] = rb;
                        improved = true;
                        break 'scan;
                    }
                }
                // relocate u before v
                let delta = inst.c(pb, u) + inst.c(u, v) - inst.c(pb, v) - remove_gain;
                if delta < -1e-9 {
                    let mut ra = routes[a].clone();
                    ra.remove(i);
                    let mut rb = routes[b].clone();
                    rb.insert(j, u);
                    if (ra.is_empty() || inst.feasible(&ra)) && inst.feasible(&rb) {
                        routes[a] = ra;
                        routes[b] = rb;
                        improved = true;
                        break 'scan;
                    }
                }
                // swap u and v
                let delta = inst.c(pa, v) + inst.c(v, na) - inst.c(pa, u) - inst.c(u, na)
                    + inst.c(pb, u)
                    + inst.c(u, nb)
                    - inst.c(pb, v)
                    - inst.c(v, nb);
                if delta < -1e-9 {
                    let mut ra = routes[a].clone();
                    let mut rb = routes[b].clone();
                    ra[i] = v;
                    rb[j] = u;
                    if inst.feasible(&ra) && inst.feasible(&rb) {
                        routes[a] = ra;
                        routes[b] = rb;
                        improved = true;
                        break 'scan;
                    }
                }
                // 2-opt*: exchange the tails after u and after v
                let delta = inst.c(u, nb) + inst.c(v, na) - inst.c(u, na) - inst.c(v, nb);
                if delta < -1e-9 {
                    let mut ra: Vec<usize> = routes[a][..=i].to_vec();
                    ra.extend_from_slice(&routes[b][j + 1..]);
                    let mut rb: Vec<usize> = routes[b][..=j].to_vec();
                    rb.extend_from_slice(&routes[a][i + 1..]);
                    if inst.feasible(&ra) && inst.feasible(&rb) {
                        routes[a] = ra;
                        routes[b] = rb;
                        improved = true;
                        break 'scan;
                    }
                }
                // SWAP*: u goes to its best position in b without v, v to its best position in a without u
                let remove_gain_v = inst.c(pb, v) + inst.c(v, nb) - inst.c(pb, nb);
                let mut ra = routes[a].clone();
                ra.remove(i);
                let mut rb = routes[b].clone();
                rb.remove(j);
                let (ins_u, pos_u) = best_insertion(inst, &rb, u);
                let (ins_v, pos_v) = best_insertion(inst, &ra, v);
                let delta = ins_u + ins_v - remove_gain - remove_gain_v;
                if delta < -1e-9 {
                    rb.insert(pos_u, u);
                    ra.insert(pos_v, v);
                    if inst.feasible(&ra) && inst.feasible(&rb) {
                        routes[a] = ra;
                        routes[b] = rb;
                        improved = true;
                        break 'scan;
                    }
                }
            }
        }
        routes.retain(|r| !r.is_empty());
        if !improved || guard > 10_000 {
            break;
        }
    }
}

/// Cheapest insertion of `u` into `route` (cost increase and position).
fn best_insertion(inst: &Instance, route: &[usize], u: usize) -> (f64, usize) {
    let d = inst.depot;
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

struct Indiv {
    tour: Vec<usize>,
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

/// Biased fitness ranks (lower is better) from cost and diversity.
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

/// Hybrid genetic search. Returns the best routes, their travel and the
/// offspring at which they were found.
pub(crate) fn hgs(
    inst: &Instance,
    units: &[usize],
    iterations: usize,
    alpha: f64,
    time_limit: f64,
    rng: &mut Rng,
) -> Option<(Vec<Vec<usize>>, f64, usize)> {
    let n = inst.m.len();
    let (mu, lambda, nb_elite, nb_close) = (10usize, 20usize, 4usize, 3usize);
    let start = std::time::Instant::now();
    let mut pop: Vec<Indiv> = Vec::with_capacity(mu + lambda + 1);
    let mut best: Option<(Vec<Vec<usize>>, f64, usize)> = None;
    let mut last_improvement = 0usize;
    let patience = (iterations / 4).max(500);
    let add = |pop: &mut Vec<Indiv>,
               tour: Vec<usize>,
               it: usize,
               best: &mut Option<(Vec<Vec<usize>>, f64, usize)>,
               rng: &mut Rng|
     -> bool {
        let mut routes = match split(inst, &tour) {
            Some(r) => r,
            None => return false,
        };
        local_search(inst, &mut routes, rng);
        let cost = total_travel(inst, &routes);
        let tour: Vec<usize> = routes.iter().flatten().copied().collect();
        if best.as_ref().map_or(true, |b| cost < b.1 - 1e-9) {
            *best = Some((routes.clone(), cost, it));
        }
        pop.push(Indiv { tour, cost });
        true
    };
    for k in 0..mu {
        let a = if k == 0 { 0.0 } else { alpha.max(0.2) };
        let tour = construct(inst.m, inst.depot, units, a, rng);
        if !add(&mut pop, tour, 0, &mut best, rng) {
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
        // restart after a long run without improvement, keeping the best
        if let Some(b) = &best {
            if it - b.2.max(last_improvement) > patience {
                let keep: Vec<usize> = b.0.iter().flatten().copied().collect();
                pop.clear();
                pop.push(Indiv {
                    tour: keep,
                    cost: b.1,
                });
                for _ in 1..mu {
                    let tour = construct(inst.m, inst.depot, units, alpha.max(0.3), rng);
                    add(&mut pop, tour, it, &mut best, rng);
                }
                last_improvement = it;
            }
        }
        let fit = biased_fitness(&pop, n, inst.depot, nb_elite, nb_close);
        let pick = |rng: &mut Rng| {
            let a = rng.below(pop.len());
            let b = rng.below(pop.len());
            if fit[a] <= fit[b] {
                a
            } else {
                b
            }
        };
        let (p1, p2) = (pick(rng), pick(rng));
        let child = order_crossover(&pop[p1].tour, &pop[p2].tour, rng);
        add(&mut pop, child, it, &mut best, rng);
        if pop.len() >= mu + lambda {
            // survivor selection: drop clones first, then the worst biased fitness; keep the best
            while pop.len() > mu {
                let fit = biased_fitness(&pop, n, inst.depot, nb_elite, nb_close);
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
    }
    best
}

/// Exact solution by dynamic programming over subsets: optimal tours of
/// every subset (Held and Karp 1962) and an optimal partition into feasible
/// routes. For at most [`EXACT_MAX_UNITS`] units. Returns `None` when the
/// limits cannot be met.
pub(crate) fn exact(inst: &Instance, units: &[usize]) -> Option<(Vec<Vec<usize>>, f64)> {
    let n = units.len();
    if n == 0 || n > EXACT_MAX_UNITS {
        return None;
    }
    let d = inst.depot;
    let full = (1usize << n) - 1;
    let size = 1usize << n;
    let inf = f64::INFINITY;
    // dp[s * n + j]: cheapest path depot -> units of s -> units[j] (j in s)
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
    // best tour of each subset and its feasibility
    let mut tour = vec![inf; size];
    let mut last = vec![usize::MAX; size];
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
        for j in 0..n {
            if s & (1 << j) != 0 {
                let v = dp[s * n + j] + inst.c(units[j], d);
                if v < tour[s] {
                    tour[s] = v;
                    last[s] = j;
                }
            }
        }
        ok[s] = stops <= inst.limits.max_stops
            && load <= inst.limits.capacity + 1e-9
            && tour[s] + served <= inst.limits.max_length + 1e-9;
    }
    // partition
    let mut g = vec![inf; size];
    let mut gpart = vec![0usize; size];
    g[0] = 0.0;
    for s in 1..size {
        let low = s & s.wrapping_neg();
        let mut r = s;
        while r > 0 {
            if r & low != 0 && ok[r] && g[s ^ r].is_finite() {
                let v = tour[r] + g[s ^ r];
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
    let mut s = full;
    while s > 0 {
        let r = gpart[s];
        // reconstruct the tour of r
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
        s ^= r;
    }
    Some((routes, g[full]))
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
            let (routes, cost) = exact(&inst, &units).unwrap();
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
            let (_, hcost, _) = hgs(&inst, &units, 100, 0.3, f64::INFINITY, &mut rng).unwrap();
            assert!(
                hcost <= cost + 1e-9,
                "seed {seed}: hgs {hcost} > exact {cost}"
            );
        }
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
            let (_, opt) = exact(&inst, &units).unwrap();
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
