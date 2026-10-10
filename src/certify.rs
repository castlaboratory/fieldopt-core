//! Proof of optimality for a single tour: branch-and-bound on Held-Karp
//! 1-trees (Held and Karp 1971; Volgenant and Jonker 1982). Each node of the
//! search fixes some edges in the tour and forbids others; its bound is the
//! best 1-tree under those constraints, improved by subgradient ascent on
//! the node penalties. Branching takes a node of degree three or more in the
//! 1-tree and either forbids one of its free edges or forces it. A 1-tree
//! that is itself a tour gives an upper bound. The search stops when the
//! bound of every open node reaches the best tour, within a relative
//! tolerance, or when it runs out of nodes or time.
//!
//! Only symmetric matrices are handled (the 1-tree bound needs them).

use crate::distance::Matrix;

/// Outcome of a certification run.
#[derive(Clone, Debug)]
pub struct Certificate {
    /// Best lower bound on any tour through the depot and the units.
    pub lower_bound: f64,
    /// Whether `tour` is proven optimal within the tolerance.
    pub optimal: bool,
    /// The best tour found (unit indices, depot excluded), when the search
    /// improved on the given upper bound or confirmed it.
    pub tour: Option<Vec<usize>>,
    /// Cost of the best tour known.
    pub upper_bound: f64,
    /// Nodes of the search tree explored.
    pub nodes: usize,
}

/// A 1-tree: Lagrangian value, degrees and edges.
type OneTree = (f64, Vec<usize>, Vec<(usize, usize)>);
/// Result of an ascent: best bound and the edges of a 1-tree that is a tour.
type Ascent = (f64, Option<Vec<(usize, usize)>>);

struct Problem<'a> {
    m: &'a Matrix,
    nodes: Vec<usize>,
    k: usize,
}

impl<'a> Problem<'a> {
    #[inline]
    fn c(&self, i: usize, j: usize) -> f64 {
        self.m.get(self.nodes[i], self.nodes[j])
    }
}

/// Edge status in a search node: 0 free, 1 forced, 2 forbidden.
#[derive(Clone)]
struct Constraints {
    status: Vec<u8>, // k x k
    k: usize,
}

impl Constraints {
    fn new(k: usize) -> Self {
        Constraints {
            status: vec![0; k * k],
            k,
        }
    }
    #[inline]
    fn get(&self, i: usize, j: usize) -> u8 {
        self.status[i * self.k + j]
    }
    fn set(&mut self, i: usize, j: usize, s: u8) {
        self.status[i * self.k + j] = s;
        self.status[j * self.k + i] = s;
    }
    fn forced_degree(&self, i: usize) -> usize {
        (0..self.k)
            .filter(|&j| j != i && self.get(i, j) == 1)
            .count()
    }
}

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
        }
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        self.parent[ra] = rb;
        true
    }
}

/// One 1-tree under the constraints with node penalties `pi`: its Lagrangian
/// value, the degrees, and the edges. `None` when the constraints admit no
/// 1-tree.
fn one_tree(p: &Problem, cons: &Constraints, pi: &[f64]) -> Option<OneTree> {
    let k = p.k;
    let cost = |i: usize, j: usize| p.c(i, j) + pi[i] + pi[j];
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(k - 1 + 2);
    let mut deg = vec![0usize; k];
    let mut value = 0.0;
    // spanning tree on nodes 1..k: forced edges first, then Kruskal on the rest
    let mut uf = UnionFind::new(k);
    let mut tree_edges = 0usize;
    for i in 1..k {
        for j in (i + 1)..k {
            if cons.get(i, j) == 1 {
                if !uf.union(i, j) {
                    return None; // forced edges form a cycle
                }
                edges.push((i, j));
                deg[i] += 1;
                deg[j] += 1;
                value += cost(i, j);
                tree_edges += 1;
            }
        }
    }
    let mut cand: Vec<(f64, usize, usize)> = Vec::with_capacity((k - 1) * (k - 2) / 2);
    for i in 1..k {
        for j in (i + 1)..k {
            if cons.get(i, j) == 0 {
                cand.push((cost(i, j), i, j));
            }
        }
    }
    cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    for (c, i, j) in cand {
        if tree_edges == k - 2 {
            break;
        }
        if uf.union(i, j) {
            edges.push((i, j));
            deg[i] += 1;
            deg[j] += 1;
            value += c;
            tree_edges += 1;
        }
    }
    if tree_edges != k - 2 {
        return None; // forbidden edges disconnect the units
    }
    // two edges at the depot (node 0): forced ones first, then the cheapest free ones
    let mut depot_edges: Vec<usize> = (1..k).filter(|&j| cons.get(0, j) == 1).collect();
    if depot_edges.len() > 2 {
        return None;
    }
    let mut free: Vec<(f64, usize)> = (1..k)
        .filter(|&j| cons.get(0, j) == 0)
        .map(|j| (cost(0, j), j))
        .collect();
    free.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    for (_, j) in free {
        if depot_edges.len() == 2 {
            break;
        }
        depot_edges.push(j);
    }
    if depot_edges.len() < 2 {
        return None;
    }
    for &j in &depot_edges {
        edges.push((0, j));
        deg[0] += 1;
        deg[j] += 1;
        value += cost(0, j);
    }
    let lagrangian = value - 2.0 * pi.iter().sum::<f64>();
    Some((lagrangian, deg, edges))
}

/// Subgradient ascent from `pi` under the constraints; returns the best bound,
/// the penalties reached, and a tour (ordered node indices, depot first) if
/// some 1-tree was one.
fn ascent(
    p: &Problem,
    cons: &Constraints,
    pi: &mut [f64],
    upper: f64,
    iterations: usize,
) -> Option<Ascent> {
    let k = p.k;
    let mut best = f64::NEG_INFINITY;
    let mut best_pi = pi.to_vec();
    let mut tour_edges: Option<Vec<(usize, usize)>> = None;
    let mut lambda = 2.0;
    let mut since = 0;
    for _ in 0..iterations {
        let (w, deg, edges) = one_tree(p, cons, pi)?;
        if w > best + 1e-9 {
            best = w;
            best_pi.copy_from_slice(pi);
            since = 0;
        } else {
            since += 1;
            if since >= 10 {
                lambda /= 2.0;
                since = 0;
            }
        }
        let norm: f64 = deg.iter().map(|&d| (d as f64 - 2.0).powi(2)).sum();
        if norm == 0.0 {
            // a tour: its real cost is an upper bound
            let real: f64 = edges.iter().map(|&(i, j)| p.c(i, j)).sum();
            tour_edges = Some(edges);
            best = best.max(w);
            let _ = real;
            break;
        }
        if best >= upper - 1e-9 || lambda < 1e-3 {
            break;
        }
        let t = lambda * (upper - w).max(1e-9) / norm;
        for i in 0..k {
            pi[i] += t * (deg[i] as f64 - 2.0);
        }
    }
    pi.copy_from_slice(&best_pi);
    Some((best, tour_edges))
}

/// Order the edges of a tour (as a 1-tree whose degrees are all 2) into a
/// sequence starting at node 0, returned as unit indices (node 0 excluded).
fn order_tour(k: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut adj = vec![Vec::new(); k];
    for &(i, j) in edges {
        adj[i].push(j);
        adj[j].push(i);
    }
    let mut seq = Vec::with_capacity(k - 1);
    let (mut prev, mut cur) = (0usize, adj[0][0]);
    while cur != 0 {
        seq.push(cur);
        let next = if adj[cur][0] == prev {
            adj[cur][1]
        } else {
            adj[cur][0]
        };
        prev = cur;
        cur = next;
    }
    seq
}

/// Prove (or improve) a tour through `depot` and `units` on a symmetric
/// matrix. `upper` is the cost of the best known tour and `tour` that tour
/// (unit indices); `max_nodes` and `time_limit` (seconds) bound the search;
/// `tolerance` is the relative gap at which a tour counts as optimal.
pub fn certify_tour(
    m: &Matrix,
    depot: usize,
    units: &[usize],
    tour: &[usize],
    upper: f64,
    max_nodes: usize,
    time_limit: f64,
    tolerance: f64,
) -> Certificate {
    let nodes: Vec<usize> = std::iter::once(depot)
        .chain(units.iter().copied())
        .collect();
    let k = nodes.len();
    let mut best_upper = upper;
    let mut best_tour: Option<Vec<usize>> = None;
    if k <= 3 {
        return Certificate {
            lower_bound: upper,
            optimal: true,
            tour: Some(tour.to_vec()),
            upper_bound: upper,
            nodes: 0,
        };
    }
    let p = Problem {
        m,
        nodes: nodes.clone(),
        k,
    };
    let start = std::time::Instant::now();
    let unit_of = |node: usize| p.nodes[node]; // node index -> matrix index
                                               // root
    let mut pi = vec![0.0; k];
    let root = Constraints::new(k);
    let mut explored = 0usize;
    let mut global_lb = f64::INFINITY; // min bound over open nodes (tracked through the stack)
    let mut stack: Vec<(Constraints, Vec<f64>, f64)> = Vec::new();
    match ascent(&p, &root, &mut pi, best_upper, 300) {
        None => {
            return Certificate {
                lower_bound: upper,
                optimal: false,
                tour: None,
                upper_bound: upper,
                nodes: 0,
            };
        }
        Some((lb, tour_edges)) => {
            if let Some(e) = tour_edges {
                let seq = order_tour(k, &e);
                let real: f64 = e.iter().map(|&(i, j)| p.c(i, j)).sum();
                if real < best_upper - 1e-9 {
                    best_upper = real;
                    best_tour = Some(seq.iter().map(|&n| unit_of(n)).collect());
                }
            }
            stack.push((root, pi.clone(), lb));
        }
    }
    let tol = |ub: f64| tolerance * ub.max(1.0);
    let mut pruned_bound = f64::INFINITY; // smallest bound among pruned/unexplored nodes
    while let Some((cons, pi0, lb)) = stack.pop() {
        if lb >= best_upper - tol(best_upper) {
            pruned_bound = pruned_bound.min(lb);
            continue;
        }
        if explored >= max_nodes || start.elapsed().as_secs_f64() > time_limit {
            // out of budget: everything left is open
            global_lb = global_lb.min(lb);
            for (_, _, l) in &stack {
                global_lb = global_lb.min(*l);
            }
            break;
        }
        explored += 1;
        let mut pi = pi0;
        let (bound, tour_edges, deg, edges) = match ascent(&p, &cons, &mut pi, best_upper, 60) {
            None => continue,
            Some((b, t)) => {
                let (_, deg, edges) = match one_tree(&p, &cons, &pi) {
                    Some(x) => x,
                    None => continue,
                };
                (b, t, deg, edges)
            }
        };
        if let Some(e) = tour_edges {
            let real: f64 = e.iter().map(|&(i, j)| p.c(i, j)).sum();
            if real < best_upper - 1e-9 {
                best_upper = real;
                best_tour = Some(order_tour(k, &e).iter().map(|&n| unit_of(n)).collect());
            }
            continue; // the subproblem is solved: its best 1-tree is a tour
        }
        if bound >= best_upper - tol(best_upper) {
            pruned_bound = pruned_bound.min(bound);
            continue;
        }
        // branch on a node of largest degree in the 1-tree: forbid / force one of its free edges
        let v = (0..k).max_by_key(|&i| deg[i]).unwrap();
        if deg[v] <= 2 {
            continue;
        }
        let free_edges: Vec<(usize, usize)> = edges
            .iter()
            .copied()
            .filter(|&(i, j)| (i == v || j == v) && cons.get(i, j) == 0)
            .collect();
        if free_edges.is_empty() {
            continue;
        }
        // the free edge of v with the largest cost is forbidden in one child and forced in the other
        let &(i, j) = free_edges
            .iter()
            .max_by(|a, b| {
                p.c(a.0, a.1)
                    .partial_cmp(&p.c(b.0, b.1))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let mut forced = cons.clone();
        forced.set(i, j, 1);
        let mut forbidden = cons.clone();
        forbidden.set(i, j, 2);
        // if forcing gives a node two forced edges, forbid its other free edges
        for &x in &[i, j] {
            if forced.forced_degree(x) >= 2 {
                for y in 0..k {
                    if y != x && forced.get(x, y) == 0 {
                        forced.set(x, y, 2);
                    }
                }
            }
        }
        // explore the forced child first (depth first, better bound expected)
        stack.push((forbidden, pi.clone(), bound));
        stack.push((forced, pi, bound));
    }
    let lower_bound = if stack.is_empty() && global_lb.is_infinite() {
        best_upper
            .min(pruned_bound.max(0.0))
            .max(0.0)
            .min(best_upper)
    } else {
        global_lb.min(best_upper)
    };
    let optimal = stack.is_empty() && global_lb.is_infinite();
    Certificate {
        lower_bound: if optimal { best_upper } else { lower_bound },
        optimal,
        tour: best_tour.or(if optimal { Some(tour.to_vec()) } else { None }),
        upper_bound: best_upper,
        nodes: explored,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::euclidean_matrix;
    use crate::rng::Rng;
    use crate::routing::{grasp_routes, GraspOptions, RouteLimits};
    use crate::vrp::{exact, Instance};

    fn random(n: usize, seed: u64) -> Matrix {
        let mut rng = Rng::new(seed);
        let x: Vec<f64> = (0..=n).map(|_| rng.next_f64() * 100.0).collect();
        let y: Vec<f64> = (0..=n).map(|_| rng.next_f64() * 100.0).collect();
        euclidean_matrix(&x, &y).unwrap()
    }

    #[test]
    fn certificate_agrees_with_the_exact_optimum() {
        for seed in 0..10u64 {
            let m = random(12, seed);
            let units: Vec<usize> = (1..13).collect();
            let inst = Instance::new(
                &m,
                0,
                &units,
                vec![0.0; 13],
                vec![0.0; 13],
                RouteLimits::default(),
                true,
            );
            let (_, _, opt) = exact(&inst, &units).unwrap();
            // a deliberately poor upper bound: the identity order
            let tour = units.clone();
            let cost: f64 = crate::routing::route_length(&m, 0, &tour);
            let cert = certify_tour(&m, 0, &units, &tour, cost, 100_000, 10.0, 1e-9);
            assert!(
                cert.optimal,
                "seed {seed}: not proven, nodes {}",
                cert.nodes
            );
            assert!(
                (cert.upper_bound - opt).abs() < 1e-6,
                "seed {seed}: {} vs {opt}",
                cert.upper_bound
            );
            let t = cert.tour.unwrap();
            assert!((crate::routing::route_length(&m, 0, &t) - opt).abs() < 1e-6);
        }
    }

    #[test]
    fn certifies_heuristic_tours_of_moderate_size() {
        for (n, seed) in [(20usize, 1u64), (30, 2), (40, 3)] {
            let m = random(n, seed);
            let units: Vec<usize> = (1..=n).collect();
            let o = GraspOptions {
                iterations: 500,
                time_limit: f64::INFINITY,
                certify_time: 0.0,
                alpha: 0.3,
                seed: 1,
            };
            let sol = grasp_routes(&m, 0, &units, RouteLimits::default(), o).unwrap();
            let cert = certify_tour(
                &m,
                0,
                &units,
                &sol.routes[0],
                sol.total,
                200_000,
                20.0,
                1e-7,
            );
            assert!(cert.lower_bound <= sol.total + 1e-6);
            assert!(
                cert.optimal,
                "n {n}: not proven after {} nodes, bound {} vs {}",
                cert.nodes, cert.lower_bound, cert.upper_bound
            );
            assert!(cert.upper_bound <= sol.total + 1e-9);
        }
    }
}
