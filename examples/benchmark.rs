//! Benchmark of the routing solver on TSPLIB (EUC_2D, single tour) and
//! CVRPLIB set A (Augerat et al., capacity) instances.
//!
//! ```text
//! cargo run --release --example benchmark -- <dir with .tsp/.vrp/.mdvrp files> [iterations] [time limit in seconds] [seed]
//! ```
//! Distances are rounded to the nearest integer as the libraries do, so the
//! costs compare directly with the published optima (the `Optimal value`
//! comment of the .vrp files, or the table below for TSPLIB).

use fieldopt_core::distance::Matrix;
use fieldopt_core::routing::{solve_fleet, solve_routes, Fleet, GraspOptions, RouteLimits};
use std::time::Instant;

const TSPLIB_OPTIMA: &[(&str, f64)] = &[
    ("eil51", 426.0),
    ("berlin52", 7542.0),
    ("st70", 675.0),
    ("eil76", 538.0),
    ("rat99", 1211.0),
    ("kroA100", 21282.0),
    ("eil101", 629.0),
];

struct Parsed {
    name: String,
    coords: Vec<(f64, f64)>,
    demand: Vec<f64>,
    service: Vec<f64>,
    capacity: f64,
    max_duration: f64,
    depots: Vec<usize>,
    routes_per_depot: usize,
    optimum: Option<f64>,
    rounded: bool,
}

/// Cordeau's MDVRP format: `type m n t`, then `t` lines `D Q`, then `n`
/// customers `i x y d q ...`, then `t` depots `i x y ...`.
fn parse_cordeau(path: &std::path::Path) -> Parsed {
    let text = std::fs::read_to_string(path).expect("read instance");
    let name = path.file_stem().unwrap().to_string_lossy().to_string();
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let head: Vec<f64> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    let (m, n, t) = (head[1] as usize, head[2] as usize, head[3] as usize);
    let (mut max_duration, mut capacity) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..t {
        let v: Vec<f64> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .filter_map(|x| x.parse().ok())
            .collect();
        if v[0] > 0.0 {
            max_duration = v[0];
        }
        capacity = v[1];
    }
    let mut coords = Vec::new();
    let mut demand = Vec::new();
    let mut service = Vec::new();
    for _ in 0..n {
        let v: Vec<f64> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .filter_map(|x| x.parse().ok())
            .collect();
        coords.push((v[1], v[2]));
        service.push(v[3]);
        demand.push(v[4]);
    }
    let mut depots = Vec::new();
    for _ in 0..t {
        let v: Vec<f64> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .filter_map(|x| x.parse().ok())
            .collect();
        depots.push(coords.len());
        coords.push((v[1], v[2]));
        service.push(0.0);
        demand.push(0.0);
    }
    let optimum = MDVRP_BEST
        .iter()
        .find(|(nm, _)| *nm == name)
        .map(|(_, v)| *v);
    Parsed {
        name,
        coords,
        demand,
        service,
        capacity,
        max_duration,
        depots,
        routes_per_depot: m,
        optimum,
        rounded: false,
    }
}

/// Best known solutions of Cordeau's MDVRP instances (Euclidean distances,
/// not rounded) as distributed with the instances by the NEO group
/// (neo.lcc.uma.es/vrp, file C-mdvrp-sol.zip).
const MDVRP_BEST: &[(&str, f64)] = &[
    ("p01", 576.87),
    ("p02", 473.53),
    ("p03", 641.19),
    ("p04", 1001.59),
    ("p05", 750.03),
    ("p06", 876.50),
    ("p07", 885.80),
    ("p08", 4437.68),
    ("p09", 3900.22),
    ("p10", 3663.02),
    ("p11", 3554.18),
    ("p12", 1318.95),
    ("p13", 1318.95),
    ("p14", 1360.12),
    ("p15", 2505.42),
    ("p16", 2572.23),
    ("p17", 2709.09),
    ("p18", 3702.85),
    ("p19", 3827.06),
    ("p20", 4058.07),
    ("p21", 5474.84),
    ("p22", 5702.16),
    ("p23", 6095.46),
];

fn parse(path: &std::path::Path) -> Parsed {
    let text = std::fs::read_to_string(path).expect("read instance");
    let mut name = path.file_stem().unwrap().to_string_lossy().to_string();
    let mut coords = Vec::new();
    let mut demand = Vec::new();
    let mut capacity = f64::INFINITY;
    let mut depot = 0usize;
    let mut optimum = None;
    let mut section = "";
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        if let Some(v) = l.strip_prefix("NAME") {
            name = v.trim_start_matches([':', ' ']).trim().to_string();
        } else if l.starts_with("COMMENT") {
            if let Some(i) = l.find("Optimal value:") {
                let rest: String = l[i + 14..]
                    .chars()
                    .filter(|c| c.is_ascii_digit() || *c == '.')
                    .collect();
                optimum = rest.parse().ok();
            }
        } else if let Some(v) = l.strip_prefix("CAPACITY") {
            capacity = v.trim_start_matches([':', ' ']).trim().parse().unwrap();
        } else if l.starts_with("NODE_COORD_SECTION") {
            section = "coord";
        } else if l.starts_with("DEMAND_SECTION") {
            section = "demand";
        } else if l.starts_with("DEPOT_SECTION") {
            section = "depot";
        } else if l.starts_with("EOF") {
            break;
        } else if l.contains(':') {
            continue;
        } else {
            let nums: Vec<f64> = l
                .split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect();
            match section {
                "coord" if nums.len() >= 3 => coords.push((nums[1], nums[2])),
                "demand" if nums.len() >= 2 => demand.push(nums[1]),
                "depot" if nums.len() == 1 && nums[0] >= 1.0 => depot = nums[0] as usize - 1,
                _ => {}
            }
        }
    }
    if demand.is_empty() {
        demand = vec![0.0; coords.len()];
    }
    if optimum.is_none() {
        optimum = TSPLIB_OPTIMA
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| *v);
    }
    let n = coords.len();
    Parsed {
        name,
        coords,
        demand,
        service: vec![0.0; n],
        capacity,
        max_duration: f64::INFINITY,
        depots: vec![depot],
        routes_per_depot: usize::MAX,
        optimum,
        rounded: true,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).expect("directory with instances");
    let iterations: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(2000);
    let seed: u64 = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(1);
    let time_limit: f64 = args
        .get(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(f64::INFINITY);
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .expect("read dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("tsp") | Some("vrp") | Some("mdvrp")
            )
        })
        .collect();
    files.sort();
    println!(
        "{:<12} {:>5} {:>7} {:>9} {:>9} {:>7} {:>8} {:>6}",
        "instance", "n", "routes", "cost", "optimum", "gap%", "bound%", "sec"
    );
    for f in files {
        let p = if f.extension().and_then(|e| e.to_str()) == Some("mdvrp") {
            parse_cordeau(&f)
        } else {
            parse(&f)
        };
        let n = p.coords.len();
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let d = ((p.coords[i].0 - p.coords[j].0).powi(2)
                    + (p.coords[i].1 - p.coords[j].1).powi(2))
                .sqrt();
                data[i * n + j] = if p.rounded { d.round() } else { d };
            }
        }
        let m = Matrix::from_vec(n, data).unwrap();
        let units: Vec<usize> = (0..n).filter(|&i| !p.depots.contains(&i)).collect();
        let limits = RouteLimits {
            max_length: p.max_duration,
            max_stops: usize::MAX,
            capacity: p.capacity,
        };
        let fleet = Fleet {
            depots: p.depots.clone(),
            routes_available: vec![p.routes_per_depot; p.depots.len()],
            fixed_cost: 0.0,
        };
        let options = GraspOptions {
            iterations,
            time_limit,
            alpha: 0.3,
            seed,
        };
        let start = Instant::now();
        let sol = if p.depots.len() == 1 && p.routes_per_depot == usize::MAX {
            solve_routes(
                &m,
                p.depots[0],
                &units,
                &p.service,
                &p.demand,
                limits,
                options,
            )
            .unwrap()
        } else {
            solve_fleet(&m, &fleet, &units, &p.service, &p.demand, limits, options).unwrap()
        };
        let secs = start.elapsed().as_secs_f64();
        let gap = p.optimum.map(|o| 100.0 * (sol.total - o) / o);
        let bound_gap = 100.0 * (sol.total - sol.lower_bound) / sol.total;
        println!(
            "{:<12} {:>5} {:>7} {:>9.2} {:>9} {:>7} {:>8.2} {:>6.1}",
            p.name,
            units.len(),
            sol.routes.len(),
            sol.total,
            p.optimum.map_or("?".to_string(), |o| format!("{o:.2}")),
            gap.map_or("?".to_string(), |g| format!("{g:.2}")),
            bound_gap,
            secs
        );
    }
}
