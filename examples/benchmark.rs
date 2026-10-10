//! Benchmark of the routing solver on TSPLIB (EUC_2D, single tour) and
//! CVRPLIB set A (Augerat et al., capacity) instances.
//!
//! ```text
//! cargo run --release --example benchmark -- <dir with .tsp/.vrp files> [iterations] [time limit in seconds]
//! ```
//! Distances are rounded to the nearest integer as the libraries do, so the
//! costs compare directly with the published optima (the `Optimal value`
//! comment of the .vrp files, or the table below for TSPLIB).

use fieldopt_core::distance::Matrix;
use fieldopt_core::routing::{solve_routes, GraspOptions, RouteLimits};
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
    capacity: f64,
    depot: usize,
    optimum: Option<f64>,
}

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
    Parsed {
        name,
        coords,
        demand,
        capacity,
        depot,
        optimum,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).expect("directory with instances");
    let iterations: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(2000);
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
                Some("tsp") | Some("vrp")
            )
        })
        .collect();
    files.sort();
    println!(
        "{:<12} {:>5} {:>7} {:>9} {:>9} {:>7} {:>8} {:>6}",
        "instance", "n", "routes", "cost", "optimum", "gap%", "bound%", "sec"
    );
    for f in files {
        let p = parse(&f);
        let n = p.coords.len();
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let d = ((p.coords[i].0 - p.coords[j].0).powi(2)
                    + (p.coords[i].1 - p.coords[j].1).powi(2))
                .sqrt();
                data[i * n + j] = d.round();
            }
        }
        let m = Matrix::from_vec(n, data).unwrap();
        let units: Vec<usize> = (0..n).filter(|&i| i != p.depot).collect();
        let limits = RouteLimits {
            max_length: f64::INFINITY,
            max_stops: usize::MAX,
            capacity: p.capacity,
        };
        let options = GraspOptions {
            iterations,
            time_limit,
            alpha: 0.3,
            seed: 1,
        };
        let start = Instant::now();
        let sol = solve_routes(&m, p.depot, &units, &[], &p.demand, limits, options).unwrap();
        let secs = start.elapsed().as_secs_f64();
        let gap = p.optimum.map(|o| 100.0 * (sol.total - o) / o);
        let bound_gap = 100.0 * (sol.total - sol.lower_bound) / sol.total;
        println!(
            "{:<12} {:>5} {:>7} {:>9.0} {:>9} {:>7} {:>8.2} {:>6.1}",
            p.name,
            units.len(),
            sol.routes.len(),
            sol.total,
            p.optimum.map_or("?".to_string(), |o| format!("{o:.0}")),
            gap.map_or("?".to_string(), |g| format!("{g:.2}")),
            bound_gap,
            secs
        );
    }
}
