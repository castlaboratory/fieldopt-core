# fieldopt-core

Field-survey design under real field cost, in pure Rust with no dependencies:

- **Travel matrices** from coordinates (great-circle or Euclidean), or supplied
  by the user from a road network (for example OSRM).
- **Routing of field work**: instances of up to 13 units are solved exactly
  (dynamic programming over subsets); larger ones by a hybrid genetic search
  (Vidal 2022) with feasible and infeasible subpopulations under adaptive
  penalties, the optimal split of Prins (2004), and a granular local search
  (2-opt, Or-opt, relocate, swap, SWAP*, 2-opt*) priced from route totals in
  constant time, under limits on route duration (travel plus service time),
  stops and capacity, with one or several depots (each with its own number
  of routes available) and a fixed cost per route, on symmetric or
  asymmetric matrices. Optimal on the
  seven classical TSPLIB instances up to 101 nodes and on 24 of the 27
  Augerat set A CVRP instances (mean gap 0.01%, largest 0.15%) in about two
  seconds each; on Cordeau's 23 multi-depot instances (50 to 360 customers,
  with route duration limits) at or below the best known solution on 13,
  mean gap 0.20%, largest 1.6%, in 1 to 100 seconds; results vary by less
  than one percent across seeds. The Held-Karp bound certifies single
  tours; stops on offspring or on time.
- **Spatially balanced sampling** with the local pivotal method (Grafström,
  Lundström and Schelin 2012) and size-proportional inclusion probabilities.
- **Design-based estimation**: Horvitz-Thompson total with the local-mean
  variance estimator (Grafström and Schelin 2014) and the simple-random-sampling
  variance as a reference.
- **Cost-aware allocation** across strata or frames: minimum cost for a target
  variance, or minimum variance for a budget (Cochran 1977).

Every randomised step is reproducible from a seed. The crate is the engine of
the R package [fieldopt](https://github.com/castlaboratory/fieldopt) (CAST Lab,
Universidade Federal de Pernambuco) and can be used on its own.

```toml
[dependencies]
fieldopt-core = "0.1"
```

See the crate documentation for an end-to-end example (select segments, route
the field work, estimate).


Version 0.2 adds the pieces an area-frame or dual-frame survey needs:

- `spatial`: order of units along a Hilbert curve and systematic PPS
  sampling along it, as independent interpenetrating replicates for
  design-based variance estimation.
- `dualframe`: Hartley's dual-frame allocation (area + list with overlap):
  cost-optimal sample sizes in each frame and the optimal mixing weight
  `theta`, or the screening design (`theta = 0`), for a target variance or
  a budget, with design effects.
- `twostage`: Cochran's two-stage allocation with the cost function
  `c1 n + c2 n m`: optimal secondary units per primary unit and the number
  of primary units for a target variance or a budget.

## License

GPL-3.0-or-later. © CAST Lab, Universidade Federal de Pernambuco.
