# fieldopt-core

Field-survey design under real field cost, in pure Rust with no dependencies:

- **Travel matrices** from coordinates (great-circle or Euclidean), or supplied
  by the user from a road network (for example OSRM).
- **Routing of field work** with GRASP (randomised greedy construction, 2-opt
  and Or-opt local search) for one tour or several routes under a maximum
  length and number of stops (optimal split of the giant tour, Prins 2004),
  with a lower bound and the gap to it.
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

## License

GPL-3.0-or-later. © CAST Lab, Universidade Federal de Pernambuco.
