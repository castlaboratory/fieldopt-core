//! A uniform grid over planar points for nearest-neighbour queries with
//! removals, used by the local pivotal method and the spatial balance
//! measure so that both scale almost linearly with the number of units.

/// Grid index over the points `0..n` of a planar coordinate matrix.
pub(crate) struct GridIndex {
    g: usize,
    x0: f64,
    y0: f64,
    cell: f64,
    cells: Vec<Vec<usize>>,
    /// Cell of each point and its position in that cell (`usize::MAX` once removed).
    pos: Vec<(usize, usize)>,
    xy: Vec<(f64, f64)>,
}

impl GridIndex {
    /// Build over the points listed in `members` (indices into the `n x 2`
    /// row-major `coords`).
    pub fn new(coords: &[f64], members: &[usize]) -> Self {
        let n = coords.len() / 2;
        let xy: Vec<(f64, f64)> = (0..n).map(|i| (coords[2 * i], coords[2 * i + 1])).collect();
        let (mut x0, mut y0, mut x1, mut y1) = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        for &i in members {
            let (x, y) = xy[i];
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        if members.is_empty() {
            x0 = 0.0;
            y0 = 0.0;
            x1 = 1.0;
            y1 = 1.0;
        }
        let range = (x1 - x0).max(y1 - y0).max(1e-12);
        let g = ((members.len() as f64 / 4.0).sqrt().ceil() as usize).max(1);
        let cell = range / g as f64 * (1.0 + 1e-9);
        let mut index = GridIndex {
            g,
            x0,
            y0,
            cell,
            cells: vec![Vec::new(); g * g],
            pos: vec![(usize::MAX, usize::MAX); n],
            xy,
        };
        for &i in members {
            let c = index.cell_of(i);
            index.pos[i] = (c, index.cells[c].len());
            index.cells[c].push(i);
        }
        index
    }

    #[inline]
    fn cell_coords(&self, x: f64, y: f64) -> (usize, usize) {
        let cx = (((x - self.x0) / self.cell).floor().max(0.0) as usize).min(self.g - 1);
        let cy = (((y - self.y0) / self.cell).floor().max(0.0) as usize).min(self.g - 1);
        (cx, cy)
    }

    #[inline]
    fn cell_of(&self, i: usize) -> usize {
        let (x, y) = self.xy[i];
        let (cx, cy) = self.cell_coords(x, y);
        cy * self.g + cx
    }

    /// Remove point `i` from the index.
    pub fn remove(&mut self, i: usize) {
        let (c, k) = self.pos[i];
        if c == usize::MAX {
            return;
        }
        let last = *self.cells[c].last().unwrap();
        self.cells[c].swap_remove(k);
        if last != i {
            self.pos[last] = (c, k);
        }
        self.pos[i] = (usize::MAX, usize::MAX);
    }

    /// Nearest indexed points to `(x, y)` other than `exclude`: the squared
    /// distance and every point at that distance (within 1e-12), in
    /// increasing index order. `None` when the index is empty.
    pub fn nearest(&self, x: f64, y: f64, exclude: usize) -> Option<(f64, Vec<usize>)> {
        let (cx, cy) = self.cell_coords(x, y);
        let (cx, cy) = (cx as isize, cy as isize);
        let g = self.g as isize;
        let mut best = f64::INFINITY;
        let mut ties: Vec<usize> = Vec::new();
        let mut r: isize = 0;
        loop {
            let mut any_cell = false;
            for dy in -r..=r {
                let yy = cy + dy;
                if yy < 0 || yy >= g {
                    continue;
                }
                let on_edge_y = dy == -r || dy == r;
                let mut dx = -r;
                while dx <= r {
                    let xx = cx + dx;
                    if xx >= 0 && xx < g {
                        any_cell = true;
                        for &j in &self.cells[(yy * g + xx) as usize] {
                            if j == exclude {
                                continue;
                            }
                            let (px, py) = self.xy[j];
                            let d = (px - x) * (px - x) + (py - y) * (py - y);
                            if d < best - 1e-12 {
                                best = d;
                                ties.clear();
                                ties.push(j);
                            } else if (d - best).abs() <= 1e-12 {
                                ties.push(j);
                            }
                        }
                    }
                    // inside the ring only the two edge columns belong to it
                    dx += if on_edge_y || dx == r { 1 } else { 2 * r };
                }
            }
            // anything in ring r + 1 is at least r * cell away from the query
            let reach = r as f64 * self.cell;
            if (best.is_finite() && best < reach * reach - 1e-12)
                || (!any_cell && r > 0)
                || r > 2 * g
            {
                break;
            }
            r += 1;
        }
        if ties.is_empty() {
            None
        } else {
            ties.sort_unstable();
            Some((best, ties))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn nearest_matches_brute_force_with_removals() {
        let mut rng = Rng::new(3);
        let n = 500;
        let coords: Vec<f64> = (0..2 * n)
            .map(|_| (rng.next_f64() * 50.0).floor())
            .collect(); // many ties
        let members: Vec<usize> = (0..n).collect();
        let mut index = GridIndex::new(&coords, &members);
        let mut alive = vec![true; n];
        for step in 0..400 {
            let q = rng.below(n);
            let (qx, qy) = (coords[2 * q], coords[2 * q + 1]);
            let mut best = f64::INFINITY;
            let mut ties = Vec::new();
            for j in 0..n {
                if !alive[j] || j == q {
                    continue;
                }
                let d = (coords[2 * j] - qx).powi(2) + (coords[2 * j + 1] - qy).powi(2);
                if d < best - 1e-12 {
                    best = d;
                    ties.clear();
                    ties.push(j);
                } else if (d - best).abs() <= 1e-12 {
                    ties.push(j);
                }
            }
            match index.nearest(qx, qy, q) {
                None => assert!(ties.is_empty()),
                Some((b, t)) => {
                    assert!((b - best).abs() < 1e-9, "step {step}");
                    assert_eq!(t, ties, "step {step}");
                }
            }
            let r = rng.below(n);
            if alive[r] {
                alive[r] = false;
                index.remove(r);
            }
        }
    }
}
