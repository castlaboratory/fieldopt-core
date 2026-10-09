//! Travel matrices from coordinates.
//!
//! A [`Matrix`] is a dense, row-major `n x n` matrix of travel distances or
//! times. Users with a road-network matrix (for example from OSRM) build one
//! with [`Matrix::from_vec`]; otherwise [`haversine_matrix`] (great-circle
//! distance in kilometres from latitude and longitude in degrees) or
//! [`euclidean_matrix`] (planar coordinates in any unit) give a first
//! approximation.

/// Dense square matrix of travel costs between units.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    n: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Build from a row-major vector of length `n * n`.
    pub fn from_vec(n: usize, data: Vec<f64>) -> Result<Self, String> {
        if data.len() != n * n {
            return Err(format!(
                "expected {} values for an {n} x {n} matrix, got {}",
                n * n,
                data.len()
            ));
        }
        if data.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("travel costs must be finite and non-negative".into());
        }
        Ok(Matrix { n, data })
    }

    /// Number of units.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Travel cost from `i` to `j`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    /// The underlying row-major data.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}

/// Earth radius used by [`haversine_matrix`], in kilometres.
pub const EARTH_RADIUS_KM: f64 = 6371.0088;

/// Great-circle distances in kilometres between points given by latitude and
/// longitude in degrees.
pub fn haversine_matrix(lat: &[f64], lon: &[f64]) -> Result<Matrix, String> {
    if lat.len() != lon.len() {
        return Err("latitude and longitude must have the same length".into());
    }
    if lat.iter().any(|v| !(-90.0..=90.0).contains(v))
        || lon.iter().any(|v| !(-180.0..=180.0).contains(v))
    {
        return Err("latitude must be in [-90, 90] and longitude in [-180, 180] degrees".into());
    }
    let n = lat.len();
    let (la, lo): (Vec<f64>, Vec<f64>) = (
        lat.iter().map(|v| v.to_radians()).collect(),
        lon.iter().map(|v| v.to_radians()).collect(),
    );
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dlat = la[j] - la[i];
            let dlon = lo[j] - lo[i];
            let a =
                (dlat / 2.0).sin().powi(2) + la[i].cos() * la[j].cos() * (dlon / 2.0).sin().powi(2);
            let d = 2.0 * EARTH_RADIUS_KM * a.sqrt().asin();
            data[i * n + j] = d;
            data[j * n + i] = d;
        }
    }
    Ok(Matrix { n, data })
}

/// Euclidean distances between planar points.
pub fn euclidean_matrix(x: &[f64], y: &[f64]) -> Result<Matrix, String> {
    if x.len() != y.len() {
        return Err("x and y must have the same length".into());
    }
    if x.iter().chain(y.iter()).any(|v| !v.is_finite()) {
        return Err("coordinates must be finite".into());
    }
    let n = x.len();
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = ((x[i] - x[j]).powi(2) + (y[i] - y[j]).powi(2)).sqrt();
            data[i * n + j] = d;
            data[j * n + i] = d;
        }
    }
    Ok(Matrix { n, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haversine_known_distance() {
        // Recife (-8.05, -34.9) to Salvador (-12.97, -38.51): about 675 km
        let m = haversine_matrix(&[-8.05, -12.97], &[-34.9, -38.51]).unwrap();
        assert!((m.get(0, 1) - 675.0).abs() < 5.0);
        assert_eq!(m.get(0, 1), m.get(1, 0));
        assert_eq!(m.get(0, 0), 0.0);
    }

    #[test]
    fn euclidean_and_validation() {
        let m = euclidean_matrix(&[0.0, 3.0], &[0.0, 4.0]).unwrap();
        assert_eq!(m.get(0, 1), 5.0);
        assert!(Matrix::from_vec(2, vec![0.0, 1.0, 1.0]).is_err());
        assert!(Matrix::from_vec(2, vec![0.0, -1.0, 1.0, 0.0]).is_err());
        assert!(haversine_matrix(&[95.0], &[0.0]).is_err());
    }
}
