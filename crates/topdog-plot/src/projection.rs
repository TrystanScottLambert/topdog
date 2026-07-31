//! All-sky map projections (Aitoff, Mollweide) and graticule generation.
//!
//! Conventions: longitude λ ∈ [-π, π] measured from the map center,
//! latitude φ ∈ [-π/2, π/2]. Callers convert RA/Dec (degrees) to centered
//! radians; the GUI flips x so RA increases leftward, as sky maps expect.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkyProjection {
    Aitoff,
    Mollweide,
}

impl SkyProjection {
    pub const ALL: [SkyProjection; 2] = [SkyProjection::Aitoff, SkyProjection::Mollweide];

    /// Project (λ, φ) radians to map coordinates.
    ///
    /// Aitoff: x ∈ [-π, π], y ∈ [-π/2, π/2].
    /// Mollweide: x ∈ [-2√2, 2√2], y ∈ [-√2, √2].
    /// Both are 2:1 ellipses, so the GUI can scale either by
    /// [`Self::extent`] without caring which is active.
    pub fn project(&self, lon: f64, lat: f64) -> (f64, f64) {
        match self {
            SkyProjection::Aitoff => {
                let half = lon / 2.0;
                let alpha = (lat.cos() * half.cos()).acos();
                // sinc(alpha), safe at alpha == 0.
                let sinc = if alpha.abs() < 1e-12 {
                    1.0
                } else {
                    alpha.sin() / alpha
                };
                (2.0 * lat.cos() * half.sin() / sinc, lat.sin() / sinc)
            }
            SkyProjection::Mollweide => {
                // Solve 2θ + sin 2θ = π sin φ by Newton's method.
                let target = std::f64::consts::PI * lat.sin();
                let mut theta = lat;
                for _ in 0..16 {
                    let f = 2.0 * theta + (2.0 * theta).sin() - target;
                    let d = 2.0 + 2.0 * (2.0 * theta).cos();
                    if d.abs() < 1e-12 {
                        break; // at the poles the iteration is already exact
                    }
                    let next = theta - f / d;
                    if (next - theta).abs() < 1e-12 {
                        theta = next;
                        break;
                    }
                    theta = next;
                }
                let sqrt2 = std::f64::consts::SQRT_2;
                (
                    2.0 * sqrt2 / std::f64::consts::PI * lon * theta.cos(),
                    sqrt2 * theta.sin(),
                )
            }
        }
    }

    /// Half-extent (x_max, y_max) of the projected map.
    pub fn extent(&self) -> (f64, f64) {
        match self {
            SkyProjection::Aitoff => (std::f64::consts::PI, std::f64::consts::FRAC_PI_2),
            SkyProjection::Mollweide => (2.0 * std::f64::consts::SQRT_2, std::f64::consts::SQRT_2),
        }
    }

    /// The map boundary (the ±180° meridian), as a closed polyline in
    /// projected coordinates.
    pub fn boundary(&self, samples: usize) -> Vec<(f64, f64)> {
        let pi = std::f64::consts::PI;
        let mut points = Vec::with_capacity(2 * samples + 1);
        // West edge from south to north pole, then east edge back down.
        for i in 0..=samples {
            let lat = -pi / 2.0 + pi * i as f64 / samples as f64;
            points.push(self.project(-pi, lat));
        }
        for i in (0..=samples).rev() {
            let lat = -pi / 2.0 + pi * i as f64 / samples as f64;
            points.push(self.project(pi, lat));
        }
        points
    }

    /// Graticule lines (meridians every `lon_step`°, parallels every
    /// `lat_step`°) as polylines in projected coordinates.
    pub fn graticule(&self, lon_step: f64, lat_step: f64) -> Vec<Vec<(f64, f64)>> {
        let deg = std::f64::consts::PI / 180.0;
        let mut lines = Vec::new();

        let mut lon = -180.0 + lon_step;
        while lon < 180.0 - 1e-9 {
            let line: Vec<(f64, f64)> = (0..=60)
                .map(|i| {
                    let lat = -90.0 + 180.0 * i as f64 / 60.0;
                    self.project(lon * deg, lat * deg)
                })
                .collect();
            lines.push(line);
            lon += lon_step;
        }

        let mut lat = -90.0 + lat_step;
        while lat < 90.0 - 1e-9 {
            let line: Vec<(f64, f64)> = (0..=90)
                .map(|i| {
                    let l = -180.0 + 360.0 * i as f64 / 90.0;
                    self.project(l * deg, lat * deg)
                })
                .collect();
            lines.push(line);
            lat += lat_step;
        }

        lines
    }
}

impl std::fmt::Display for SkyProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SkyProjection::Aitoff => write!(f, "Aitoff"),
            SkyProjection::Mollweide => write!(f, "Mollweide"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI, SQRT_2};

    #[test]
    fn center_maps_to_origin() {
        for p in SkyProjection::ALL {
            let (x, y) = p.project(0.0, 0.0);
            assert!(x.abs() < 1e-12, "{p}: x={x}");
            assert!(y.abs() < 1e-12, "{p}: y={y}");
        }
    }

    #[test]
    fn equator_edges_reach_full_width() {
        let (x, _) = SkyProjection::Aitoff.project(PI, 0.0);
        assert!((x - PI).abs() < 1e-9);
        let (x, _) = SkyProjection::Mollweide.project(PI, 0.0);
        assert!((x - 2.0 * SQRT_2).abs() < 1e-9);
    }

    #[test]
    fn poles_project_to_top_and_bottom() {
        let (x, y) = SkyProjection::Aitoff.project(0.0, FRAC_PI_2);
        assert!(x.abs() < 1e-9);
        assert!((y - FRAC_PI_2).abs() < 1e-9);

        let (x, y) = SkyProjection::Mollweide.project(0.0, FRAC_PI_2);
        assert!(x.abs() < 1e-9);
        assert!((y - SQRT_2).abs() < 1e-6);

        let (_, y) = SkyProjection::Mollweide.project(0.0, -FRAC_PI_2);
        assert!((y + SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn projection_is_symmetric() {
        for p in SkyProjection::ALL {
            let (x1, y1) = p.project(1.0, 0.7);
            let (x2, y2) = p.project(-1.0, 0.7);
            assert!((x1 + x2).abs() < 1e-9);
            assert!((y1 - y2).abs() < 1e-9);
            let (x3, y3) = p.project(1.0, -0.7);
            assert!((x1 - x3).abs() < 1e-9);
            assert!((y1 + y3).abs() < 1e-9);
        }
    }

    #[test]
    fn points_stay_inside_extent() {
        for p in SkyProjection::ALL {
            let (xm, ym) = p.extent();
            for i in 0..50 {
                let lon = -PI + 2.0 * PI * i as f64 / 49.0;
                for j in 0..25 {
                    let lat = -FRAC_PI_2 + PI * j as f64 / 24.0;
                    let (x, y) = p.project(lon, lat);
                    assert!(x.abs() <= xm + 1e-6);
                    assert!(y.abs() <= ym + 1e-6);
                }
            }
        }
    }

    #[test]
    fn graticule_has_expected_line_count() {
        let lines = SkyProjection::Aitoff.graticule(30.0, 30.0);
        // meridians: -150..150 step 30 = 11; parallels: -60..60 step 30 = 5
        assert_eq!(lines.len(), 11 + 5);
        assert!(lines.iter().all(|l| l.len() > 10));
    }

    #[test]
    fn boundary_is_closed_ellipse() {
        let b = SkyProjection::Mollweide.boundary(40);
        assert!(b.len() > 80);
        let first = b.first().unwrap();
        let last = b.last().unwrap();
        assert!((first.0 - last.0).abs() < 1e-9); // both at a pole
    }
}
