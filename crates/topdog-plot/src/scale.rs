//! Scale transforms and tick generation.
//!
//! Hand-rolled because tick quality is a paper-readiness issue: we want the
//! 1-2-5 "nice number" progression journals expect, not whatever a charting
//! library defaults to.

/// Maps a data-space domain onto a screen-space pixel range (linear).
///
/// `range` may be inverted (start > end), which is how y-axes map data-up to
/// screen-down.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearScale {
    pub domain: (f64, f64),
    pub range: (f32, f32),
}

impl LinearScale {
    pub fn new(domain: (f64, f64), range: (f32, f32)) -> Self {
        Self { domain, range }
    }

    pub fn to_screen(&self, v: f64) -> f32 {
        let (d0, d1) = self.domain;
        let (r0, r1) = self.range;
        if d1 == d0 {
            return (r0 + r1) / 2.0;
        }
        let t = (v - d0) / (d1 - d0);
        r0 + (t as f32) * (r1 - r0)
    }

    pub fn from_screen(&self, px: f32) -> f64 {
        let (d0, d1) = self.domain;
        let (r0, r1) = self.range;
        if r1 == r0 {
            return d0;
        }
        let t = ((px - r0) / (r1 - r0)) as f64;
        d0 + t * (d1 - d0)
    }
}

/// Expand a raw data range into padded axis limits.
///
/// 5% padding on each side so markers never sit on the frame; degenerate
/// ranges (single value, empty) get a unit span so plots stay drawable.
pub fn padded_limits(min: f64, max: f64) -> (f64, f64) {
    if !min.is_finite() || !max.is_finite() {
        return (0.0, 1.0);
    }
    if min == max {
        return (min - 0.5, max + 0.5);
    }
    let pad = (max - min) * 0.05;
    (min - pad, max + pad)
}

/// Generate "nice" tick positions covering `[min, max]`, aiming for roughly
/// `target` ticks. Steps follow the 1-2-5 progression.
pub fn nice_ticks(min: f64, max: f64, target: usize) -> Vec<f64> {
    if !(min.is_finite() && max.is_finite()) || min >= max || target == 0 {
        return Vec::new();
    }
    let step = nice_step((max - min) / target as f64);
    let start = (min / step).ceil() * step;
    let mut ticks = Vec::new();
    let mut i = 0u32;
    loop {
        let v = start + step * f64::from(i);
        if v > max + step * 1e-9 {
            break;
        }
        // Snap values that should be zero (avoids "-0" and 1e-17 labels).
        ticks.push(if v.abs() < step * 1e-9 { 0.0 } else { v });
        i += 1;
    }
    ticks
}

/// Round a raw step to the nearest 1/2/5 × 10^k (thresholds follow the
/// classic "nice numbers" graph-labeling heuristic).
fn nice_step(raw: f64) -> f64 {
    let mag = 10f64.powf(raw.abs().log10().floor());
    let norm = raw / mag;
    let factor = if norm < 1.5 {
        1.0
    } else if norm < 3.0 {
        2.0
    } else if norm < 7.0 {
        5.0
    } else {
        10.0
    };
    factor * mag
}

/// Format a tick value with precision appropriate to the tick step.
pub fn format_tick(value: f64, step: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    if !(1e-4..1e5).contains(&value.abs()) {
        return format!("{value:.1e}");
    }
    let decimals = if step > 0.0 {
        (-step.log10().floor()).max(0.0) as usize
    } else {
        0
    };
    format!("{value:.decimals$}")
}

/// The step between consecutive ticks (0 if fewer than two).
pub fn tick_step(ticks: &[f64]) -> f64 {
    if ticks.len() >= 2 {
        ticks[1] - ticks[0]
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_maps_endpoints_and_midpoint() {
        let s = LinearScale::new((0.0, 10.0), (100.0, 200.0));
        assert_eq!(s.to_screen(0.0), 100.0);
        assert_eq!(s.to_screen(10.0), 200.0);
        assert_eq!(s.to_screen(5.0), 150.0);
        assert!((s.from_screen(150.0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn inverted_range_for_y_axis() {
        let s = LinearScale::new((0.0, 1.0), (300.0, 0.0));
        assert_eq!(s.to_screen(0.0), 300.0);
        assert_eq!(s.to_screen(1.0), 0.0);
    }

    #[test]
    fn ticks_are_nice_and_cover_range() {
        let t = nice_ticks(0.0, 10.0, 5);
        assert_eq!(t, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);

        let t = nice_ticks(-1.3, 1.3, 5);
        assert_eq!(t, vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
        assert!(t.contains(&0.0));
    }

    #[test]
    fn ticks_handle_odd_ranges() {
        let t = nice_ticks(17.3, 18.9, 4);
        assert!(!t.is_empty());
        assert!(t.iter().all(|&v| (17.3..=18.9).contains(&v)));

        assert!(nice_ticks(1.0, 1.0, 5).is_empty());
        assert!(nice_ticks(f64::NAN, 1.0, 5).is_empty());
    }

    #[test]
    fn tick_formatting_matches_step() {
        assert_eq!(format_tick(2.0, 0.5), "2.0");
        assert_eq!(format_tick(2.5, 0.5), "2.5");
        assert_eq!(format_tick(2.0, 2.0), "2");
        assert_eq!(format_tick(0.0, 0.5), "0");
        assert_eq!(format_tick(123456.0, 20000.0), "1.2e5");
    }

    #[test]
    fn padded_limits_expand_and_handle_degenerate() {
        let (lo, hi) = padded_limits(0.0, 10.0);
        assert!(lo < 0.0 && hi > 10.0);
        assert_eq!(padded_limits(3.0, 3.0), (2.5, 3.5));
        assert_eq!(padded_limits(f64::NAN, 1.0), (0.0, 1.0));
    }
}
