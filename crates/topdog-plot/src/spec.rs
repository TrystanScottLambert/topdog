//! The plot specification: single source of truth for what a plot draws and
//! how it looks (CLAUDE.md §5). The renderer reads it; every edit — click-
//! to-edit labels, properties panel, future export — writes back into it.

/// RGBA color, 0–1 floats. Own type so this crate stays toolkit-free.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Rgba {
    pub const BLACK: Rgba = Rgba::new(0.0, 0.0, 0.0, 1.0);
    pub const WHITE: Rgba = Rgba::new(1.0, 1.0, 1.0, 1.0);
    /// Muted blue (Paul Tol palette) — legible in print and colorblind-safe.
    pub const STEEL: Rgba = Rgba::new(0.267, 0.467, 0.667, 1.0);

    /// Qualitative palette for plot layers (Paul Tol "bright"):
    /// colorblind-safe and distinguishable in grayscale print.
    pub const PALETTE: [Rgba; 7] = [
        Rgba::new(0.267, 0.467, 0.667, 1.0), // blue
        Rgba::new(0.933, 0.400, 0.467, 1.0), // red
        Rgba::new(0.133, 0.533, 0.200, 1.0), // green
        Rgba::new(0.800, 0.733, 0.267, 1.0), // yellow
        Rgba::new(0.400, 0.800, 0.933, 1.0), // cyan
        Rgba::new(0.667, 0.200, 0.467, 1.0), // purple
        Rgba::new(0.733, 0.733, 0.733, 1.0), // grey
    ];

    /// The i-th layer color, cycling through the palette.
    pub fn layer_color(i: usize) -> Rgba {
        Self::PALETTE[i % Self::PALETTE.len()]
    }

    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
}

/// Per-layer styling for multi-layer (overplotted) figures.
///
/// A plot always has at least one layer; the legend is drawn when there are
/// two or more. Layer labels are click-to-edit like any other plot text.
/// Every visual property is per-layer so overlaid subsamples can be styled
/// apart (different markers, filled vs open, bar vs step histograms, ...).
#[derive(Debug, Clone, PartialEq)]
pub struct LayerStyle {
    pub label: String,
    pub color: Rgba,
    pub marker: MarkerShape,
    pub marker_size: f32,
    /// Filled markers, or outline-only ("open") markers.
    pub filled: bool,
    /// Stroke width for lines, step histograms, and open markers.
    pub line_width: f32,
    pub hist_style: HistStyle,
}

impl LayerStyle {
    /// Defaults for the i-th layer: palette color, filled circles, bars.
    pub fn new(label: impl Into<String>, i: usize) -> Self {
        Self {
            label: label.into(),
            color: Rgba::layer_color(i),
            marker: MarkerShape::Circle,
            marker_size: 3.0,
            filled: true,
            line_width: 1.5,
            hist_style: HistStyle::Bars,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerShape {
    Circle,
    Square,
    Diamond,
    Cross,
}

impl MarkerShape {
    pub const ALL: [MarkerShape; 4] = [
        MarkerShape::Circle,
        MarkerShape::Square,
        MarkerShape::Diamond,
        MarkerShape::Cross,
    ];
}

impl std::fmt::Display for MarkerShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarkerShape::Circle => write!(f, "● circle"),
            MarkerShape::Square => write!(f, "■ square"),
            MarkerShape::Diamond => write!(f, "◆ diamond"),
            MarkerShape::Cross => write!(f, "✚ cross"),
        }
    }
}

/// How a histogram layer is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistStyle {
    /// Filled bars with an outline.
    Bars,
    /// Outline only ("staircase"), the style used to overlay comparisons.
    Steps,
}

impl HistStyle {
    pub const ALL: [HistStyle; 2] = [HistStyle::Bars, HistStyle::Steps];
}

impl std::fmt::Display for HistStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HistStyle::Bars => write!(f, "bars"),
            HistStyle::Steps => write!(f, "steps"),
        }
    }
}

/// One axis' user-editable state.
#[derive(Debug, Clone, PartialEq)]
pub struct Axis {
    pub label: String,
    /// `None` = auto limits derived from data (padded); `Some` = user-fixed.
    pub limits: Option<(f64, f64)>,
    /// Rough number of ticks to aim for.
    pub tick_target: usize,
}

impl Axis {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            limits: None,
            tick_target: 6,
        }
    }
}

/// What kind of plot this spec describes, with its data bindings.
///
/// Column names bind the plot to its source `DataTable`; the GUI re-fetches
/// data when these change, while pure appearance edits only redraw.
#[derive(Debug, Clone, PartialEq)]
pub enum PlotKind {
    Scatter {
        x_column: String,
        y_column: String,
        x_err_column: Option<String>,
        y_err_column: Option<String>,
    },
    Histogram {
        column: String,
        n_bins: usize,
        log_bins: bool,
    },
    /// Points connected in x-sorted order (the intuitive default; TOPCAT
    /// connects in row order, which surprises everyone at least once).
    Line { x_column: String, y_column: String },
    /// All-sky map. Longitude/latitude columns are in degrees
    /// (RA 0–360, Dec −90–90).
    Sky {
        lon_column: String,
        lat_column: String,
        projection: crate::projection::SkyProjection,
    },
    /// 3D point cloud; camera state lives with the view, not the spec.
    Scatter3D {
        x_column: String,
        y_column: String,
        z_column: String,
    },
    /// Points on (or in) the celestial sphere: lon/lat in degrees, plus an
    /// optional radial distance column. Without distance, points sit on the
    /// unit sphere (TOPCAT's sphere view); with it, at their 3D positions.
    Sphere {
        lon_column: String,
        lat_column: String,
        dist_column: Option<String>,
    },
}

impl PlotKind {
    /// Whether this kind draws cartesian axes (frame, ticks, x/y labels).
    /// Sky and 3D plots manage their own annotation instead.
    pub fn has_axes(&self) -> bool {
        matches!(
            self,
            PlotKind::Scatter { .. } | PlotKind::Histogram { .. } | PlotKind::Line { .. }
        )
    }
}

/// Everything about one plot: kind + data bindings + appearance.
#[derive(Debug, Clone, PartialEq)]
pub struct PlotSpec {
    pub kind: PlotKind,
    pub title: String,
    pub x_axis: Axis,
    pub y_axis: Axis,
    /// One entry per data layer, in draw order. Empty means a single
    /// unlabeled layer using `marker_color`.
    pub layers: Vec<LayerStyle>,
    pub marker_shape: MarkerShape,
    pub marker_size: f32,
    pub marker_color: Rgba,
    /// Stroke width for line plots.
    pub line_width: f32,
    /// Draw markers on top of the line in line plots.
    pub line_markers: bool,
    pub show_grid: bool,
    pub title_size: f32,
    pub label_size: f32,
    pub tick_label_size: f32,
}

impl PlotSpec {
    /// Paper-ready defaults: axis labels seeded from column names so the
    /// figure is publication-plausible with zero configuration.
    pub fn scatter(
        x_column: impl Into<String>,
        y_column: impl Into<String>,
        x_err_column: Option<String>,
        y_err_column: Option<String>,
    ) -> Self {
        let x_column = x_column.into();
        let y_column = y_column.into();
        Self {
            title: String::new(),
            x_axis: Axis::new(x_column.clone()),
            y_axis: Axis::new(y_column.clone()),
            kind: PlotKind::Scatter {
                x_column,
                y_column,
                x_err_column,
                y_err_column,
            },
            ..Self::base()
        }
    }

    pub fn histogram(column: impl Into<String>, n_bins: usize, log_bins: bool) -> Self {
        let column = column.into();
        Self {
            title: String::new(),
            x_axis: Axis::new(column.clone()),
            y_axis: Axis::new("N"),
            kind: PlotKind::Histogram {
                column,
                n_bins,
                log_bins,
            },
            ..Self::base()
        }
    }

    /// Line plot connecting x-sorted points.
    pub fn line(x_column: impl Into<String>, y_column: impl Into<String>) -> Self {
        let x_column = x_column.into();
        let y_column = y_column.into();
        Self {
            title: String::new(),
            x_axis: Axis::new(x_column.clone()),
            y_axis: Axis::new(y_column.clone()),
            kind: PlotKind::Line { x_column, y_column },
            ..Self::base()
        }
    }

    /// All-sky map of lon/lat (degree) columns.
    pub fn sky(
        lon_column: impl Into<String>,
        lat_column: impl Into<String>,
        projection: crate::projection::SkyProjection,
    ) -> Self {
        let lon_column = lon_column.into();
        let lat_column = lat_column.into();
        Self {
            title: String::new(),
            x_axis: Axis::new(lon_column.clone()),
            y_axis: Axis::new(lat_column.clone()),
            kind: PlotKind::Sky {
                lon_column,
                lat_column,
                projection,
            },
            marker_size: 2.0,
            ..Self::base()
        }
    }

    /// Celestial-sphere plot: lon/lat (degrees), optional radial distance.
    pub fn sphere(
        lon_column: impl Into<String>,
        lat_column: impl Into<String>,
        dist_column: Option<String>,
    ) -> Self {
        let lon_column = lon_column.into();
        let lat_column = lat_column.into();
        Self {
            title: String::new(),
            x_axis: Axis::new(lon_column.clone()),
            y_axis: Axis::new(lat_column.clone()),
            kind: PlotKind::Sphere {
                lon_column,
                lat_column,
                dist_column,
            },
            ..Self::base()
        }
    }

    /// 3D point cloud of three columns.
    pub fn scatter3d(
        x_column: impl Into<String>,
        y_column: impl Into<String>,
        z_column: impl Into<String>,
    ) -> Self {
        let x_column = x_column.into();
        let y_column = y_column.into();
        let z_column = z_column.into();
        Self {
            title: String::new(),
            x_axis: Axis::new(x_column.clone()),
            y_axis: Axis::new(y_column.clone()),
            kind: PlotKind::Scatter3D {
                x_column,
                y_column,
                z_column,
            },
            marker_size: 2.5,
            ..Self::base()
        }
    }

    fn base() -> Self {
        Self {
            kind: PlotKind::Histogram {
                column: String::new(),
                n_bins: 0,
                log_bins: false,
            },
            title: String::new(),
            x_axis: Axis::new(""),
            y_axis: Axis::new(""),
            layers: Vec::new(),
            marker_shape: MarkerShape::Circle,
            marker_size: 3.0,
            marker_color: Rgba::STEEL,
            line_width: 1.5,
            line_markers: false,
            show_grid: false,
            title_size: 18.0,
            label_size: 15.0,
            tick_label_size: 12.0,
        }
    }
}
