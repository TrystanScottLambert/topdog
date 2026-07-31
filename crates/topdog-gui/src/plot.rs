//! Plot pane: canvas renderer for `PlotSpec` + fetched data layers, the
//! click-to-edit overlay state, and the 3D camera interaction.
//!
//! A pane holds N data layers drawn in order, styled by `spec.layers`
//! (palette colors, editable legend labels). Rendering reads only
//! (`PlotSpec`, layers, camera, canvas size); the same `PlotLayout` geometry
//! drives drawing and click hit-testing, which is what makes every drawn
//! label — title, axis labels, legend entries — clickable (CLAUDE.md §5).

use iced::mouse;
use iced::widget::canvas::{self, Canvas, Event, Path, Stroke, Text};
use iced::{Color, Element, Length, Point, Rectangle, Size};
use topdog_core::{ColumnsData, Histogram, ScatterData};
use topdog_plot::{
    scale::{self, LinearScale},
    MarkerShape, PlotKind, PlotLayout, PlotSpec, PointF, RectF, Rgba, SkyProjection, TextElement,
};

use crate::Message;

/// Fetched, materialized data behind one plot layer.
#[derive(Debug, Clone)]
pub enum PlotData {
    Scatter(ScatterData),
    Histogram(Histogram),
    /// Pairs already sorted by x.
    Line(ScatterData),
    /// x = longitude (deg), y = latitude (deg).
    Sky(ScatterData),
    /// columns[0..3] = x, y, z.
    Xyz(ColumnsData),
}

/// View state of the 3D camera. Deliberately not part of `PlotSpec`:
/// orientation is interaction state, not figure appearance.
#[derive(Debug, Clone, Copy)]
pub struct Camera3D {
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
    pub pan: (f32, f32),
}

impl Default for Camera3D {
    fn default() -> Self {
        Self {
            yaw: -0.6,
            pitch: 0.4,
            zoom: 1.0,
            pan: (0.0, 0.0),
        }
    }
}

/// An in-progress label edit: which element, where it is on the canvas, and
/// the current text_input contents.
#[derive(Debug, Clone)]
pub struct EditState {
    pub element: TextElement,
    pub rect: RectF,
    pub value: String,
}

/// Messages produced by the plot pane's canvas.
#[derive(Debug, Clone)]
pub enum PlotMessage {
    LabelClicked { element: TextElement, rect: RectF },
    EditChanged(String),
    EditSubmitted,
    Rotate { dyaw: f32, dpitch: f32 },
    PanBy { dx: f32, dy: f32 },
    ZoomBy(f32),
}

pub struct PlotPane {
    pub spec: PlotSpec,
    /// Data layers, parallel to `spec.layers`.
    pub layers: Vec<PlotData>,
    pub cache: canvas::Cache,
    pub editing: Option<EditState>,
    pub camera: Camera3D,
}

impl PlotPane {
    pub fn new(spec: PlotSpec, layers: Vec<PlotData>) -> Self {
        Self {
            spec,
            layers,
            cache: canvas::Cache::default(),
            editing: None,
            camera: Camera3D::default(),
        }
    }

    pub fn add_layer(&mut self, label: String, data: PlotData) {
        let color = Rgba::layer_color(self.spec.layers.len());
        self.spec
            .layers
            .push(topdog_plot::LayerStyle { label, color });
        self.layers.push(data);
        self.cache.clear();
    }

    pub fn remove_layer(&mut self, i: usize) {
        if i < self.layers.len() {
            self.layers.remove(i);
            self.spec.layers.remove(i);
            self.cache.clear();
        }
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Current text of an editable element.
    pub fn text_of(&self, element: TextElement) -> &str {
        match element {
            TextElement::Title => &self.spec.title,
            TextElement::XLabel => &self.spec.x_axis.label,
            TextElement::YLabel => &self.spec.y_axis.label,
            TextElement::Legend(i) => self
                .spec
                .layers
                .get(i)
                .map(|l| l.label.as_str())
                .unwrap_or(""),
        }
    }

    /// Write an edited value back into the spec (appearance-only: redraw,
    /// no data re-fetch).
    pub fn commit_edit(&mut self) {
        if let Some(edit) = self.editing.take() {
            let value = edit.value;
            match edit.element {
                TextElement::Title => self.spec.title = value,
                TextElement::XLabel => self.spec.x_axis.label = value,
                TextElement::YLabel => self.spec.y_axis.label = value,
                TextElement::Legend(i) => {
                    if let Some(layer) = self.spec.layers.get_mut(i) {
                        layer.label = value;
                    }
                }
            }
            self.cache.clear();
        }
    }

    pub fn canvas(&self) -> Element<'_, Message> {
        Canvas::new(PlotProgram { pane: self })
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}

struct PlotProgram<'a> {
    pane: &'a PlotPane,
}

/// Per-canvas interaction state: an active drag (button + last position).
#[derive(Debug, Clone, Copy, Default)]
pub struct DragState {
    drag: Option<(mouse::Button, Point)>,
}

impl canvas::Program<Message> for PlotProgram<'_> {
    type State = DragState;

    fn update(
        &self,
        state: &mut Self::State,
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<canvas::Action<Message>> {
        let is_3d = matches!(self.pane.spec.kind, PlotKind::Scatter3D { .. });

        match event {
            Event::Mouse(mouse::Event::ButtonPressed(button)) => {
                let pos = cursor.position_in(bounds)?;
                if *button == mouse::Button::Left {
                    let layout = PlotLayout::compute(&self.pane.spec, bounds.width, bounds.height);
                    if let Some(element) = layout.hit_test(PointF { x: pos.x, y: pos.y }) {
                        return Some(canvas::Action::publish(Message::Plot(
                            PlotMessage::LabelClicked {
                                element,
                                rect: layout.rect_of(element),
                            },
                        )));
                    }
                }
                if is_3d && matches!(button, mouse::Button::Left | mouse::Button::Right) {
                    state.drag = Some((*button, pos));
                    return Some(canvas::Action::capture());
                }
                None
            }
            Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                let (button, last) = state.drag?;
                let pos = cursor.position_in(bounds)?;
                let (dx, dy) = (pos.x - last.x, pos.y - last.y);
                state.drag = Some((button, pos));
                let message = match button {
                    mouse::Button::Left => PlotMessage::Rotate {
                        dyaw: dx * 0.01,
                        dpitch: dy * 0.01,
                    },
                    _ => PlotMessage::PanBy { dx, dy },
                };
                Some(canvas::Action::publish(Message::Plot(message)).and_capture())
            }
            Event::Mouse(mouse::Event::ButtonReleased(_)) => {
                state.drag = None;
                None
            }
            Event::Mouse(mouse::Event::WheelScrolled { delta }) if is_3d => {
                cursor.position_in(bounds)?;
                let amount = match delta {
                    mouse::ScrollDelta::Lines { y, .. } => *y,
                    mouse::ScrollDelta::Pixels { y, .. } => *y / 40.0,
                };
                Some(
                    canvas::Action::publish(Message::Plot(PlotMessage::ZoomBy(amount)))
                        .and_capture(),
                )
            }
            _ => None,
        }
    }

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &iced::Renderer,
        _theme: &iced::Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let geometry = self.pane.cache.draw(renderer, bounds.size(), |frame| {
            draw_plot(
                frame,
                &self.pane.spec,
                &self.pane.layers,
                &self.pane.camera,
                self.pane.editing.as_ref().map(|e| e.element),
            );
        });
        vec![geometry]
    }

    fn mouse_interaction(
        &self,
        state: &Self::State,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if state.drag.is_some() {
            return mouse::Interaction::Grabbing;
        }
        if let Some(pos) = cursor.position_in(bounds) {
            let layout = PlotLayout::compute(&self.pane.spec, bounds.width, bounds.height);
            if layout.hit_test(PointF { x: pos.x, y: pos.y }).is_some() {
                return mouse::Interaction::Pointer;
            }
            if matches!(self.pane.spec.kind, PlotKind::Scatter3D { .. }) {
                return mouse::Interaction::Grab;
            }
        }
        mouse::Interaction::default()
    }
}

fn color(c: Rgba) -> Color {
    Color::from_rgba(c.r, c.g, c.b, c.a)
}

/// Color for the i-th layer (spec style, falling back to marker_color).
fn layer_color(spec: &PlotSpec, i: usize) -> Color {
    spec.layers
        .get(i)
        .map(|l| color(l.color))
        .unwrap_or_else(|| color(spec.marker_color))
}

const FRAME_COLOR: Color = Color::BLACK;
const GRID_COLOR: Color = Color::from_rgba(0.0, 0.0, 0.0, 0.12);
const GRATICULE_COLOR: Color = Color::from_rgba(0.0, 0.0, 0.0, 0.18);
const ANNOTATION_COLOR: Color = Color::from_rgba(0.0, 0.0, 0.0, 0.45);
const TICK_LEN: f32 = 5.0;

/// Render the whole figure. `editing` suppresses that element's text so the
/// floating text_input overlay visually replaces it.
fn draw_plot(
    frame: &mut canvas::Frame,
    spec: &PlotSpec,
    layers: &[PlotData],
    camera: &Camera3D,
    editing: Option<TextElement>,
) {
    let width = frame.width();
    let height = frame.height();
    let layout = PlotLayout::compute(spec, width, height);

    // Figures are white regardless of app theme: they should look like the
    // exported (paper) version at all times.
    frame.fill_rectangle(Point::ORIGIN, Size::new(width, height), Color::WHITE);

    if spec.kind.has_axes() {
        let pa = layout.plot_area;
        let (x_domain, y_domain) = domains(spec, layers);
        let x_scale = LinearScale::new(x_domain, (pa.x, pa.x + pa.width));
        let y_scale = LinearScale::new(y_domain, (pa.y + pa.height, pa.y));

        draw_axes(frame, spec, &layout, &x_scale, &y_scale);

        for (i, data) in layers.iter().enumerate() {
            let lc = layer_color(spec, i);
            match data {
                PlotData::Scatter(points) => {
                    draw_scatter(frame, spec, points, lc, &x_scale, &y_scale)
                }
                PlotData::Histogram(hist) => draw_histogram(frame, hist, lc, &x_scale, &y_scale),
                PlotData::Line(points) => draw_line(frame, spec, points, lc, &x_scale, &y_scale),
                _ => {}
            }
        }
        downsample_note(frame, spec, layers, layout.plot_area);
    } else {
        match &spec.kind {
            PlotKind::Sky { projection, .. } => draw_sky(frame, spec, layers, *projection, &layout),
            PlotKind::Scatter3D { .. } => draw_3d(frame, spec, layers, camera, &layout),
            _ => {}
        }
    }

    draw_legend(frame, spec, &layout, editing);
    draw_labels(frame, spec, &layout, editing);
}

fn scatter_like(data: &PlotData) -> Option<&ScatterData> {
    match data {
        PlotData::Scatter(p) | PlotData::Line(p) | PlotData::Sky(p) => Some(p),
        _ => None,
    }
}

/// Axis domains: user-set limits win; otherwise the union of all layers.
fn domains(spec: &PlotSpec, layers: &[PlotData]) -> ((f64, f64), (f64, f64)) {
    let mut x = (f64::INFINITY, f64::NEG_INFINITY);
    let mut y = (f64::INFINITY, f64::NEG_INFINITY);
    let fold = |r: &mut (f64, f64), lo: f64, hi: f64| {
        r.0 = r.0.min(lo);
        r.1 = r.1.max(hi);
    };
    for data in layers {
        match data {
            PlotData::Histogram(hist) => {
                if let (Some(&lo), Some(&hi)) = (hist.edges.first(), hist.edges.last()) {
                    fold(&mut x, lo, hi);
                }
                let peak = hist.counts.iter().max().copied().unwrap_or(0) as f64;
                fold(&mut y, 0.0, (peak * 1.08).max(1.0));
            }
            other => {
                if let Some(points) = scatter_like(other) {
                    let (lo, hi) = min_max(&points.x);
                    let (plo, phi) = scale::padded_limits(lo, hi);
                    fold(&mut x, plo, phi);
                    let (lo, hi) = min_max(&points.y);
                    let (plo, phi) = scale::padded_limits(lo, hi);
                    fold(&mut y, plo, phi);
                }
            }
        }
    }
    if !x.0.is_finite() || !x.1.is_finite() {
        x = (0.0, 1.0);
    }
    if !y.0.is_finite() || !y.1.is_finite() {
        y = (0.0, 1.0);
    }
    (
        spec.x_axis.limits.unwrap_or(x),
        spec.y_axis.limits.unwrap_or(y),
    )
}

fn min_max(values: &[f64]) -> (f64, f64) {
    values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        })
}

fn draw_axes(
    frame: &mut canvas::Frame,
    spec: &PlotSpec,
    layout: &PlotLayout,
    x_scale: &LinearScale,
    y_scale: &LinearScale,
) {
    let pa = layout.plot_area;
    let thin = Stroke::default().with_width(1.0).with_color(FRAME_COLOR);
    let grid = Stroke::default().with_width(1.0).with_color(GRID_COLOR);

    let x_ticks = scale::nice_ticks(x_scale.domain.0, x_scale.domain.1, spec.x_axis.tick_target);
    let x_step = scale::tick_step(&x_ticks);
    for &t in &x_ticks {
        let sx = x_scale.to_screen(t);
        if spec.show_grid {
            let path = Path::line(Point::new(sx, pa.y), Point::new(sx, pa.y + pa.height));
            frame.stroke(&path, grid);
        }
        let bottom = pa.y + pa.height;
        let path = Path::line(Point::new(sx, bottom), Point::new(sx, bottom - TICK_LEN));
        frame.stroke(&path, thin);
        frame.fill_text(Text {
            content: scale::format_tick(t, x_step),
            position: Point::new(sx, bottom + 4.0),
            color: FRAME_COLOR,
            size: spec.tick_label_size.into(),
            align_x: iced::widget::text::Alignment::Center,
            align_y: iced::alignment::Vertical::Top,
            ..Text::default()
        });
    }

    let y_ticks = scale::nice_ticks(y_scale.domain.0, y_scale.domain.1, spec.y_axis.tick_target);
    let y_step = scale::tick_step(&y_ticks);
    for &t in &y_ticks {
        let sy = y_scale.to_screen(t);
        if spec.show_grid {
            let path = Path::line(Point::new(pa.x, sy), Point::new(pa.x + pa.width, sy));
            frame.stroke(&path, grid);
        }
        let path = Path::line(Point::new(pa.x, sy), Point::new(pa.x + TICK_LEN, sy));
        frame.stroke(&path, thin);
        frame.fill_text(Text {
            content: scale::format_tick(t, y_step),
            position: Point::new(pa.x - 6.0, sy),
            color: FRAME_COLOR,
            size: spec.tick_label_size.into(),
            align_x: iced::widget::text::Alignment::Right,
            align_y: iced::alignment::Vertical::Center,
            ..Text::default()
        });
    }

    // Frame last so it sits on top of gridlines.
    frame.stroke_rectangle(Point::new(pa.x, pa.y), Size::new(pa.width, pa.height), thin);
}

fn draw_scatter(
    frame: &mut canvas::Frame,
    spec: &PlotSpec,
    points: &ScatterData,
    mc: Color,
    x_scale: &LinearScale,
    y_scale: &LinearScale,
) {
    let err_stroke = Stroke::default().with_width(1.0).with_color(mc);
    let (x0, x1) = x_scale.domain;
    let (y0, y1) = y_scale.domain;
    let r = spec.marker_size;

    for i in 0..points.x.len() {
        let (dx, dy) = (points.x[i], points.y[i]);
        if dx < x0 || dx > x1 || dy < y0 || dy > y1 {
            continue; // outside current limits
        }
        let p = Point::new(x_scale.to_screen(dx), y_scale.to_screen(dy));

        if let Some(errs) = &points.x_err {
            let e = errs[i];
            let a = Point::new(x_scale.to_screen(dx - e), p.y);
            let b = Point::new(x_scale.to_screen(dx + e), p.y);
            frame.stroke(&Path::line(a, b), err_stroke);
            let cap = r.max(2.0);
            frame.stroke(
                &Path::line(Point::new(a.x, a.y - cap), Point::new(a.x, a.y + cap)),
                err_stroke,
            );
            frame.stroke(
                &Path::line(Point::new(b.x, b.y - cap), Point::new(b.x, b.y + cap)),
                err_stroke,
            );
        }
        if let Some(errs) = &points.y_err {
            let e = errs[i];
            let a = Point::new(p.x, y_scale.to_screen(dy - e));
            let b = Point::new(p.x, y_scale.to_screen(dy + e));
            frame.stroke(&Path::line(a, b), err_stroke);
            let cap = r.max(2.0);
            frame.stroke(
                &Path::line(Point::new(a.x - cap, a.y), Point::new(a.x + cap, a.y)),
                err_stroke,
            );
            frame.stroke(
                &Path::line(Point::new(b.x - cap, b.y), Point::new(b.x + cap, b.y)),
                err_stroke,
            );
        }

        draw_marker(frame, p, r, spec.marker_shape, mc);
    }
}

fn draw_line(
    frame: &mut canvas::Frame,
    spec: &PlotSpec,
    points: &ScatterData,
    mc: Color,
    x_scale: &LinearScale,
    y_scale: &LinearScale,
) {
    let stroke = Stroke::default().with_width(spec.line_width).with_color(mc);

    // One polyline through all points in (pre-sorted) x order.
    let mut first = true;
    let path = Path::new(|b| {
        for i in 0..points.x.len() {
            let p = Point::new(
                x_scale.to_screen(points.x[i]),
                y_scale.to_screen(points.y[i]),
            );
            if first {
                b.move_to(p);
                first = false;
            } else {
                b.line_to(p);
            }
        }
    });
    frame.stroke(&path, stroke);

    if spec.line_markers {
        for i in 0..points.x.len() {
            let p = Point::new(
                x_scale.to_screen(points.x[i]),
                y_scale.to_screen(points.y[i]),
            );
            draw_marker(frame, p, spec.marker_size, spec.marker_shape, mc);
        }
    }
}

fn draw_sky(
    frame: &mut canvas::Frame,
    spec: &PlotSpec,
    layers: &[PlotData],
    projection: SkyProjection,
    layout: &PlotLayout,
) {
    let pa = layout.plot_area;
    let (xm, ym) = projection.extent();
    let s = ((pa.width as f64 / (2.0 * xm)).min(pa.height as f64 / (2.0 * ym)) * 0.97) as f32;
    let cx = pa.x + pa.width / 2.0;
    let cy = pa.y + pa.height / 2.0;
    let deg = std::f64::consts::PI / 180.0;

    // Sky convention: longitude (RA) increases to the LEFT, map centered on
    // lon = 180°.
    let to_screen =
        |px: f64, py: f64| -> Point { Point::new(cx - s * px as f32, cy - s * py as f32) };
    let polyline = |frame: &mut canvas::Frame, pts: &[(f64, f64)], stroke: Stroke<'_>| {
        let path = Path::new(|b| {
            for (i, &(px, py)) in pts.iter().enumerate() {
                let p = to_screen(px, py);
                if i == 0 {
                    b.move_to(p);
                } else {
                    b.line_to(p);
                }
            }
        });
        frame.stroke(&path, stroke);
    };

    let grat = Stroke::default()
        .with_width(1.0)
        .with_color(GRATICULE_COLOR);
    for line in projection.graticule(30.0, 30.0) {
        polyline(frame, &line, grat);
    }
    let edge = Stroke::default().with_width(1.2).with_color(FRAME_COLOR);
    polyline(frame, &projection.boundary(60), edge);

    // Coordinate annotations: RA along the equator, Dec down the center.
    for lon in [0.0f64, 60.0, 120.0, 180.0, 240.0, 300.0] {
        let (px, py) = projection.project((lon - 180.0) * deg, 0.0);
        frame.fill_text(Text {
            content: format!("{lon:.0}°"),
            position: to_screen(px, py),
            color: ANNOTATION_COLOR,
            size: (spec.tick_label_size - 1.0).max(8.0).into(),
            align_x: iced::widget::text::Alignment::Center,
            align_y: iced::alignment::Vertical::Bottom,
            ..Text::default()
        });
    }
    for lat in [-60.0f64, -30.0, 30.0, 60.0] {
        let (px, py) = projection.project(0.0, lat * deg);
        frame.fill_text(Text {
            content: format!("{lat:+.0}°"),
            position: to_screen(px, py),
            color: ANNOTATION_COLOR,
            size: (spec.tick_label_size - 1.0).max(8.0).into(),
            align_x: iced::widget::text::Alignment::Center,
            align_y: iced::alignment::Vertical::Center,
            ..Text::default()
        });
    }

    for (i, data) in layers.iter().enumerate() {
        let Some(points) = scatter_like(data) else {
            continue;
        };
        let mc = layer_color(spec, i);
        for j in 0..points.x.len() {
            let lon = points.x[j];
            let lat = points.y[j];
            if !(-90.0..=90.0).contains(&lat) {
                continue;
            }
            // Wrap RA into [0, 360), then center.
            let lon = lon.rem_euclid(360.0);
            let (px, py) = projection.project((lon - 180.0) * deg, lat * deg);
            draw_marker(
                frame,
                to_screen(px, py),
                spec.marker_size,
                spec.marker_shape,
                mc,
            );
        }
    }

    downsample_note(frame, spec, layers, pa);
}

fn draw_3d(
    frame: &mut canvas::Frame,
    spec: &PlotSpec,
    layers: &[PlotData],
    camera: &Camera3D,
    layout: &PlotLayout,
) {
    let pa = layout.plot_area;
    let scale_px = camera.zoom * pa.width.min(pa.height) * 0.55;
    let cx = pa.x + pa.width / 2.0 + camera.pan.0;
    let cy = pa.y + pa.height / 2.0 + camera.pan.1;
    let (sy, cyaw) = camera.yaw.sin_cos();
    let (sp, cp) = camera.pitch.sin_cos();

    // Orthographic camera: yaw about the vertical axis, then pitch; screen
    // x is rotated x, screen y is (negated) rotated z. Depth is unused.
    let project = |x: f32, y: f32, z: f32| -> Point {
        let x1 = x * cyaw - y * sy;
        let y1 = x * sy + y * cyaw;
        let z2 = y1 * sp + z * cp;
        Point::new(cx + x1 * scale_px, cy - z2 * scale_px)
    };

    // Normalize into the unit cube using the union range across all layers,
    // so layers stay mutually comparable.
    let mut ranges = [(f64::INFINITY, f64::NEG_INFINITY); 3];
    for data in layers {
        if let PlotData::Xyz(points) = data {
            for (axis, range) in ranges.iter_mut().enumerate() {
                let (lo, hi) = min_max(&points.columns[axis]);
                range.0 = range.0.min(lo);
                range.1 = range.1.max(hi);
            }
        }
    }
    let norm = |v: f64, (lo, hi): (f64, f64)| -> f32 {
        if hi > lo {
            (((v - lo) / (hi - lo)) - 0.5) as f32
        } else {
            0.0
        }
    };

    // Cube wireframe.
    let edge_stroke = Stroke::default()
        .with_width(1.0)
        .with_color(GRATICULE_COLOR);
    let corners: Vec<Point> = (0..8)
        .map(|i| {
            let x = if i & 1 == 0 { -0.5 } else { 0.5 };
            let y = if i & 2 == 0 { -0.5 } else { 0.5 };
            let z = if i & 4 == 0 { -0.5 } else { 0.5 };
            project(x, y, z)
        })
        .collect();
    for (a, b) in [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ] {
        frame.stroke(&Path::line(corners[a], corners[b]), edge_stroke);
    }

    // Axis labels at the ends of the three edges leaving corner 0.
    if let PlotKind::Scatter3D {
        x_column,
        y_column,
        z_column,
    } = &spec.kind
    {
        for (label, end) in [
            (x_column, project(0.62, -0.5, -0.5)),
            (y_column, project(-0.5, 0.62, -0.5)),
            (z_column, project(-0.5, -0.5, 0.62)),
        ] {
            frame.fill_text(Text {
                content: label.clone(),
                position: end,
                color: FRAME_COLOR,
                size: spec.tick_label_size.into(),
                align_x: iced::widget::text::Alignment::Center,
                align_y: iced::alignment::Vertical::Center,
                ..Text::default()
            });
        }
    }

    let r = spec.marker_size * 0.8;
    for (i, data) in layers.iter().enumerate() {
        let PlotData::Xyz(points) = data else {
            continue;
        };
        let mc = layer_color(spec, i);
        let (xs, ys, zs) = (&points.columns[0], &points.columns[1], &points.columns[2]);
        for j in 0..xs.len() {
            let p = project(
                norm(xs[j], ranges[0]),
                norm(ys[j], ranges[1]),
                norm(zs[j], ranges[2]),
            );
            frame.fill(&Path::circle(p, r), mc);
        }
    }

    downsample_note(frame, spec, layers, pa);
    annotation(
        frame,
        spec,
        "drag rotate · right-drag pan · scroll zoom".to_string(),
        Point::new(pa.x + pa.width - 4.0, pa.y + pa.height - 14.0),
    );
}

fn annotation(frame: &mut canvas::Frame, spec: &PlotSpec, content: String, position: Point) {
    frame.fill_text(Text {
        content,
        position,
        color: ANNOTATION_COLOR,
        size: (spec.tick_label_size - 1.0).max(8.0).into(),
        align_x: iced::widget::text::Alignment::Right,
        align_y: iced::alignment::Vertical::Top,
        ..Text::default()
    });
}

/// One shared honesty note when any layer was stride-downsampled.
fn downsample_note(frame: &mut canvas::Frame, spec: &PlotSpec, layers: &[PlotData], pa: RectF) {
    let max_stride = layers
        .iter()
        .filter_map(|d| match d {
            PlotData::Scatter(p) | PlotData::Line(p) | PlotData::Sky(p) => Some(p.stride),
            PlotData::Xyz(p) => Some(p.stride),
            PlotData::Histogram(_) => None,
        })
        .max()
        .unwrap_or(1);
    if max_stride > 1 {
        // Legend occupies the top-right; place the note bottom-left inside
        // the plot area to avoid collisions.
        frame.fill_text(Text {
            content: format!("downsampled (up to every {max_stride}th point)"),
            position: Point::new(pa.x + 4.0, pa.y + pa.height - 4.0),
            color: ANNOTATION_COLOR,
            size: (spec.tick_label_size - 1.0).max(8.0).into(),
            align_x: iced::widget::text::Alignment::Left,
            align_y: iced::alignment::Vertical::Bottom,
            ..Text::default()
        });
    }
}

fn draw_marker(frame: &mut canvas::Frame, p: Point, r: f32, shape: MarkerShape, color: Color) {
    match shape {
        MarkerShape::Circle => frame.fill(&Path::circle(p, r), color),
        MarkerShape::Square => frame.fill_rectangle(
            Point::new(p.x - r, p.y - r),
            Size::new(2.0 * r, 2.0 * r),
            color,
        ),
        MarkerShape::Diamond => {
            let path = Path::new(|b| {
                b.move_to(Point::new(p.x, p.y - r * 1.3));
                b.line_to(Point::new(p.x + r * 1.3, p.y));
                b.line_to(Point::new(p.x, p.y + r * 1.3));
                b.line_to(Point::new(p.x - r * 1.3, p.y));
                b.close();
            });
            frame.fill(&path, color);
        }
        MarkerShape::Cross => {
            let stroke = Stroke::default().with_width(1.5).with_color(color);
            frame.stroke(
                &Path::line(Point::new(p.x - r, p.y), Point::new(p.x + r, p.y)),
                stroke,
            );
            frame.stroke(
                &Path::line(Point::new(p.x, p.y - r), Point::new(p.x, p.y + r)),
                stroke,
            );
        }
    }
}

fn draw_histogram(
    frame: &mut canvas::Frame,
    hist: &Histogram,
    mc: Color,
    x_scale: &LinearScale,
    y_scale: &LinearScale,
) {
    // Translucent so overlaid layer histograms stay readable.
    let fill = Color { a: 0.55, ..mc };
    let outline = Stroke::default().with_width(1.0).with_color(mc);
    let base = y_scale.to_screen(0.0);

    for (i, &count) in hist.counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let left = x_scale.to_screen(hist.edges[i]);
        let right = x_scale.to_screen(hist.edges[i + 1]);
        let top = y_scale.to_screen(count as f64);
        frame.fill_rectangle(
            Point::new(left, top),
            Size::new(right - left, base - top),
            fill,
        );
        frame.stroke_rectangle(
            Point::new(left, top),
            Size::new(right - left, base - top),
            outline,
        );
    }
}

/// Legend box with color swatches; entry labels are click-to-edit.
fn draw_legend(
    frame: &mut canvas::Frame,
    spec: &PlotSpec,
    layout: &PlotLayout,
    editing: Option<TextElement>,
) {
    if layout.legend_labels.is_empty() {
        return;
    }
    let b = layout.legend_box;
    frame.fill_rectangle(
        Point::new(b.x, b.y),
        Size::new(b.width, b.height),
        Color::from_rgba(1.0, 1.0, 1.0, 0.85),
    );
    frame.stroke_rectangle(
        Point::new(b.x, b.y),
        Size::new(b.width, b.height),
        Stroke::default().with_width(1.0).with_color(FRAME_COLOR),
    );

    for (i, layer) in spec.layers.iter().enumerate() {
        let sw = layout.legend_swatches[i];
        frame.fill_rectangle(
            Point::new(sw.x, sw.y),
            Size::new(sw.width, sw.height),
            color(layer.color),
        );
        if editing == Some(TextElement::Legend(i)) {
            continue;
        }
        let lr = layout.legend_labels[i];
        frame.fill_text(Text {
            content: layer.label.clone(),
            position: Point::new(lr.x, lr.y + lr.height / 2.0),
            color: FRAME_COLOR,
            size: spec.tick_label_size.into(),
            align_x: iced::widget::text::Alignment::Left,
            align_y: iced::alignment::Vertical::Center,
            ..Text::default()
        });
    }
}

fn draw_labels(
    frame: &mut canvas::Frame,
    spec: &PlotSpec,
    layout: &PlotLayout,
    editing: Option<TextElement>,
) {
    let center = |r: RectF| Point::new(r.x + r.width / 2.0, r.y + r.height / 2.0);

    if editing != Some(TextElement::Title) && !spec.title.is_empty() {
        frame.fill_text(Text {
            content: spec.title.clone(),
            position: center(layout.title_rect),
            color: FRAME_COLOR,
            size: spec.title_size.into(),
            align_x: iced::widget::text::Alignment::Center,
            align_y: iced::alignment::Vertical::Center,
            ..Text::default()
        });
    }

    if !spec.kind.has_axes() {
        return; // sky/3D annotate themselves
    }

    if editing != Some(TextElement::XLabel) {
        frame.fill_text(Text {
            content: spec.x_axis.label.clone(),
            position: center(layout.x_label_rect),
            color: FRAME_COLOR,
            size: spec.label_size.into(),
            align_x: iced::widget::text::Alignment::Center,
            align_y: iced::alignment::Vertical::Center,
            ..Text::default()
        });
    }

    if editing != Some(TextElement::YLabel) {
        // Rotated 90° CCW around the label rect's center.
        let c = center(layout.y_label_rect);
        frame.with_save(|frame| {
            frame.translate(iced::Vector::new(c.x, c.y));
            frame.rotate(-std::f32::consts::FRAC_PI_2);
            frame.fill_text(Text {
                content: spec.y_axis.label.clone(),
                position: Point::ORIGIN,
                color: FRAME_COLOR,
                size: spec.label_size.into(),
                align_x: iced::widget::text::Alignment::Center,
                align_y: iced::alignment::Vertical::Center,
                ..Text::default()
            });
        });
    }
}
