//! Plot pane: renderer for `PlotSpec` + fetched data layers, click-to-edit
//! overlay state, 3D camera interaction, region selection, and export.
//!
//! All drawing goes through the [`Surface`] trait with two backends: the
//! iced canvas frame (screen) and an SVG writer (export). One code path
//! renders both, so saved figures match the screen exactly (CLAUDE.md §5).

use iced::mouse;
use iced::widget::canvas::{self, Canvas, Event, Path, Stroke, Text};
use iced::{Color, Element, Length, Point, Rectangle, Size};
use topdog_core::{ColumnsData, Histogram, ScatterData};
use topdog_plot::{
    projection::{sky_to_xyz, sphere_wireframe},
    scale::{self, LinearScale},
    HistStyle, LayerStyle, MarkerShape, PlotKind, PlotLayout, PlotSpec, PointF, RectF, Rgba,
    SkyProjection, TextElement,
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
    /// columns = [lon (deg), lat (deg)] or [lon, lat, distance].
    Sphere(ColumnsData),
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
    LabelClicked {
        element: TextElement,
        rect: RectF,
    },
    EditChanged(String),
    EditSubmitted,
    Rotate {
        dyaw: f32,
        dpitch: f32,
    },
    PanBy {
        dx: f32,
        dy: f32,
    },
    ZoomBy(f32),
    /// A rectangle was rubber-band selected, in data coordinates
    /// (x ascending, y ascending).
    RegionSelected {
        x: (f64, f64),
        y: (f64, f64),
    },
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
        let style = LayerStyle::new(label, self.spec.layers.len());
        self.spec.layers.push(style);
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

    /// Render this plot to a standalone SVG document.
    pub fn to_svg(&self, width: f32, height: f32) -> String {
        let mut surface = SvgSurface::new(width, height);
        draw_plot(&mut surface, &self.spec, &self.layers, &self.camera, None);
        surface.finish()
    }

    /// Whether drag-select-a-region makes sense for this plot kind.
    pub fn supports_region_select(&self) -> bool {
        self.spec.kind.has_axes()
    }
}

// ---------------------------------------------------------------------------
// Surface: one drawing vocabulary, two backends (screen + SVG export).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HAlign {
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VAlign {
    Top,
    Center,
    Bottom,
}

/// The drawing operations `draw_plot` needs. Implemented by the iced canvas
/// frame and by [`SvgSurface`]; export therefore cannot drift from the
/// on-screen rendering.
trait Surface {
    fn size(&self) -> (f32, f32);
    fn fill_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color);
    fn stroke_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color, width: f32);
    fn line(&mut self, a: (f32, f32), b: (f32, f32), color: Color, width: f32);
    fn polyline(&mut self, pts: &[(f32, f32)], color: Color, width: f32);
    fn fill_polygon(&mut self, pts: &[(f32, f32)], color: Color);
    fn fill_circle(&mut self, c: (f32, f32), r: f32, color: Color);
    fn stroke_circle(&mut self, c: (f32, f32), r: f32, color: Color, width: f32);
    #[allow(clippy::too_many_arguments)]
    fn text(
        &mut self,
        content: &str,
        x: f32,
        y: f32,
        size: f32,
        color: Color,
        h: HAlign,
        v: VAlign,
        rotated_ccw: bool,
    );
}

impl Surface for canvas::Frame {
    fn size(&self) -> (f32, f32) {
        (self.width(), self.height())
    }

    fn fill_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color) {
        self.fill_rectangle(Point::new(x, y), Size::new(w, h), color);
    }

    fn stroke_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color, width: f32) {
        self.stroke_rectangle(
            Point::new(x, y),
            Size::new(w, h),
            Stroke::default().with_width(width).with_color(color),
        );
    }

    fn line(&mut self, a: (f32, f32), b: (f32, f32), color: Color, width: f32) {
        self.stroke(
            &Path::line(Point::new(a.0, a.1), Point::new(b.0, b.1)),
            Stroke::default().with_width(width).with_color(color),
        );
    }

    fn polyline(&mut self, pts: &[(f32, f32)], color: Color, width: f32) {
        if pts.len() < 2 {
            return;
        }
        let path = Path::new(|b| {
            b.move_to(Point::new(pts[0].0, pts[0].1));
            for &(x, y) in &pts[1..] {
                b.line_to(Point::new(x, y));
            }
        });
        self.stroke(&path, Stroke::default().with_width(width).with_color(color));
    }

    fn fill_polygon(&mut self, pts: &[(f32, f32)], color: Color) {
        if pts.len() < 3 {
            return;
        }
        let path = Path::new(|b| {
            b.move_to(Point::new(pts[0].0, pts[0].1));
            for &(x, y) in &pts[1..] {
                b.line_to(Point::new(x, y));
            }
            b.close();
        });
        self.fill(&path, color);
    }

    fn fill_circle(&mut self, c: (f32, f32), r: f32, color: Color) {
        self.fill(&Path::circle(Point::new(c.0, c.1), r), color);
    }

    fn stroke_circle(&mut self, c: (f32, f32), r: f32, color: Color, width: f32) {
        self.stroke(
            &Path::circle(Point::new(c.0, c.1), r),
            Stroke::default().with_width(width).with_color(color),
        );
    }

    fn text(
        &mut self,
        content: &str,
        x: f32,
        y: f32,
        size: f32,
        color: Color,
        h: HAlign,
        v: VAlign,
        rotated_ccw: bool,
    ) {
        let align_x = match h {
            HAlign::Left => iced::widget::text::Alignment::Left,
            HAlign::Center => iced::widget::text::Alignment::Center,
            HAlign::Right => iced::widget::text::Alignment::Right,
        };
        let align_y = match v {
            VAlign::Top => iced::alignment::Vertical::Top,
            VAlign::Center => iced::alignment::Vertical::Center,
            VAlign::Bottom => iced::alignment::Vertical::Bottom,
        };
        let make = |position: Point| Text {
            content: content.to_string(),
            position,
            color,
            size: size.into(),
            align_x,
            align_y,
            ..Text::default()
        };
        if rotated_ccw {
            self.with_save(|frame| {
                frame.translate(iced::Vector::new(x, y));
                frame.rotate(-std::f32::consts::FRAC_PI_2);
                frame.fill_text(make(Point::ORIGIN));
            });
        } else {
            self.fill_text(make(Point::new(x, y)));
        }
    }
}

/// SVG document builder implementing [`Surface`].
pub struct SvgSurface {
    body: String,
    width: f32,
    height: f32,
}

fn svg_color(c: Color) -> String {
    format!(
        "rgb({},{},{})",
        (c.r * 255.0).round() as u8,
        (c.g * 255.0).round() as u8,
        (c.b * 255.0).round() as u8
    )
}

impl SvgSurface {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            body: String::new(),
            width,
            height,
        }
    }

    pub fn finish(self) -> String {
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" \
             viewBox=\"0 0 {w} {h}\">\n{body}</svg>\n",
            w = self.width,
            h = self.height,
            body = self.body
        )
    }

    fn points_attr(pts: &[(f32, f32)]) -> String {
        pts.iter()
            .map(|(x, y)| format!("{x:.2},{y:.2}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Surface for SvgSurface {
    fn size(&self) -> (f32, f32) {
        (self.width, self.height)
    }

    fn fill_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color) {
        self.body.push_str(&format!(
            "<rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" \
             fill=\"{}\" fill-opacity=\"{:.3}\"/>\n",
            svg_color(color),
            color.a
        ));
    }

    fn stroke_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color, width: f32) {
        self.body.push_str(&format!(
            "<rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" fill=\"none\" \
             stroke=\"{}\" stroke-opacity=\"{:.3}\" stroke-width=\"{width:.2}\"/>\n",
            svg_color(color),
            color.a
        ));
    }

    fn line(&mut self, a: (f32, f32), b: (f32, f32), color: Color, width: f32) {
        self.body.push_str(&format!(
            "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" \
             stroke=\"{}\" stroke-opacity=\"{:.3}\" stroke-width=\"{width:.2}\"/>\n",
            a.0,
            a.1,
            b.0,
            b.1,
            svg_color(color),
            color.a
        ));
    }

    fn polyline(&mut self, pts: &[(f32, f32)], color: Color, width: f32) {
        if pts.len() < 2 {
            return;
        }
        self.body.push_str(&format!(
            "<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-opacity=\"{:.3}\" \
             stroke-width=\"{width:.2}\" stroke-linejoin=\"round\"/>\n",
            Self::points_attr(pts),
            svg_color(color),
            color.a
        ));
    }

    fn fill_polygon(&mut self, pts: &[(f32, f32)], color: Color) {
        if pts.len() < 3 {
            return;
        }
        self.body.push_str(&format!(
            "<polygon points=\"{}\" fill=\"{}\" fill-opacity=\"{:.3}\"/>\n",
            Self::points_attr(pts),
            svg_color(color),
            color.a
        ));
    }

    fn fill_circle(&mut self, c: (f32, f32), r: f32, color: Color) {
        self.body.push_str(&format!(
            "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"{r:.2}\" fill=\"{}\" fill-opacity=\"{:.3}\"/>\n",
            c.0,
            c.1,
            svg_color(color),
            color.a
        ));
    }

    fn stroke_circle(&mut self, c: (f32, f32), r: f32, color: Color, width: f32) {
        self.body.push_str(&format!(
            "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"{r:.2}\" fill=\"none\" stroke=\"{}\" \
             stroke-opacity=\"{:.3}\" stroke-width=\"{width:.2}\"/>\n",
            c.0,
            c.1,
            svg_color(color),
            color.a
        ));
    }

    fn text(
        &mut self,
        content: &str,
        x: f32,
        y: f32,
        size: f32,
        color: Color,
        h: HAlign,
        v: VAlign,
        rotated_ccw: bool,
    ) {
        let anchor = match h {
            HAlign::Left => "start",
            HAlign::Center => "middle",
            HAlign::Right => "end",
        };
        // dominant-baseline approximates iced's vertical alignment.
        let baseline = match v {
            VAlign::Top => "hanging",
            VAlign::Center => "central",
            VAlign::Bottom => "text-bottom",
        };
        let escaped = content
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;");
        let transform = if rotated_ccw {
            format!(" transform=\"rotate(-90 {x:.2} {y:.2})\"")
        } else {
            String::new()
        };
        self.body.push_str(&format!(
            "<text x=\"{x:.2}\" y=\"{y:.2}\" font-family=\"Helvetica, Arial, sans-serif\" \
             font-size=\"{size:.1}\" fill=\"{}\" fill-opacity=\"{:.3}\" text-anchor=\"{anchor}\" \
             dominant-baseline=\"{baseline}\"{transform}>{escaped}</text>\n",
            svg_color(color),
            color.a
        ));
    }
}

// ---------------------------------------------------------------------------
// Canvas program: events (click-to-edit, camera, region select) + draw.
// ---------------------------------------------------------------------------

struct PlotProgram<'a> {
    pane: &'a PlotPane,
}

/// Per-canvas interaction state.
#[derive(Debug, Clone, Copy, Default)]
pub struct DragState {
    /// 3D camera drag: button + last cursor position.
    drag: Option<(mouse::Button, Point)>,
    /// Region rubber band: anchor + current corner.
    select: Option<(Point, Point)>,
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
        let is_3d = matches!(
            self.pane.spec.kind,
            PlotKind::Scatter3D { .. } | PlotKind::Sphere { .. }
        );
        let selectable = self.pane.supports_region_select();

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
                    if selectable {
                        let pa = layout.plot_area;
                        let inside = pos.x >= pa.x
                            && pos.x <= pa.x + pa.width
                            && pos.y >= pa.y
                            && pos.y <= pa.y + pa.height;
                        if inside {
                            state.select = Some((pos, pos));
                            return Some(canvas::Action::capture());
                        }
                    }
                }
                if is_3d && matches!(button, mouse::Button::Left | mouse::Button::Right) {
                    state.drag = Some((*button, pos));
                    return Some(canvas::Action::capture());
                }
                None
            }
            Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                if let Some((anchor, _)) = state.select {
                    let pos = cursor.position_in(bounds)?;
                    state.select = Some((anchor, pos));
                    return Some(canvas::Action::request_redraw().and_capture());
                }
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
                if let Some((a, b)) = state.select.take() {
                    // Ignore accidental micro-drags.
                    if (a.x - b.x).abs() > 5.0 && (a.y - b.y).abs() > 5.0 {
                        if let Some((_, xs, ys)) =
                            cartesian_scales(self.pane, bounds.width, bounds.height)
                        {
                            let (x0, x1) =
                                (xs.from_screen(a.x.min(b.x)), xs.from_screen(a.x.max(b.x)));
                            // Screen y grows downward; from_screen handles
                            // the inverted range, sort afterwards.
                            let (ya, yb) = (ys.from_screen(a.y), ys.from_screen(b.y));
                            return Some(canvas::Action::publish(Message::Plot(
                                PlotMessage::RegionSelected {
                                    x: (x0, x1),
                                    y: (ya.min(yb), ya.max(yb)),
                                },
                            )));
                        }
                    }
                    return Some(canvas::Action::request_redraw());
                }
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
        state: &Self::State,
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
        let mut out = vec![geometry];

        // Rubber band drawn uncached so it follows the cursor cheaply.
        if let Some((a, b)) = state.select {
            let mut frame = canvas::Frame::new(renderer, bounds.size());
            let (x, y) = (a.x.min(b.x), a.y.min(b.y));
            let (w, h) = ((a.x - b.x).abs(), (a.y - b.y).abs());
            frame.fill_rectangle(
                Point::new(x, y),
                Size::new(w, h),
                Color::from_rgba(0.267, 0.467, 0.667, 0.15),
            );
            frame.stroke_rectangle(
                Point::new(x, y),
                Size::new(w, h),
                Stroke::default()
                    .with_width(1.0)
                    .with_color(Color::from_rgba(0.267, 0.467, 0.667, 0.9)),
            );
            out.push(frame.into_geometry());
        }
        out
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
        if state.select.is_some() {
            return mouse::Interaction::Crosshair;
        }
        if let Some(pos) = cursor.position_in(bounds) {
            let layout = PlotLayout::compute(&self.pane.spec, bounds.width, bounds.height);
            if layout.hit_test(PointF { x: pos.x, y: pos.y }).is_some() {
                return mouse::Interaction::Pointer;
            }
            if matches!(
                self.pane.spec.kind,
                PlotKind::Scatter3D { .. } | PlotKind::Sphere { .. }
            ) {
                return mouse::Interaction::Grab;
            }
            if self.pane.supports_region_select() {
                let pa = layout.plot_area;
                if pos.x >= pa.x
                    && pos.x <= pa.x + pa.width
                    && pos.y >= pa.y
                    && pos.y <= pa.y + pa.height
                {
                    return mouse::Interaction::Crosshair;
                }
            }
        }
        mouse::Interaction::default()
    }
}

/// Layout + x/y scales for cartesian plots (None for sky/sphere/3D).
fn cartesian_scales(
    pane: &PlotPane,
    width: f32,
    height: f32,
) -> Option<(PlotLayout, LinearScale, LinearScale)> {
    if !pane.spec.kind.has_axes() {
        return None;
    }
    let layout = PlotLayout::compute(&pane.spec, width, height);
    let pa = layout.plot_area;
    let (x_domain, y_domain) = domains(&pane.spec, &pane.layers);
    let x_scale = LinearScale::new(x_domain, (pa.x, pa.x + pa.width));
    let y_scale = LinearScale::new(y_domain, (pa.y + pa.height, pa.y));
    Some((layout, x_scale, y_scale))
}

fn color(c: Rgba) -> Color {
    Color::from_rgba(c.r, c.g, c.b, c.a)
}

/// Style of the i-th layer (owned default if the spec has none, which only
/// happens transiently before the first layer lands).
fn layer_style(spec: &PlotSpec, i: usize) -> LayerStyle {
    spec.layers
        .get(i)
        .cloned()
        .unwrap_or_else(|| LayerStyle::new("", i))
}

const FRAME_COLOR: Color = Color::BLACK;
const GRID_COLOR: Color = Color::from_rgba(0.0, 0.0, 0.0, 0.12);
const GRATICULE_COLOR: Color = Color::from_rgba(0.0, 0.0, 0.0, 0.18);
const ANNOTATION_COLOR: Color = Color::from_rgba(0.0, 0.0, 0.0, 0.45);

/// Render the whole figure. `editing` suppresses that element's text so the
/// floating text_input overlay visually replaces it.
fn draw_plot<S: Surface>(
    s: &mut S,
    spec: &PlotSpec,
    layers: &[PlotData],
    camera: &Camera3D,
    editing: Option<TextElement>,
) {
    let (width, height) = s.size();
    let layout = PlotLayout::compute(spec, width, height);

    // Figures are white regardless of app theme: they should look like the
    // exported (paper) version at all times.
    s.fill_rect(0.0, 0.0, width, height, Color::WHITE);

    if spec.kind.has_axes() {
        let pa = layout.plot_area;
        let (x_domain, y_domain) = domains(spec, layers);
        let x_scale = LinearScale::new(x_domain, (pa.x, pa.x + pa.width));
        let y_scale = LinearScale::new(y_domain, (pa.y + pa.height, pa.y));

        draw_axes(s, spec, &layout, &x_scale, &y_scale);

        for (i, data) in layers.iter().enumerate() {
            let style = layer_style(spec, i);
            match data {
                PlotData::Scatter(points) => draw_scatter(s, &style, points, &x_scale, &y_scale),
                PlotData::Histogram(hist) => draw_histogram(s, &style, hist, &x_scale, &y_scale),
                PlotData::Line(points) => draw_line(s, &style, points, &x_scale, &y_scale),
                _ => {}
            }
        }
        downsample_note(s, spec, layers, layout.plot_area);
    } else {
        match &spec.kind {
            PlotKind::Sky { projection, .. } => draw_sky(s, spec, layers, *projection, &layout),
            PlotKind::Scatter3D { .. } => draw_3d(s, spec, layers, camera, &layout),
            PlotKind::Sphere { .. } => draw_sphere(s, spec, layers, camera, &layout),
            _ => {}
        }
    }

    draw_legend(s, spec, &layout, editing);
    draw_labels(s, spec, &layout, editing);
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

/// Frame, gridlines, and mirrored inward ticks on all four sides — the
/// journal-figure convention.
fn draw_axes<S: Surface>(
    s: &mut S,
    spec: &PlotSpec,
    layout: &PlotLayout,
    x_scale: &LinearScale,
    y_scale: &LinearScale,
) {
    let pa = layout.plot_area;
    let tw = spec.tick_width;
    let tl = spec.tick_length;
    let (top, bottom) = (pa.y, pa.y + pa.height);
    let (left, right) = (pa.x, pa.x + pa.width);

    let x_ticks = scale::nice_ticks(x_scale.domain.0, x_scale.domain.1, spec.x_axis.tick_target);
    let x_step = scale::tick_step(&x_ticks);
    for &t in &x_ticks {
        let sx = x_scale.to_screen(t);
        if spec.show_grid {
            s.line((sx, top), (sx, bottom), GRID_COLOR, 1.0);
        }
        s.line((sx, bottom), (sx, bottom - tl), FRAME_COLOR, tw);
        s.line((sx, top), (sx, top + tl), FRAME_COLOR, tw);
        s.text(
            &scale::format_tick(t, x_step),
            sx,
            bottom + 4.0,
            spec.tick_label_size,
            FRAME_COLOR,
            HAlign::Center,
            VAlign::Top,
            false,
        );
    }

    let y_ticks = scale::nice_ticks(y_scale.domain.0, y_scale.domain.1, spec.y_axis.tick_target);
    let y_step = scale::tick_step(&y_ticks);
    for &t in &y_ticks {
        let sy = y_scale.to_screen(t);
        if spec.show_grid {
            s.line((left, sy), (right, sy), GRID_COLOR, 1.0);
        }
        s.line((left, sy), (left + tl, sy), FRAME_COLOR, tw);
        s.line((right, sy), (right - tl, sy), FRAME_COLOR, tw);
        s.text(
            &scale::format_tick(t, y_step),
            left - 6.0,
            sy,
            spec.tick_label_size,
            FRAME_COLOR,
            HAlign::Right,
            VAlign::Center,
            false,
        );
    }

    // Frame last so it sits on top of gridlines.
    s.stroke_rect(pa.x, pa.y, pa.width, pa.height, FRAME_COLOR, tw);
}

fn draw_scatter<S: Surface>(
    s: &mut S,
    style: &LayerStyle,
    points: &ScatterData,
    x_scale: &LinearScale,
    y_scale: &LinearScale,
) {
    let mc = color(style.color);
    let (x0, x1) = x_scale.domain;
    let (y0, y1) = y_scale.domain;
    let r = style.marker_size;

    for i in 0..points.x.len() {
        let (dx, dy) = (points.x[i], points.y[i]);
        if dx < x0 || dx > x1 || dy < y0 || dy > y1 {
            continue; // outside current limits
        }
        let (px, py) = (x_scale.to_screen(dx), y_scale.to_screen(dy));

        if let Some(errs) = &points.x_err {
            let e = errs[i];
            let ax = x_scale.to_screen(dx - e);
            let bx = x_scale.to_screen(dx + e);
            let cap = r.max(2.0);
            s.line((ax, py), (bx, py), mc, 1.0);
            s.line((ax, py - cap), (ax, py + cap), mc, 1.0);
            s.line((bx, py - cap), (bx, py + cap), mc, 1.0);
        }
        if let Some(errs) = &points.y_err {
            let e = errs[i];
            let ay = y_scale.to_screen(dy - e);
            let by = y_scale.to_screen(dy + e);
            let cap = r.max(2.0);
            s.line((px, ay), (px, by), mc, 1.0);
            s.line((px - cap, ay), (px + cap, ay), mc, 1.0);
            s.line((px - cap, by), (px + cap, by), mc, 1.0);
        }

        draw_marker(s, (px, py), style, mc);
    }
}

fn draw_line<S: Surface>(
    s: &mut S,
    style: &LayerStyle,
    points: &ScatterData,
    x_scale: &LinearScale,
    y_scale: &LinearScale,
) {
    let pts: Vec<(f32, f32)> = (0..points.x.len())
        .map(|i| {
            (
                x_scale.to_screen(points.x[i]),
                y_scale.to_screen(points.y[i]),
            )
        })
        .collect();
    s.polyline(&pts, color(style.color), style.line_width);
}

fn draw_sky<S: Surface>(
    s: &mut S,
    spec: &PlotSpec,
    layers: &[PlotData],
    projection: SkyProjection,
    layout: &PlotLayout,
) {
    let pa = layout.plot_area;
    let (xm, ym) = projection.extent();
    let sc = ((pa.width as f64 / (2.0 * xm)).min(pa.height as f64 / (2.0 * ym)) * 0.97) as f32;
    let cx = pa.x + pa.width / 2.0;
    let cy = pa.y + pa.height / 2.0;
    let deg = std::f64::consts::PI / 180.0;

    // Sky convention: longitude (RA) increases to the LEFT, map centered on
    // lon = 180°.
    let to_screen = |px: f64, py: f64| -> (f32, f32) { (cx - sc * px as f32, cy - sc * py as f32) };
    let project_line = |pts: &[(f64, f64)]| -> Vec<(f32, f32)> {
        pts.iter().map(|&(px, py)| to_screen(px, py)).collect()
    };

    for line in projection.graticule(30.0, 30.0) {
        s.polyline(&project_line(&line), GRATICULE_COLOR, 1.0);
    }
    s.polyline(
        &project_line(&projection.boundary(60)),
        FRAME_COLOR,
        spec.tick_width.max(1.2),
    );

    // Coordinate annotations: RA along the equator, Dec down the center.
    for lon in [0.0f64, 60.0, 120.0, 180.0, 240.0, 300.0] {
        let (px, py) = projection.project((lon - 180.0) * deg, 0.0);
        let (tx, ty) = to_screen(px, py);
        s.text(
            &format!("{lon:.0}°"),
            tx,
            ty,
            (spec.tick_label_size - 1.0).max(8.0),
            ANNOTATION_COLOR,
            HAlign::Center,
            VAlign::Bottom,
            false,
        );
    }
    for lat in [-60.0f64, -30.0, 30.0, 60.0] {
        let (px, py) = projection.project(0.0, lat * deg);
        let (tx, ty) = to_screen(px, py);
        s.text(
            &format!("{lat:+.0}°"),
            tx,
            ty,
            (spec.tick_label_size - 1.0).max(8.0),
            ANNOTATION_COLOR,
            HAlign::Center,
            VAlign::Center,
            false,
        );
    }

    for (i, data) in layers.iter().enumerate() {
        let Some(points) = scatter_like(data) else {
            continue;
        };
        let style = layer_style(spec, i);
        let mc = color(style.color);
        for j in 0..points.x.len() {
            let lon = points.x[j];
            let lat = points.y[j];
            if !(-90.0..=90.0).contains(&lat) {
                continue;
            }
            // Wrap RA into [0, 360), then center.
            let lon = lon.rem_euclid(360.0);
            let (px, py) = projection.project((lon - 180.0) * deg, lat * deg);
            draw_marker(s, to_screen(px, py), &style, mc);
        }
    }

    downsample_note(s, spec, layers, pa);
}

fn draw_3d<S: Surface>(
    s: &mut S,
    spec: &PlotSpec,
    layers: &[PlotData],
    camera: &Camera3D,
    layout: &PlotLayout,
) {
    let pa = layout.plot_area;
    let scale_px = camera.zoom * pa.width.min(pa.height) * 0.55;
    let cx = pa.x + pa.width / 2.0 + camera.pan.0;
    let cy = pa.y + pa.height / 2.0 + camera.pan.1;
    let (syaw, cyaw) = camera.yaw.sin_cos();
    let (sp, cp) = camera.pitch.sin_cos();

    // Orthographic camera: yaw about the vertical axis, then pitch; screen
    // x is rotated x, screen y is (negated) rotated z. Depth is unused.
    let project = |x: f32, y: f32, z: f32| -> (f32, f32) {
        let x1 = x * cyaw - y * syaw;
        let y1 = x * syaw + y * cyaw;
        let z2 = y1 * sp + z * cp;
        (cx + x1 * scale_px, cy - z2 * scale_px)
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
    let corners: Vec<(f32, f32)> = (0..8)
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
        s.line(corners[a], corners[b], GRATICULE_COLOR, 1.0);
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
            s.text(
                label,
                end.0,
                end.1,
                spec.tick_label_size,
                FRAME_COLOR,
                HAlign::Center,
                VAlign::Center,
                false,
            );
        }
    }

    for (i, data) in layers.iter().enumerate() {
        let PlotData::Xyz(points) = data else {
            continue;
        };
        let style = layer_style(spec, i);
        let mc = color(style.color);
        let (xs, ys, zs) = (&points.columns[0], &points.columns[1], &points.columns[2]);
        for j in 0..xs.len() {
            let p = project(
                norm(xs[j], ranges[0]),
                norm(ys[j], ranges[1]),
                norm(zs[j], ranges[2]),
            );
            draw_marker(s, p, &style, mc);
        }
    }

    downsample_note(s, spec, layers, pa);
    annotation(
        s,
        spec,
        "drag rotate · right-drag pan · scroll zoom",
        pa.x + pa.width - 4.0,
        pa.y + pa.height - 14.0,
    );
}

/// Celestial sphere: points at (lon, lat[, distance]) inside a wireframe
/// sphere, seen through the shared 3D camera. Far-hemisphere points are
/// faded so the front of the sky reads clearly.
fn draw_sphere<S: Surface>(
    s: &mut S,
    spec: &PlotSpec,
    layers: &[PlotData],
    camera: &Camera3D,
    layout: &PlotLayout,
) {
    let pa = layout.plot_area;
    let scale_px = camera.zoom * pa.width.min(pa.height) * 0.45;
    let cx = pa.x + pa.width / 2.0 + camera.pan.0;
    let cy = pa.y + pa.height / 2.0 + camera.pan.1;
    let (syaw, cyaw) = camera.yaw.sin_cos();
    let (sp, cp) = camera.pitch.sin_cos();

    // Same orthographic camera as the 3D cube, but returning depth so the
    // far hemisphere can be faded (positive depth = away from the viewer).
    let project = |x: f64, y: f64, z: f64| -> ((f32, f32), f32) {
        let (x, y, z) = (x as f32, y as f32, z as f32);
        let x1 = x * cyaw - y * syaw;
        let y1 = x * syaw + y * cyaw;
        let depth = y1 * cp - z * sp;
        let z2 = y1 * sp + z * cp;
        ((cx + x1 * scale_px, cy - z2 * scale_px), depth)
    };

    // Radial scale: unit sphere unless distances are present, then the
    // largest distance across all layers touches the wireframe.
    let mut dmax = f64::NEG_INFINITY;
    for data in layers {
        if let PlotData::Sphere(points) = data {
            if let Some(dist) = points.columns.get(2) {
                let (_, hi) = min_max(dist);
                dmax = dmax.max(hi);
            }
        }
    }
    let has_dist = dmax.is_finite() && dmax > 0.0;

    for line in sphere_wireframe(30.0, 30.0, 48) {
        let pts: Vec<(f32, f32)> = line.iter().map(|&(x, y, z)| project(x, y, z).0).collect();
        s.polyline(&pts, GRATICULE_COLOR, 1.0);
    }

    // Pole labels orient the viewer.
    for (label, lat) in [("N", 90.0), ("S", -90.0)] {
        let (x, y, z) = sky_to_xyz(0.0, lat, 1.1);
        let (p, _) = project(x, y, z);
        s.text(
            label,
            p.0,
            p.1,
            spec.tick_label_size,
            ANNOTATION_COLOR,
            HAlign::Center,
            VAlign::Center,
            false,
        );
    }

    for (i, data) in layers.iter().enumerate() {
        let PlotData::Sphere(points) = data else {
            continue;
        };
        let style = layer_style(spec, i);
        let mc = color(style.color);
        let faded = Color {
            a: mc.a * 0.25,
            ..mc
        };
        let (lons, lats) = (&points.columns[0], &points.columns[1]);
        let dists = points.columns.get(2);
        for j in 0..lons.len() {
            let lat = lats[j];
            if !(-90.0..=90.0).contains(&lat) {
                continue;
            }
            let r = match (has_dist, dists) {
                (true, Some(d)) => d[j] / dmax,
                _ => 1.0,
            };
            if !(0.0..=1.0).contains(&r) {
                continue; // negative distances have no home on this plot
            }
            let (x, y, z) = sky_to_xyz(lons[j].rem_euclid(360.0), lat, r);
            let (p, depth) = project(x, y, z);
            draw_marker(s, p, &style, if depth > 0.0 { faded } else { mc });
        }
    }

    downsample_note(s, spec, layers, pa);
    annotation(
        s,
        spec,
        "drag rotate · right-drag pan · scroll zoom",
        pa.x + pa.width - 4.0,
        pa.y + pa.height - 14.0,
    );
}

fn annotation<S: Surface>(s: &mut S, spec: &PlotSpec, content: &str, x: f32, y: f32) {
    s.text(
        content,
        x,
        y,
        (spec.tick_label_size - 1.0).max(8.0),
        ANNOTATION_COLOR,
        HAlign::Right,
        VAlign::Top,
        false,
    );
}

/// One shared honesty note when any layer was stride-downsampled.
fn downsample_note<S: Surface>(s: &mut S, spec: &PlotSpec, layers: &[PlotData], pa: RectF) {
    let max_stride = layers
        .iter()
        .filter_map(|d| match d {
            PlotData::Scatter(p) | PlotData::Line(p) | PlotData::Sky(p) => Some(p.stride),
            PlotData::Xyz(p) | PlotData::Sphere(p) => Some(p.stride),
            PlotData::Histogram(_) => None,
        })
        .max()
        .unwrap_or(1);
    if max_stride > 1 {
        // Legend occupies the top-right; place the note bottom-left inside
        // the plot area to avoid collisions.
        s.text(
            &format!("downsampled (up to every {max_stride}th point)"),
            pa.x + 4.0,
            pa.y + pa.height - 4.0,
            (spec.tick_label_size - 1.0).max(8.0),
            ANNOTATION_COLOR,
            HAlign::Left,
            VAlign::Bottom,
            false,
        );
    }
}

/// Draw one marker per the layer style (shape, size, filled/open) in the
/// given color (callers may override the style color, e.g. depth fading).
fn draw_marker<S: Surface>(s: &mut S, p: (f32, f32), style: &LayerStyle, color: Color) {
    let r = style.marker_size;
    let lw = style.line_width.max(1.0);
    match style.marker {
        MarkerShape::Circle => {
            if style.filled {
                s.fill_circle(p, r, color);
            } else {
                s.stroke_circle(p, r, color, lw);
            }
        }
        MarkerShape::Square => {
            if style.filled {
                s.fill_rect(p.0 - r, p.1 - r, 2.0 * r, 2.0 * r, color);
            } else {
                s.stroke_rect(p.0 - r, p.1 - r, 2.0 * r, 2.0 * r, color, lw);
            }
        }
        MarkerShape::Diamond => {
            let pts = [
                (p.0, p.1 - r * 1.3),
                (p.0 + r * 1.3, p.1),
                (p.0, p.1 + r * 1.3),
                (p.0 - r * 1.3, p.1),
            ];
            if style.filled {
                s.fill_polygon(&pts, color);
            } else {
                let closed = [pts[0], pts[1], pts[2], pts[3], pts[0]];
                s.polyline(&closed, color, lw);
            }
        }
        // A cross has no interior: filled/open draw identically.
        MarkerShape::Cross => {
            let lw = style.line_width.max(1.5);
            s.line((p.0 - r, p.1), (p.0 + r, p.1), color, lw);
            s.line((p.0, p.1 - r), (p.0, p.1 + r), color, lw);
        }
    }
}

fn draw_histogram<S: Surface>(
    s: &mut S,
    style: &LayerStyle,
    hist: &Histogram,
    x_scale: &LinearScale,
    y_scale: &LinearScale,
) {
    let mc = color(style.color);
    let base = y_scale.to_screen(0.0);

    match style.hist_style {
        HistStyle::Bars => {
            // Translucent so overlaid layer histograms stay readable.
            let fill = Color { a: 0.55, ..mc };
            for (i, &count) in hist.counts.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let left = x_scale.to_screen(hist.edges[i]);
                let right = x_scale.to_screen(hist.edges[i + 1]);
                let top = y_scale.to_screen(count as f64);
                s.fill_rect(left, top, right - left, base - top, fill);
                s.stroke_rect(left, top, right - left, base - top, mc, style.line_width);
            }
        }
        HistStyle::Steps => {
            // One staircase outline: across each bin top, closing to zero at
            // the ends. Zero-count bins draw at the baseline.
            let mut pts = Vec::with_capacity(hist.counts.len() * 2 + 2);
            pts.push((x_scale.to_screen(hist.edges[0]), base));
            for (i, &count) in hist.counts.iter().enumerate() {
                let left = x_scale.to_screen(hist.edges[i]);
                let right = x_scale.to_screen(hist.edges[i + 1]);
                let top = y_scale.to_screen(count as f64);
                pts.push((left, top));
                pts.push((right, top));
            }
            pts.push((x_scale.to_screen(hist.edges[hist.counts.len()]), base));
            s.polyline(&pts, mc, style.line_width);
        }
    }
}

/// Legend box with color swatches; entry labels are click-to-edit.
fn draw_legend<S: Surface>(
    s: &mut S,
    spec: &PlotSpec,
    layout: &PlotLayout,
    editing: Option<TextElement>,
) {
    if layout.legend_labels.is_empty() {
        return;
    }
    let b = layout.legend_box;
    s.fill_rect(
        b.x,
        b.y,
        b.width,
        b.height,
        Color::from_rgba(1.0, 1.0, 1.0, 0.9),
    );
    s.stroke_rect(b.x, b.y, b.width, b.height, FRAME_COLOR, spec.tick_width);

    for (i, layer) in spec.layers.iter().enumerate() {
        let sw = layout.legend_swatches[i];
        s.fill_rect(sw.x, sw.y, sw.width, sw.height, color(layer.color));
        if editing == Some(TextElement::Legend(i)) {
            continue;
        }
        let lr = layout.legend_labels[i];
        s.text(
            &layer.label,
            lr.x,
            lr.y + lr.height / 2.0,
            spec.tick_label_size,
            FRAME_COLOR,
            HAlign::Left,
            VAlign::Center,
            false,
        );
    }
}

fn draw_labels<S: Surface>(
    s: &mut S,
    spec: &PlotSpec,
    layout: &PlotLayout,
    editing: Option<TextElement>,
) {
    let center = |r: RectF| (r.x + r.width / 2.0, r.y + r.height / 2.0);

    if editing != Some(TextElement::Title) && !spec.title.is_empty() {
        let (x, y) = center(layout.title_rect);
        s.text(
            &spec.title,
            x,
            y,
            spec.title_size,
            FRAME_COLOR,
            HAlign::Center,
            VAlign::Center,
            false,
        );
    }

    if !spec.kind.has_axes() {
        return; // sky/sphere/3D annotate themselves
    }

    if editing != Some(TextElement::XLabel) {
        let (x, y) = center(layout.x_label_rect);
        s.text(
            &spec.x_axis.label,
            x,
            y,
            spec.label_size,
            FRAME_COLOR,
            HAlign::Center,
            VAlign::Center,
            false,
        );
    }

    if editing != Some(TextElement::YLabel) {
        // Rotated 90° CCW around the label rect's center.
        let (x, y) = center(layout.y_label_rect);
        s.text(
            &spec.y_axis.label,
            x,
            y,
            spec.label_size,
            FRAME_COLOR,
            HAlign::Center,
            VAlign::Center,
            true,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use topdog_core::ScatterData;

    fn scatter_pane() -> PlotPane {
        let data = ScatterData {
            x: vec![1.0, 2.0, 3.0],
            y: vec![2.0, 4.0, 6.0],
            x_err: None,
            y_err: Some(vec![0.1, 0.2, 0.3]),
            total_points: 3,
            stride: 1,
        };
        let mut spec = PlotSpec::scatter("ra", "dec", None, Some("e".into()));
        spec.title = "Test <figure> & more".to_string();
        let mut pane = PlotPane::new(spec, Vec::new());
        pane.add_layer("all".to_string(), PlotData::Scatter(data.clone()));
        pane.add_layer("bright".to_string(), PlotData::Scatter(data));
        pane
    }

    #[test]
    fn svg_export_is_wellformed_and_contains_content() {
        let svg = scatter_pane().to_svg(800.0, 600.0);
        assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
        assert!(svg.trim_end().ends_with("</svg>"));
        assert!(svg.contains("width=\"800\""));
        // Axis labels, legend entries, and markers all made it in.
        assert!(svg.contains(">ra<"), "x label missing");
        assert!(svg.contains(">dec<"), "y label missing");
        assert!(svg.contains(">bright<"), "legend entry missing");
        assert!(svg.contains("<circle"), "markers missing");
        // XML special characters in user text are escaped.
        assert!(svg.contains("Test &lt;figure&gt; &amp; more"));
        assert!(!svg.contains("<figure>"));
    }

    #[test]
    fn svg_export_renders_every_plot_kind() {
        let kinds: Vec<(PlotSpec, PlotData)> = vec![
            (
                PlotSpec::histogram("mag", 0, false),
                PlotData::Histogram(topdog_core::Histogram {
                    edges: vec![0.0, 1.0, 2.0],
                    counts: vec![3, 5],
                    log_bins: false,
                }),
            ),
            (
                PlotSpec::sky("ra", "dec", SkyProjection::Mollweide),
                PlotData::Sky(ScatterData {
                    x: vec![10.0, 200.0],
                    y: vec![-20.0, 45.0],
                    x_err: None,
                    y_err: None,
                    total_points: 2,
                    stride: 1,
                }),
            ),
            (
                PlotSpec::scatter3d("x", "y", "z"),
                PlotData::Xyz(topdog_core::ColumnsData {
                    columns: vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
                    total_points: 2,
                    stride: 1,
                }),
            ),
            (
                PlotSpec::sphere("ra", "dec", Some("dist".into())),
                PlotData::Sphere(topdog_core::ColumnsData {
                    columns: vec![vec![10.0, 200.0], vec![-20.0, 45.0], vec![1.0, 2.0]],
                    total_points: 2,
                    stride: 1,
                }),
            ),
        ];
        for (spec, data) in kinds {
            let mut pane = PlotPane::new(spec, Vec::new());
            pane.add_layer("layer".to_string(), data);
            let svg = pane.to_svg(640.0, 480.0);
            assert!(svg.trim_end().ends_with("</svg>"));
            // Something beyond the white background was drawn.
            assert!(
                svg.matches('<').count() > 3,
                "kind rendered nothing: {svg:?}"
            );
        }
    }

    #[test]
    fn region_select_only_for_cartesian_kinds() {
        assert!(scatter_pane().supports_region_select());
        let sky = PlotPane::new(
            PlotSpec::sky("ra", "dec", SkyProjection::Aitoff),
            Vec::new(),
        );
        assert!(!sky.supports_region_select());
    }
}
