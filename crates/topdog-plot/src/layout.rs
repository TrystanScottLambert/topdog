//! Plot layout geometry: where everything lands for a given canvas size.
//!
//! Computed as a pure function of ([`PlotSpec`], width, height) so the same
//! rects drive both drawing and click hit-testing — this is the mechanism
//! behind click-to-edit labels (CLAUDE.md §5): the renderer draws text at
//! these rects, and a click inside one identifies the element to edit.

use crate::geometry::{PointF, RectF};
use crate::spec::PlotSpec;

/// A clickable, editable text element of a plot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextElement {
    Title,
    XLabel,
    YLabel,
    /// The label of the i-th legend entry.
    Legend(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlotLayout {
    /// The data region (inside the axes frame).
    pub plot_area: RectF,
    pub title_rect: RectF,
    pub x_label_rect: RectF,
    /// Screen-space rect of the (rotated) y label: a vertical strip.
    pub y_label_rect: RectF,
    /// Background box of the legend; zero-sized when no legend is drawn.
    pub legend_box: RectF,
    /// Label rect per legend entry (the editable part, after the swatch).
    pub legend_labels: Vec<RectF>,
    /// Color-swatch rect per legend entry.
    pub legend_swatches: Vec<RectF>,
}

/// Estimated width of `text` at `size` px. We have no font shaper here;
/// 0.55 em/char is close enough for hit rects and margin sizing.
pub fn text_width(text: &str, size: f32) -> f32 {
    text.chars().count() as f32 * size * 0.55
}

pub fn text_height(size: f32) -> f32 {
    size * 1.3
}

impl PlotLayout {
    pub fn compute(spec: &PlotSpec, width: f32, height: f32) -> Self {
        let has_axes = spec.kind.has_axes();
        let tick_h = text_height(spec.tick_label_size);
        // Reserve room for the widest plausible tick label on the left.
        let tick_w = 7.0 * spec.tick_label_size * 0.55;

        let title_h = if spec.title.is_empty() {
            text_height(spec.title_size) * 0.6 // breathing room even untitled
        } else {
            text_height(spec.title_size)
        };
        let label_h = text_height(spec.label_size);

        let margin_top = title_h + 8.0;
        // Axis-less plots (sky, 3D) annotate themselves; give them the room.
        let (margin_bottom, margin_left, margin_right) = if has_axes {
            (tick_h + label_h + 14.0, tick_w + label_h + 12.0, 14.0)
        } else {
            (14.0, 14.0, 14.0)
        };

        let plot_area = RectF::new(
            margin_left,
            margin_top,
            (width - margin_left - margin_right).max(10.0),
            (height - margin_top - margin_bottom).max(10.0),
        );

        // Title: centered over the plot area. When empty we still expose a
        // clickable strip so a title can be added by clicking where it goes.
        let title_w = if spec.title.is_empty() {
            plot_area.width * 0.5
        } else {
            text_width(&spec.title, spec.title_size) + 20.0
        };
        let title_rect = RectF::new(
            plot_area.x + (plot_area.width - title_w) / 2.0,
            2.0,
            title_w,
            text_height(spec.title_size),
        );

        let (x_label_rect, y_label_rect) = if has_axes {
            let x_label_w = text_width(&spec.x_axis.label, spec.label_size).max(60.0) + 20.0;
            let x_label_rect = RectF::new(
                plot_area.x + (plot_area.width - x_label_w) / 2.0,
                plot_area.y + plot_area.height + tick_h + 4.0,
                x_label_w,
                label_h,
            );
            let y_label_len = text_width(&spec.y_axis.label, spec.label_size).max(60.0) + 20.0;
            let y_label_rect = RectF::new(
                2.0,
                plot_area.y + (plot_area.height - y_label_len) / 2.0,
                label_h,
                y_label_len,
            );
            (x_label_rect, y_label_rect)
        } else {
            // Zero-sized: never drawn, never hit.
            (RectF::default(), RectF::default())
        };

        // Legend: top-right inside the plot area, one row per layer.
        // Drawn (and clickable) only with 2+ layers — a single layer's
        // identity is already in the axis labels.
        let (legend_box, legend_labels, legend_swatches) = if spec.layers.len() >= 2 {
            let entry_h = text_height(spec.tick_label_size) * 1.15;
            let swatch_w = 14.0;
            let gap = 6.0;
            let pad = 8.0;
            let max_label_w = spec
                .layers
                .iter()
                .map(|l| text_width(&l.label, spec.tick_label_size))
                .fold(30.0f32, f32::max);
            let box_w = pad + swatch_w + gap + max_label_w + pad;
            let box_h = pad + entry_h * spec.layers.len() as f32 + pad;
            let bx = plot_area.x + plot_area.width - box_w - 8.0;
            let by = plot_area.y + 8.0;
            let legend_box = RectF::new(bx, by, box_w, box_h);
            let mut labels = Vec::with_capacity(spec.layers.len());
            let mut swatches = Vec::with_capacity(spec.layers.len());
            for i in 0..spec.layers.len() {
                let ey = by + pad + entry_h * i as f32;
                swatches.push(RectF::new(
                    bx + pad,
                    ey + (entry_h - 10.0) / 2.0,
                    swatch_w,
                    10.0,
                ));
                labels.push(RectF::new(
                    bx + pad + swatch_w + gap,
                    ey,
                    max_label_w,
                    entry_h,
                ));
            }
            (legend_box, labels, swatches)
        } else {
            (RectF::default(), Vec::new(), Vec::new())
        };

        Self {
            plot_area,
            title_rect,
            x_label_rect,
            y_label_rect,
            legend_box,
            legend_labels,
            legend_swatches,
        }
    }

    /// Which editable text element (if any) is under `p`.
    pub fn hit_test(&self, p: PointF) -> Option<TextElement> {
        let hit = |r: &RectF| r.width > 0.0 && r.contains(p);
        if hit(&self.title_rect) {
            return Some(TextElement::Title);
        }
        if hit(&self.x_label_rect) {
            return Some(TextElement::XLabel);
        }
        if hit(&self.y_label_rect) {
            return Some(TextElement::YLabel);
        }
        // Legend labels sit inside the plot area, so check before giving up.
        for (i, r) in self.legend_labels.iter().enumerate() {
            if hit(r) {
                return Some(TextElement::Legend(i));
            }
        }
        None
    }

    /// The rect for one element (for positioning the edit overlay).
    pub fn rect_of(&self, element: TextElement) -> RectF {
        match element {
            TextElement::Title => self.title_rect,
            TextElement::XLabel => self.x_label_rect,
            TextElement::YLabel => self.y_label_rect,
            TextElement::Legend(i) => self
                .legend_labels
                .get(i)
                .copied()
                .unwrap_or(self.legend_box),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::PlotSpec;

    fn spec() -> PlotSpec {
        let mut s = PlotSpec::scatter("ra", "dec", None, None);
        s.title = "My survey".to_string();
        s
    }

    #[test]
    fn plot_area_fits_inside_canvas() {
        let l = PlotLayout::compute(&spec(), 800.0, 600.0);
        assert!(l.plot_area.x > 0.0);
        assert!(l.plot_area.y > 0.0);
        assert!(l.plot_area.x + l.plot_area.width < 800.0);
        assert!(l.plot_area.y + l.plot_area.height < 600.0);
    }

    #[test]
    fn hit_test_finds_each_element() {
        let l = PlotLayout::compute(&spec(), 800.0, 600.0);
        let center = |r: RectF| PointF {
            x: r.x + r.width / 2.0,
            y: r.y + r.height / 2.0,
        };
        assert_eq!(l.hit_test(center(l.title_rect)), Some(TextElement::Title));
        assert_eq!(
            l.hit_test(center(l.x_label_rect)),
            Some(TextElement::XLabel)
        );
        assert_eq!(
            l.hit_test(center(l.y_label_rect)),
            Some(TextElement::YLabel)
        );
        let mid = center(l.plot_area);
        assert_eq!(l.hit_test(mid), None);
    }

    #[test]
    fn empty_title_still_clickable() {
        let mut s = spec();
        s.title.clear();
        let l = PlotLayout::compute(&s, 800.0, 600.0);
        assert!(l.title_rect.width > 0.0);
    }

    #[test]
    fn legend_appears_with_two_layers_and_is_editable() {
        use crate::spec::LayerStyle;
        let mut s = spec();
        let l = PlotLayout::compute(&s, 800.0, 600.0);
        assert!(l.legend_labels.is_empty()); // single layer: no legend

        s.layers = vec![LayerStyle::new("bright", 0), LayerStyle::new("faint", 1)];
        let l = PlotLayout::compute(&s, 800.0, 600.0);
        assert_eq!(l.legend_labels.len(), 2);
        assert!(l.legend_box.width > 0.0);
        let c = PointF {
            x: l.legend_labels[1].x + 5.0,
            y: l.legend_labels[1].y + 5.0,
        };
        assert_eq!(l.hit_test(c), Some(TextElement::Legend(1)));
        // Legend sits inside the plot area, top-right.
        assert!(l.legend_box.x > l.plot_area.x);
        assert!(l.legend_box.y >= l.plot_area.y);
    }

    #[test]
    fn axisless_kinds_have_no_label_rects() {
        let s = PlotSpec::sky("ra", "dec", crate::projection::SkyProjection::Aitoff);
        let l = PlotLayout::compute(&s, 800.0, 600.0);
        assert_eq!(l.x_label_rect.width, 0.0);
        assert_eq!(l.hit_test(PointF { x: 0.0, y: 0.0 }), None);
        // Title strip still clickable.
        assert!(l.title_rect.width > 0.0);
    }

    #[test]
    fn tiny_canvas_does_not_collapse() {
        let l = PlotLayout::compute(&spec(), 40.0, 30.0);
        assert!(l.plot_area.width >= 10.0);
        assert!(l.plot_area.height >= 10.0);
    }
}
