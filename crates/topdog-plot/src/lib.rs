//! Plot math and data model for topdog.
//!
//! This crate owns everything about a plot that is not data access (that's
//! `topdog-core`) and not widget plumbing (that's `topdog-gui`):
//!
//! - [`PlotSpec`]: the single source of truth for a plot's appearance. Every
//!   editable property lives here; the GUI renders it and writes edits back.
//! - Scale transforms and tick generation ([`scale`]).
//! - Layout geometry ([`layout`]): where the title, axis labels, and plot
//!   area land for a given canvas size — used both to draw and to hit-test
//!   clicks for the click-to-edit interaction.
//!
//! Kept dependency-free on purpose: correctness and control over tick
//! placement matter more than reuse here, and the GUI must not be the only
//! way to exercise this logic.

pub mod geometry;
pub mod layout;
pub mod projection;
pub mod scale;
pub mod spec;

pub use geometry::{PointF, RectF};
pub use layout::{PlotLayout, TextElement};
pub use projection::SkyProjection;
pub use scale::LinearScale;
pub use spec::{Axis, HistStyle, LayerStyle, MarkerShape, PlotKind, PlotSpec, Rgba};
