//! Core data engine for topdog.
//!
//! Everything here is GUI-agnostic: a [`DataTable`] wraps a lazily-scanned
//! polars `LazyFrame` plus per-column metadata, and only ever materializes
//! the row window or aggregate a caller explicitly asks for. This is the
//! layer that makes "open a 50 GB parquet file in under a second" possible —
//! nothing in this crate should collect a full table.

mod column;
mod error;
mod plotdata;
mod stats;
mod table;

pub use column::ColumnMeta;
pub use error::CoreError;
pub use plotdata::{ColumnsData, Histogram, ScatterData};
pub use stats::ColumnStats;
pub use table::{DataTable, RowWindow, SortSpec};

// Re-exported so downstream crates (topdog-io, topdog-gui) don't need to
// depend on polars directly for the common types they exchange with core.
pub use polars::prelude::{DataType, LazyFrame};
