//! File-format readers for topdog.
//!
//! Each supported format implements [`TableSource`], so the rest of the app
//! is format-agnostic: the GUI calls [`open_table`] and gets a lazy
//! `DataTable` back regardless of what's on disk. Parquet is the only
//! implementation today; FITS arrives in phase 2 behind the same trait.

mod parquet;

use std::path::Path;

use thiserror::Error;
use topdog_core::{CoreError, DataTable};

pub use parquet::ParquetSource;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("unsupported file type: {0}")]
    UnsupportedFormat(String),

    #[error("path is not valid UTF-8: {0}")]
    NonUtf8Path(String),

    #[error(transparent)]
    Core(#[from] CoreError),

    #[error("failed to open {path}: {source}")]
    Open {
        path: String,
        source: polars::error::PolarsError,
    },
}

/// A file format topdog can lazily scan into a [`DataTable`].
pub trait TableSource {
    /// Quick extension-based check, used to pick a source for a path.
    fn can_read(&self, path: &Path) -> bool;

    /// Lazily scan `path`. Must not read column data eagerly — opening a
    /// table should cost metadata reads only, per the RAM-footprint
    /// philosophy in CLAUDE.md.
    fn scan(&self, path: &Path) -> Result<DataTable, IoError>;
}

/// All registered sources, in preference order.
fn sources() -> Vec<Box<dyn TableSource>> {
    vec![Box::new(ParquetSource)]
}

/// Open a table by dispatching on file extension across registered sources.
pub fn open_table(path: &Path) -> Result<DataTable, IoError> {
    sources()
        .iter()
        .find(|s| s.can_read(path))
        .ok_or_else(|| IoError::UnsupportedFormat(path.display().to_string()))?
        .scan(path)
}

/// Display name for a loaded table: the file stem.
pub(crate) fn table_name(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}
