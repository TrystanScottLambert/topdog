use std::path::Path;

use polars::prelude::{LazyFrame, PlRefPath, ScanArgsParquet};
use topdog_core::DataTable;

use crate::{table_name, IoError, TableSource};

/// Parquet reader backed by `polars` lazy scanning.
///
/// `scan_parquet` reads only file metadata (schema + row counts), giving us
/// predicate/projection pushdown and streaming for free on every later query.
pub struct ParquetSource;

impl TableSource for ParquetSource {
    fn can_read(&self, path: &Path) -> bool {
        matches!(
            path.extension().and_then(|e| e.to_str()),
            Some(ext) if ext.eq_ignore_ascii_case("parquet") || ext.eq_ignore_ascii_case("pq")
        )
    }

    fn scan(&self, path: &Path) -> Result<DataTable, IoError> {
        let pl_path = PlRefPath::try_from_path(path)
            .map_err(|_| IoError::NonUtf8Path(path.display().to_string()))?;
        let lf = LazyFrame::scan_parquet(pl_path, ScanArgsParquet::default()).map_err(|e| {
            IoError::Open {
                path: path.display().to_string(),
                source: e,
            }
        })?;
        Ok(DataTable::from_lazy(table_name(path), lf)?)
    }
}
