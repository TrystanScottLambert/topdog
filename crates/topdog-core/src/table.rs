use polars::prelude::*;

use crate::column::ColumnMeta;
use crate::error::CoreError;
use crate::stats::ColumnStats;

/// Sort state applied on top of the base scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortSpec {
    pub column: String,
    pub descending: bool,
}

/// A materialized window of rows, formatted for display.
///
/// This is what the virtualized table view consumes: only the rows currently
/// on (or near) screen, already rendered to strings so the GUI layer never
/// touches polars types.
#[derive(Debug, Clone)]
pub struct RowWindow {
    /// Row index (within the current sorted/filtered view) of the first row.
    pub offset: usize,
    /// `rows[i][j]` is the display string for row `offset + i`, column `j`.
    pub rows: Vec<Vec<String>>,
    /// Stable base-table row index of each displayed row (survives sorting
    /// and filtering) — the currency of row-selection subsets.
    pub indices: Vec<u64>,
}

/// Hidden column name used to thread stable row indices through queries.
const ROW_INDEX_COL: &str = "__topdog_row__";

/// A loaded table: a lazy polars scan plus column metadata and view state.
///
/// The `LazyFrame` is the single source of truth for the data; `DataTable`
/// never holds a fully collected frame. Every accessor builds a lazy query
/// (scan + filter + sort + slice/aggregation) and collects only its result,
/// so RAM usage scales with what is on screen, not with file size.
///
/// Cloning is cheap (lazy plans are refcounted), which lets the GUI hand
/// clones to background tasks for async window fetches.
#[derive(Clone)]
pub struct DataTable {
    name: String,
    base: LazyFrame,
    columns: Vec<ColumnMeta>,
    sort: Option<SortSpec>,
    filter: Option<Expr>,
    /// Total rows in the base scan (cheap for parquet: read from metadata).
    total_rows: usize,
    /// Rows remaining after `filter`; equals `total_rows` when unfiltered.
    visible_rows: usize,
}

// Manual impl because `LazyFrame` (a query plan) has no `Debug`.
impl std::fmt::Debug for DataTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataTable")
            .field("name", &self.name)
            .field("columns", &self.columns.len())
            .field("rows", &self.visible_rows)
            .field("sort", &self.sort)
            .finish_non_exhaustive()
    }
}

impl DataTable {
    /// Wrap an already-constructed lazy scan.
    ///
    /// Resolves the schema and row count up front — both are metadata-level
    /// operations for parquet and do not load column data.
    pub fn from_lazy(name: impl Into<String>, lf: LazyFrame) -> Result<Self, CoreError> {
        let mut lf = lf;
        let schema = lf.collect_schema()?;
        let columns = schema
            .iter()
            .map(|(name, dtype)| ColumnMeta::new(name.as_str(), dtype.clone()))
            .collect();
        let total_rows = count_rows(lf.clone())?;
        Ok(Self {
            name: name.into(),
            base: lf,
            columns,
            sort: None,
            filter: None,
            total_rows,
            visible_rows: total_rows,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn columns(&self) -> &[ColumnMeta] {
        &self.columns
    }

    pub fn n_columns(&self) -> usize {
        self.columns.len()
    }

    /// Rows in the current view (after filtering).
    pub fn n_rows(&self) -> usize {
        self.visible_rows
    }

    /// Rows in the underlying file, ignoring any filter.
    pub fn n_total_rows(&self) -> usize {
        self.total_rows
    }

    pub fn sort_spec(&self) -> Option<&SortSpec> {
        self.sort.as_ref()
    }

    /// Cycle a column's sort: none -> ascending -> descending -> none.
    /// Sorting a different column starts back at ascending.
    pub fn toggle_sort(&mut self, column: &str) -> Result<(), CoreError> {
        self.check_column(column)?;
        self.sort = match self.sort.take() {
            Some(s) if s.column == column && !s.descending => Some(SortSpec {
                column: column.to_string(),
                descending: true,
            }),
            Some(s) if s.column == column => None,
            _ => Some(SortSpec {
                column: column.to_string(),
                descending: false,
            }),
        };
        Ok(())
    }

    /// Apply a row filter predicate (replacing any existing one) and
    /// recompute the visible row count. `None` clears the filter.
    pub fn set_filter(&mut self, predicate: Option<Expr>) -> Result<(), CoreError> {
        self.filter = predicate;
        self.visible_rows = match &self.filter {
            Some(expr) => count_rows(self.base.clone().filter(expr.clone()))?,
            None => self.total_rows,
        };
        Ok(())
    }

    /// A named subsample: rows filtered by a SQL-style predicate (e.g.
    /// `mag < 20 AND dec > 0`), optionally projected to a chosen column set.
    ///
    /// Either part may be omitted (`None` predicate = all rows, `None`
    /// columns = all columns). The result is a standalone `DataTable`
    /// sharing the underlying lazy scan — any number of live subsets cost
    /// no data copies. Errors on an unparseable predicate, or unknown
    /// columns (surfaced when the row count is computed).
    pub fn subset(
        &self,
        name: impl Into<String>,
        predicate: Option<&str>,
        columns: Option<&[String]>,
    ) -> Result<Self, CoreError> {
        let mut lf = self.base.clone();
        if let Some(pred) = predicate {
            let expr =
                polars::sql::sql_expr(pred).map_err(|e| CoreError::BadExpression(e.to_string()))?;
            lf = lf.filter(expr);
        }
        if let Some(cols) = columns {
            self.check_columns(cols)?;
            lf = lf.select(cols.iter().map(|c| col(c.as_str())).collect::<Vec<_>>());
        }
        Self::from_lazy(name, lf).map_err(|e| CoreError::BadExpression(e.to_string()))
    }

    /// A subsample of explicitly chosen rows (indices into THIS table's
    /// current row order, as reported by [`Self::fetch_window`]), optionally
    /// projected to chosen columns. This is how "select rows in the table
    /// view, make a subset" works.
    pub fn subset_from_rows(
        &self,
        name: impl Into<String>,
        rows: &[u64],
        columns: Option<&[String]>,
    ) -> Result<Self, CoreError> {
        if let Some(cols) = columns {
            self.check_columns(cols)?;
        }
        let picked = Series::new(
            ROW_INDEX_COL.into(),
            rows.iter().map(|&v| v as IdxSize).collect::<Vec<_>>(),
        );
        let mut lf = self
            .indexed_base()
            .filter(col(ROW_INDEX_COL).is_in(lit(picked).implode(true), false))
            .drop(by_name([ROW_INDEX_COL], true, false));
        if let Some(cols) = columns {
            lf = lf.select(cols.iter().map(|c| col(c.as_str())).collect::<Vec<_>>());
        }
        Self::from_lazy(name, lf).map_err(|e| CoreError::BadExpression(e.to_string()))
    }

    /// Base scan with the stable row-index column prepended (before any
    /// filter/sort, so indices identify rows regardless of view state).
    fn indexed_base(&self) -> LazyFrame {
        self.base.clone().with_row_index(ROW_INDEX_COL, None)
    }

    /// The lazy query for the current view: base scan + filter + sort.
    ///
    /// Callers (window fetch, stats, future plot pipelines) compose on top of
    /// this so they all see the same view of the data.
    pub fn current(&self) -> LazyFrame {
        let mut lf = self.base.clone();
        if let Some(expr) = &self.filter {
            lf = lf.filter(expr.clone());
        }
        if let Some(sort) = &self.sort {
            lf = lf.sort(
                [sort.column.as_str()],
                SortMultipleOptions::default()
                    .with_order_descending(sort.descending)
                    .with_nulls_last(true),
            );
        }
        lf
    }

    /// Materialize `len` rows starting at `offset` in the current view.
    ///
    /// This is the only row-data path the table view uses; it collects just
    /// the slice, so cost is proportional to the window, not the file
    /// (modulo sort/filter work polars must do to know what lands in it).
    pub fn fetch_window(&self, offset: usize, len: usize) -> Result<RowWindow, CoreError> {
        // Same pipeline as `current()`, but over the row-indexed base so
        // every displayed row knows its stable identity.
        let mut lf = self.indexed_base();
        if let Some(expr) = &self.filter {
            lf = lf.filter(expr.clone());
        }
        if let Some(sort) = &self.sort {
            lf = lf.sort(
                [sort.column.as_str()],
                SortMultipleOptions::default()
                    .with_order_descending(sort.descending)
                    .with_nulls_last(true),
            );
        }
        let df = lf.slice(offset as i64, len as IdxSize).collect()?;
        let n = df.height();
        let indices = df
            .column(ROW_INDEX_COL)?
            .as_materialized_series()
            .u32()?
            .into_no_null_iter()
            .map(u64::from)
            .collect();
        let series: Vec<_> = df
            .columns()
            .iter()
            .filter(|c| c.name() != ROW_INDEX_COL)
            .map(|c| c.as_materialized_series())
            .collect();
        let rows = (0..n)
            .map(|i| {
                series
                    .iter()
                    .map(|s| format_value(s.get(i).expect("index within collected slice")))
                    .collect()
            })
            .collect();
        Ok(RowWindow {
            offset,
            rows,
            indices,
        })
    }

    /// Streamed summary statistics for one column, via lazy aggregation.
    pub fn stats(&self, column: &str) -> Result<ColumnStats, CoreError> {
        let meta = self
            .columns
            .iter()
            .find(|c| c.name == column)
            .ok_or_else(|| CoreError::UnknownColumn(column.to_string()))?;
        ColumnStats::compute(self.current(), meta)
    }

    fn check_column(&self, column: &str) -> Result<(), CoreError> {
        if self.columns.iter().any(|c| c.name == column) {
            Ok(())
        } else {
            Err(CoreError::UnknownColumn(column.to_string()))
        }
    }

    fn check_columns(&self, columns: &[String]) -> Result<(), CoreError> {
        for c in columns {
            self.check_column(c)?;
        }
        Ok(())
    }
}

/// Count rows of a lazy query without materializing any column data.
pub(crate) fn count_rows(lf: LazyFrame) -> Result<usize, CoreError> {
    let df = lf.select([len()]).collect()?;
    let n = df
        .columns()
        .first()
        .and_then(|c| c.as_materialized_series().get(0).ok())
        .and_then(|v| v.try_extract::<u64>().ok())
        .unwrap_or(0);
    Ok(n as usize)
}

/// Render a single cell value for display.
///
/// Centralized so the table view, and later exports/tooltips, all format
/// values identically.
fn format_value(value: AnyValue) -> String {
    match value {
        AnyValue::Null => String::new(),
        AnyValue::Float64(f) => format_float(f),
        AnyValue::Float32(f) => format_float(f as f64),
        other => other.str_value().into_owned(),
    }
}

/// Floats get enough digits to round-trip visually without the noise of
/// full 17-digit output.
fn format_float(f: f64) -> String {
    if f == 0.0 || (1e-4..1e12).contains(&f.abs()) {
        let s = format!("{f:.6}");
        let s = s.trim_end_matches('0').trim_end_matches('.');
        s.to_string()
    } else {
        format!("{f:.6e}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_table() -> DataTable {
        let df = df![
            "id" => [3i64, 1, 2, 5, 4],
            "mag" => [Some(20.5f64), Some(18.1), None, Some(19.9), Some(21.0)],
            "name" => ["c", "a", "b", "e", "d"],
        ]
        .unwrap();
        DataTable::from_lazy("sample", df.lazy()).unwrap()
    }

    #[test]
    fn schema_and_counts_resolved_on_load() {
        let t = sample_table();
        assert_eq!(t.n_rows(), 5);
        assert_eq!(t.n_total_rows(), 5);
        let names: Vec<_> = t.columns().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["id", "mag", "name"]);
        assert!(t.columns()[0].is_numeric());
        assert!(!t.columns()[2].is_numeric());
    }

    #[test]
    fn fetch_window_slices_rows() {
        let t = sample_table();
        let w = t.fetch_window(1, 2).unwrap();
        assert_eq!(w.offset, 1);
        assert_eq!(w.rows.len(), 2);
        assert_eq!(w.rows[0], ["1", "18.1", "a"]);
        assert_eq!(w.rows[1], ["2", "", "b"]);
    }

    #[test]
    fn window_past_end_is_truncated() {
        let t = sample_table();
        let w = t.fetch_window(4, 100).unwrap();
        assert_eq!(w.rows.len(), 1);
    }

    #[test]
    fn toggle_sort_cycles_and_applies() {
        let mut t = sample_table();
        t.toggle_sort("id").unwrap();
        let w = t.fetch_window(0, 5).unwrap();
        assert_eq!(w.rows[0][0], "1");
        t.toggle_sort("id").unwrap();
        let w = t.fetch_window(0, 5).unwrap();
        assert_eq!(w.rows[0][0], "5");
        t.toggle_sort("id").unwrap();
        assert!(t.sort_spec().is_none());
        assert!(t.toggle_sort("nope").is_err());
    }

    #[test]
    fn nulls_sort_last() {
        let mut t = sample_table();
        t.toggle_sort("mag").unwrap();
        let w = t.fetch_window(0, 5).unwrap();
        assert_eq!(w.rows[4][1], "");
    }

    #[test]
    fn subset_from_sql_expression() {
        let t = sample_table();
        let bright = t.subset("bright", Some("mag < 20.0"), None).unwrap();
        assert_eq!(bright.name(), "bright");
        assert_eq!(bright.n_rows(), 2); // 18.1 and 19.9; null mag excluded
                                        // Base table is untouched.
        assert_eq!(t.n_rows(), 5);

        // id > 3 leaves ids {4, 5} with mags {21.0, 19.9}; only 19.9 < 21.0.
        let combo = t.subset("c", Some("mag < 21.0 AND id > 3"), None).unwrap();
        assert_eq!(combo.n_rows(), 1);

        assert!(t.subset("bad", Some("not a filter (("), None).is_err());
        assert!(t.subset("bad", Some("nonexistent_col > 1"), None).is_err());
    }

    #[test]
    fn subset_can_project_columns() {
        let t = sample_table();
        let cols = vec!["id".to_string(), "name".to_string()];
        let s = t
            .subset("slim", Some("id > 2"), Some(cols.as_slice()))
            .unwrap();
        assert_eq!(s.n_rows(), 3);
        let names: Vec<_> = s.columns().iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["id", "name"]);
        // Columns-only subset (no predicate).
        let s = t.subset("slim2", None, Some(cols.as_slice())).unwrap();
        assert_eq!(s.n_rows(), 5);
        assert_eq!(s.n_columns(), 2);
        // Unknown column errors.
        let bad = vec!["nope".to_string()];
        assert!(t.subset("bad", None, Some(bad.as_slice())).is_err());
    }

    #[test]
    fn window_reports_stable_row_indices() {
        let mut t = sample_table();
        let w = t.fetch_window(0, 5).unwrap();
        assert_eq!(w.indices, vec![0, 1, 2, 3, 4]);
        // Sorting permutes the display order but indices track the rows.
        t.toggle_sort("id").unwrap(); // ascending by id: 1,2,3,4,5
        let w = t.fetch_window(0, 5).unwrap();
        assert_eq!(w.rows[0][0], "1");
        assert_eq!(w.indices[0], 1); // id=1 lives at base row 1
        assert_eq!(w.indices[2], 0); // id=3 lives at base row 0
    }

    #[test]
    fn subset_from_selected_rows() {
        let t = sample_table();
        let s = t.subset_from_rows("picked", &[0, 3], None).unwrap();
        assert_eq!(s.n_rows(), 2);
        let w = s.fetch_window(0, 10).unwrap();
        assert_eq!(w.rows[0][0], "3"); // base row 0 has id=3
        assert_eq!(w.rows[1][0], "5"); // base row 3 has id=5

        let cols = vec!["mag".to_string()];
        let s = t
            .subset_from_rows("picked2", &[1], Some(cols.as_slice()))
            .unwrap();
        assert_eq!(s.n_columns(), 1);
        assert_eq!(s.n_rows(), 1);
    }

    #[test]
    fn filter_updates_visible_rows() {
        let mut t = sample_table();
        t.set_filter(Some(col("id").gt(lit(2)))).unwrap();
        assert_eq!(t.n_rows(), 3);
        assert_eq!(t.n_total_rows(), 5);
        let w = t.fetch_window(0, 10).unwrap();
        assert_eq!(w.rows.len(), 3);
        t.set_filter(None).unwrap();
        assert_eq!(t.n_rows(), 5);
    }

    #[test]
    fn float_formatting() {
        assert_eq!(format_float(1.5), "1.5");
        assert_eq!(format_float(0.0), "0");
        assert_eq!(format_float(-2.25), "-2.25");
        assert_eq!(format_float(1.0e-7), "1.000000e-7");
    }
}
