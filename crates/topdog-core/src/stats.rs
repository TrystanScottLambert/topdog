use polars::prelude::*;

use crate::column::ColumnMeta;
use crate::error::CoreError;

/// Summary statistics for one column.
///
/// Numeric aggregates are `None` for non-numeric columns; min/max are kept as
/// display strings so they work for strings and timestamps too.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnStats {
    pub column: String,
    pub min: Option<String>,
    pub max: Option<String>,
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub median: Option<f64>,
    pub null_count: u64,
    pub n_unique: u64,
}

impl ColumnStats {
    /// Compute stats with a single lazy aggregation pass.
    ///
    /// Everything is expressed as polars aggregation expressions (not Rust
    /// loops over collected data) so polars can stream the scan and only the
    /// one-row result is ever materialized.
    pub(crate) fn compute(lf: LazyFrame, meta: &ColumnMeta) -> Result<ColumnStats, CoreError> {
        let c = || col(meta.name.as_str());
        let mut exprs = vec![
            c().min().alias("min"),
            c().max().alias("max"),
            c().null_count().alias("null_count"),
            c().n_unique().alias("n_unique"),
        ];
        if meta.is_numeric() {
            exprs.extend([
                c().mean().cast(DataType::Float64).alias("mean"),
                c().std(1).cast(DataType::Float64).alias("std"),
                c().median().cast(DataType::Float64).alias("median"),
            ]);
        }
        let df = lf.select(exprs).collect()?;

        let get = |name: &str| -> Result<AnyValue, CoreError> {
            Ok(df.column(name)?.as_materialized_series().get(0)?)
        };
        let display =
            |v: AnyValue| -> Option<String> { (!v.is_null()).then(|| v.str_value().into_owned()) };
        let float = |name: &str| -> Result<Option<f64>, CoreError> {
            if meta.is_numeric() {
                Ok(get(name)?.try_extract::<f64>().ok())
            } else {
                Ok(None)
            }
        };

        Ok(ColumnStats {
            column: meta.name.clone(),
            min: display(get("min")?),
            max: display(get("max")?),
            mean: float("mean")?,
            std: float("std")?,
            median: float("median")?,
            null_count: get("null_count")?.try_extract::<u64>().unwrap_or(0),
            n_unique: get("n_unique")?.try_extract::<u64>().unwrap_or(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::DataTable;

    #[test]
    fn numeric_stats() {
        let df = df![
            "x" => [Some(1.0f64), Some(2.0), Some(3.0), Some(4.0), None],
        ]
        .unwrap();
        let t = DataTable::from_lazy("t", df.lazy()).unwrap();
        let s = t.stats("x").unwrap();
        assert_eq!(s.min.as_deref(), Some("1.0"));
        assert_eq!(s.max.as_deref(), Some("4.0"));
        assert_eq!(s.mean, Some(2.5));
        assert_eq!(s.median, Some(2.5));
        assert_eq!(s.null_count, 1);
        assert_eq!(s.n_unique, 5); // null counts as a distinct value in polars
        let std = s.std.unwrap();
        assert!((std - 1.2909944487358056).abs() < 1e-12);
    }

    #[test]
    fn string_stats_skip_numeric_aggregates() {
        let df = df!["s" => ["b", "a", "c", "a"]].unwrap();
        let t = DataTable::from_lazy("t", df.lazy()).unwrap();
        let s = t.stats("s").unwrap();
        assert_eq!(s.min.as_deref(), Some("a"));
        assert_eq!(s.max.as_deref(), Some("c"));
        assert_eq!(s.mean, None);
        assert_eq!(s.n_unique, 3);
        assert_eq!(s.null_count, 0);
    }

    #[test]
    fn unknown_column_errors() {
        let df = df!["x" => [1i32]].unwrap();
        let t = DataTable::from_lazy("t", df.lazy()).unwrap();
        assert!(t.stats("y").is_err());
    }
}
