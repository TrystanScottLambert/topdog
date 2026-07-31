//! Plot-facing data fetch: materialize exactly what a plot needs, never the
//! whole table. Scatter data is stride-downsampled past a caller-set point
//! budget; histograms are counted inside polars so only `n_bins` numbers
//! come back regardless of table size.

use polars::prelude::*;

use crate::error::CoreError;
use crate::table::DataTable;

/// Materialized scatter points (already null/NaN-filtered, cast to f64).
#[derive(Debug, Clone, PartialEq)]
pub struct ScatterData {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub x_err: Option<Vec<f64>>,
    pub y_err: Option<Vec<f64>>,
    /// Valid (plottable) rows in the full view, before downsampling.
    pub total_points: usize,
    /// Every `stride`-th valid row was kept; 1 = no downsampling.
    pub stride: usize,
}

/// How histogram bins are determined.
#[derive(Debug, Clone, PartialEq)]
pub enum BinSpec {
    /// `n` equal bins spanning the data (log10-spaced when `log`).
    Count { n: usize, log: bool },
    /// Fixed bin width; edges snap to multiples of the width so histograms
    /// of different subsets with the same width are directly comparable.
    Width(f64),
    /// Explicit, ascending bin edges (n+1 edges = n bins). The escape hatch
    /// for exact cross-figure comparisons.
    Edges(Vec<f64>),
}

/// Histogram of one column, counted lazily.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    /// `n_bins + 1` bin edges, ascending (log10-spaced for log binning).
    pub edges: Vec<f64>,
    pub counts: Vec<u64>,
    pub log_bins: bool,
}

/// N parallel columns of f64 (3D plots and friends), null/NaN rows dropped,
/// stride-downsampled like [`ScatterData`].
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnsData {
    /// Same order as the names passed to [`DataTable::columns_data`].
    pub columns: Vec<Vec<f64>>,
    pub total_points: usize,
    pub stride: usize,
}

impl DataTable {
    /// Fetch x/y (and optional error) columns for a scatter plot.
    ///
    /// Rows with a null or non-finite value in any requested column are
    /// dropped. If more than `max_points` valid rows exist, keeps every
    /// k-th row so at most `max_points` come back — a plot can't show more
    /// points than pixels anyway, and this keeps huge tables interactive.
    pub fn scatter_data(
        &self,
        x: &str,
        y: &str,
        x_err: Option<&str>,
        y_err: Option<&str>,
        max_points: usize,
    ) -> Result<ScatterData, CoreError> {
        let mut names: Vec<&str> = vec![x, y];
        names.extend(x_err);
        names.extend(y_err);

        let selected: Vec<Expr> = names
            .iter()
            .map(|n| col(*n).cast(DataType::Float64))
            .collect();
        let finite: Vec<Expr> = names.iter().map(|n| col(*n).is_finite()).collect();
        let all_finite = finite
            .into_iter()
            .reduce(|a, b| a.and(b))
            .expect("at least x and y");

        let valid = self
            .current()
            .select(selected)
            .drop_nulls(None)
            .filter(all_finite);

        let total_points = crate::table::count_rows(valid.clone())?;
        let stride = if max_points == 0 {
            1
        } else {
            total_points.div_ceil(max_points).max(1)
        };
        let sampled = if stride > 1 {
            valid.select([all().as_expr().gather_every(stride, 0)])
        } else {
            valid
        };
        let df = sampled.collect()?;

        let column_f64 = |name: &str| -> Result<Vec<f64>, CoreError> {
            Ok(df.column(name)?.f64()?.into_no_null_iter().collect())
        };

        Ok(ScatterData {
            x: column_f64(x)?,
            y: column_f64(y)?,
            x_err: x_err.map(&column_f64).transpose()?,
            y_err: y_err.map(&column_f64).transpose()?,
            total_points,
            stride,
        })
    }

    /// Fetch several columns as parallel f64 vectors (rows with any null or
    /// non-finite value dropped), stride-downsampled past `max_points`.
    pub fn columns_data(
        &self,
        names: &[&str],
        max_points: usize,
    ) -> Result<ColumnsData, CoreError> {
        assert!(!names.is_empty(), "columns_data needs at least one column");
        let selected: Vec<Expr> = names
            .iter()
            .map(|n| col(*n).cast(DataType::Float64))
            .collect();
        let all_finite = names
            .iter()
            .map(|n| col(*n).is_finite())
            .reduce(|a, b| a.and(b))
            .expect("non-empty names");

        let valid = self
            .current()
            .select(selected)
            .drop_nulls(None)
            .filter(all_finite);

        let total_points = crate::table::count_rows(valid.clone())?;
        let stride = if max_points == 0 {
            1
        } else {
            total_points.div_ceil(max_points).max(1)
        };
        let sampled = if stride > 1 {
            valid.select([all().as_expr().gather_every(stride, 0)])
        } else {
            valid
        };
        let df = sampled.collect()?;

        let columns = names
            .iter()
            .map(|n| -> Result<Vec<f64>, CoreError> {
                Ok(df.column(n)?.f64()?.into_no_null_iter().collect())
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ColumnsData {
            columns,
            total_points,
            stride,
        })
    }

    /// Scatter pairs sorted by x — the data a line plot wants. Sorting
    /// happens on the (already downsampled) materialized points, not in the
    /// lazy query, so cost is bounded by the point budget.
    pub fn line_data(&self, x: &str, y: &str, max_points: usize) -> Result<ScatterData, CoreError> {
        let mut data = self.scatter_data(x, y, None, None, max_points)?;
        let mut order: Vec<usize> = (0..data.x.len()).collect();
        order.sort_by(|&a, &b| data.x[a].total_cmp(&data.x[b]));
        data.x = order.iter().map(|&i| data.x[i]).collect();
        data.y = order.iter().map(|&i| data.y[i]).collect();
        Ok(data)
    }

    /// Histogram of `column`, binned per `bins` (see [`BinSpec`]).
    ///
    /// Bin counting happens inside polars (bin-index expression + group_by),
    /// so the scan streams and only the counts are materialized. Log binning
    /// silently drops values <= 0, matching astronomers' expectations for
    /// log-space quantities.
    pub fn histogram(&self, column: &str, bins: &BinSpec) -> Result<Histogram, CoreError> {
        let log_bins = matches!(bins, BinSpec::Count { log: true, .. });
        let value = || col(column).cast(DataType::Float64);

        let mut lf = self
            .current()
            .select([value()])
            .drop_nulls(None)
            .filter(value().is_finite());
        if log_bins {
            lf = lf.filter(value().gt(lit(0.0)));
        }

        // Pass 1: data range (streamed min/max), needed for Count/Width.
        let bounds = lf
            .clone()
            .select([
                value().min().alias("lo"),
                value().max().alias("hi"),
                len().alias("n"),
            ])
            .collect()?;
        let scalar = |name: &str| -> Option<f64> {
            bounds
                .column(name)
                .ok()?
                .as_materialized_series()
                .get(0)
                .ok()?
                .try_extract::<f64>()
                .ok()
        };
        if scalar("n").unwrap_or(0.0) == 0.0 {
            return Err(CoreError::EmptyColumn(column.to_string()));
        }
        let (Some(lo), Some(hi)) = (scalar("lo"), scalar("hi")) else {
            return Err(CoreError::EmptyColumn(column.to_string()));
        };

        // Resolve the bin plan: either uniform (in linear or log space,
        // countable by arithmetic) or explicit edges (counted by summing
        // threshold indicators — no assumptions about spacing).
        enum Plan {
            Uniform { t0: f64, width: f64, n: usize },
            Explicit,
        }
        let (edges, plan): (Vec<f64>, Plan) = match bins {
            BinSpec::Count { n, log } => {
                let n = (*n).max(1);
                let (t0, t1) = if *log {
                    (lo.log10(), hi.log10())
                } else {
                    (lo, hi)
                };
                // Degenerate range (all values equal): widen to one usable bin.
                let (t0, t1) = if t0 == t1 {
                    (t0 - 0.5, t1 + 0.5)
                } else {
                    (t0, t1)
                };
                let width = (t1 - t0) / n as f64;
                let edges = (0..=n)
                    .map(|i| {
                        let e = t0 + width * i as f64;
                        if *log {
                            10f64.powf(e)
                        } else {
                            e
                        }
                    })
                    .collect();
                (edges, Plan::Uniform { t0, width, n })
            }
            BinSpec::Width(w) => {
                if !(w.is_finite() && *w > 0.0) {
                    return Err(CoreError::BadExpression(format!("bad bin width: {w}")));
                }
                // Snap the first edge to a multiple of the width so equal
                // widths give identical grids across subsets/tables.
                let t0 = (lo / w).floor() * w;
                let n = (((hi - t0) / w).floor() as usize + 1).clamp(1, 100_000);
                let edges = (0..=n).map(|i| t0 + w * i as f64).collect();
                (edges, Plan::Uniform { t0, width: *w, n })
            }
            BinSpec::Edges(edges) => {
                if edges.len() < 2 || edges.windows(2).any(|w| w[1] <= w[0]) {
                    return Err(CoreError::BadExpression(
                        "bin edges must be at least two, strictly ascending".to_string(),
                    ));
                }
                (edges.clone(), Plan::Explicit)
            }
        };
        let n_bins = edges.len() - 1;

        // Pass 2: bin index per row, then count per bin.
        let bin_idx = match &plan {
            &Plan::Uniform { t0, width, n } => {
                let transformed = if log_bins {
                    value().log(lit(10.0))
                } else {
                    value()
                };
                ((transformed - lit(t0)) / lit(width))
                    .floor()
                    .clip(lit(0.0), lit((n - 1) as f64))
                    .cast(DataType::UInt32)
                    .alias("bin")
            }
            Plan::Explicit => {
                // index = Σ [x >= interior edge]; rows outside the outer
                // edges are dropped below.
                // Seed with a row-wise zero (not a bare literal, which
                // polars would broadcast to a single row on select).
                let mut idx = (value() - value()).cast(DataType::UInt32);
                for e in &edges[1..n_bins] {
                    idx = idx + value().gt_eq(lit(*e)).cast(DataType::UInt32);
                }
                idx.cast(DataType::UInt32).alias("bin")
            }
        };
        let mut counted_lf = lf;
        if matches!(plan, Plan::Explicit) {
            counted_lf = counted_lf.filter(
                value()
                    .gt_eq(lit(edges[0]))
                    .and(value().lt_eq(lit(edges[n_bins]))),
            );
        }
        let counted = counted_lf
            .select([bin_idx])
            .group_by([col("bin")])
            .agg([len().alias("count")])
            .collect()?;

        let mut counts = vec![0u64; n_bins];
        let bins_col = counted.column("bin")?.u32()?;
        let ns = counted.column("count")?.as_materialized_series().clone();
        for (i, bin) in bins_col.into_no_null_iter().enumerate() {
            let count = ns
                .get(i)
                .ok()
                .and_then(|v| v.try_extract::<u64>().ok())
                .unwrap_or(0);
            if let Some(slot) = counts.get_mut(bin as usize) {
                *slot = count;
            }
        }

        Ok(Histogram {
            edges,
            counts,
            log_bins,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> DataTable {
        let df = df![
            "x" => [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y" => [Some(1.0f64), Some(4.0), None, Some(16.0), Some(25.0),
                    Some(36.0), Some(49.0), Some(64.0), Some(81.0), Some(100.0)],
            "e" => [0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        ]
        .unwrap();
        DataTable::from_lazy("t", df.lazy()).unwrap()
    }

    #[test]
    fn scatter_drops_nulls_and_keeps_pairs() {
        let d = table()
            .scatter_data("x", "y", None, Some("e"), 1000)
            .unwrap();
        assert_eq!(d.total_points, 9); // one y is null
        assert_eq!(d.stride, 1);
        assert_eq!(d.x.len(), 9);
        assert_eq!(d.y.len(), 9);
        assert_eq!(d.y_err.as_ref().unwrap().len(), 9);
        assert!(!d.x.contains(&3.0)); // the null-y row is gone entirely
    }

    #[test]
    fn scatter_downsamples_past_budget() {
        let d = table().scatter_data("x", "y", None, None, 3).unwrap();
        assert_eq!(d.total_points, 9);
        assert_eq!(d.stride, 3);
        assert!(d.x.len() <= 3);
    }

    #[test]
    fn line_data_is_sorted_by_x() {
        let df = df![
            "x" => [3.0f64, 1.0, 2.0],
            "y" => [30.0f64, 10.0, 20.0],
        ]
        .unwrap();
        let t = DataTable::from_lazy("t", df.lazy()).unwrap();
        let d = t.line_data("x", "y", 1000).unwrap();
        assert_eq!(d.x, vec![1.0, 2.0, 3.0]);
        assert_eq!(d.y, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn columns_data_fetches_parallel_vectors() {
        let d = table().columns_data(&["x", "y", "e"], 1000).unwrap();
        assert_eq!(d.columns.len(), 3);
        assert_eq!(d.total_points, 9); // null y row dropped everywhere
        assert!(d.columns.iter().all(|c| c.len() == 9));
        assert_eq!(d.stride, 1);
    }

    fn count_bins(n: usize, log: bool) -> BinSpec {
        BinSpec::Count { n, log }
    }

    #[test]
    fn histogram_linear_counts_all_rows() {
        let h = table().histogram("x", &count_bins(5, false)).unwrap();
        assert_eq!(h.counts.len(), 5);
        assert_eq!(h.edges.len(), 6);
        assert_eq!(h.counts.iter().sum::<u64>(), 10);
        assert!((h.edges[0] - 1.0).abs() < 1e-12);
        assert!((h.edges[5] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn histogram_log_bins_are_log_spaced() {
        let h = table().histogram("x", &count_bins(4, true)).unwrap();
        assert_eq!(h.counts.iter().sum::<u64>(), 10);
        // Log-spaced edges: constant ratio, not constant difference.
        let r0 = h.edges[1] / h.edges[0];
        let r1 = h.edges[2] / h.edges[1];
        assert!((r0 - r1).abs() < 1e-9);
    }

    #[test]
    fn histogram_of_constant_column_widens() {
        let df = df!["c" => [5.0f64, 5.0, 5.0]].unwrap();
        let t = DataTable::from_lazy("t", df.lazy()).unwrap();
        let h = t.histogram("c", &count_bins(3, false)).unwrap();
        assert_eq!(h.counts.iter().sum::<u64>(), 3);
    }

    #[test]
    fn histogram_empty_column_errors() {
        let df = df!["c" => [Option::<f64>::None, None]].unwrap();
        let t = DataTable::from_lazy("t", df.lazy()).unwrap();
        assert!(t.histogram("c", &count_bins(3, false)).is_err());
    }

    #[test]
    fn histogram_fixed_width_snaps_to_grid() {
        // Data 1..=10 with width 2.5: first edge floor(1/2.5)*2.5 = 0.
        let h = table().histogram("x", &BinSpec::Width(2.5)).unwrap();
        assert!((h.edges[0] - 0.0).abs() < 1e-12);
        assert!(h.edges.windows(2).all(|w| (w[1] - w[0] - 2.5).abs() < 1e-9));
        assert_eq!(h.counts.iter().sum::<u64>(), 10);
        // The last edge covers the max value.
        assert!(*h.edges.last().unwrap() >= 10.0);

        assert!(table().histogram("x", &BinSpec::Width(0.0)).is_err());
        assert!(table().histogram("x", &BinSpec::Width(-1.0)).is_err());
    }

    #[test]
    fn histogram_explicit_edges() {
        // Edges chosen so bins are uneven: [1,2), [2,5), [5,10].
        let edges = vec![1.0, 2.0, 5.0, 10.0];
        let h = table()
            .histogram("x", &BinSpec::Edges(edges.clone()))
            .unwrap();
        assert_eq!(h.edges, edges);
        assert_eq!(h.counts, vec![1, 3, 6]); // x=1 | x=2,3,4 | x=5..=10
                                             // Values outside the edges are dropped, not clamped.
        let h = table()
            .histogram("x", &BinSpec::Edges(vec![3.0, 5.0]))
            .unwrap();
        assert_eq!(h.counts, vec![3]); // 3, 4, 5

        assert!(table().histogram("x", &BinSpec::Edges(vec![1.0])).is_err());
        assert!(table()
            .histogram("x", &BinSpec::Edges(vec![1.0, 1.0, 2.0]))
            .is_err());
    }
}
