use thiserror::Error;

/// Errors surfaced by the core data engine.
#[derive(Debug, Error)]
pub enum CoreError {
    /// The underlying polars query failed (bad file, bad expression, ...).
    #[error("data engine error: {0}")]
    Polars(#[from] polars::error::PolarsError),

    /// A column name was requested that the table does not have.
    #[error("no such column: {0}")]
    UnknownColumn(String),

    /// An operation needed data but the column has no usable values
    /// (all null / non-finite, or all <= 0 for log binning).
    #[error("column has no plottable values: {0}")]
    EmptyColumn(String),

    /// A user-typed subset predicate failed to parse or evaluate.
    #[error("bad filter expression: {0}")]
    BadExpression(String),
}
