use polars::prelude::DataType;

/// Per-column metadata carried alongside the lazy frame.
///
/// polars schemas only know name + dtype; astronomy tables also care about
/// physical quantity and unit (CLAUDE.md "beyond parity"). Those live here so
/// they survive operations that rebuild the lazy plan, and so a future FITS
/// reader can populate them from TUNIT/TTYPE headers.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnMeta {
    /// Column name as it appears in the underlying file.
    pub name: String,
    /// polars data type of the column.
    pub dtype: DataType,
    /// Physical unit (e.g. "mag", "deg", "km/s"), if known.
    pub unit: Option<String>,
    /// Physical quantity tag (UCD-like, e.g. "pos.eq.ra"), if known.
    pub quantity: Option<String>,
}

impl ColumnMeta {
    pub fn new(name: impl Into<String>, dtype: DataType) -> Self {
        Self {
            name: name.into(),
            dtype,
            unit: None,
            quantity: None,
        }
    }

    /// Whether stats like mean/stddev make sense for this column.
    pub fn is_numeric(&self) -> bool {
        self.dtype.is_primitive_numeric()
    }
}
