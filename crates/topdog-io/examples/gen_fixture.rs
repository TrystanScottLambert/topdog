//! Regenerates `tests/fixtures/small_catalog.parquet` (workspace root).
//!
//! Run from the workspace root with:
//! `cargo run -p topdog-io --example gen_fixture`
//!
//! The fixture is a ~300-row fake galaxy catalog: monotonically increasing
//! id, ra/dec covering the sky, a magnitude column with some nulls, and a
//! string name column — enough dtype variety to exercise the reader, stats,
//! and table view.

use polars::prelude::*;

fn main() -> PolarsResult<()> {
    let n = 300u32;
    let ids: Vec<u32> = (0..n).collect();
    // Deterministic pseudo-random values so the fixture is reproducible.
    let ra: Vec<f64> = (0..n).map(|i| (i as f64 * 47.9) % 360.0).collect();
    let dec: Vec<f64> = (0..n).map(|i| ((i as f64 * 13.7) % 180.0) - 90.0).collect();
    let mag: Vec<Option<f64>> = (0..n)
        .map(|i| (i % 23 != 0).then(|| 14.0 + ((i as f64 * 7.3) % 12.0)))
        .collect();
    let name: Vec<String> = (0..n).map(|i| format!("GAL-{i:04}")).collect();

    let mut df = df![
        "id" => ids,
        "ra" => ra,
        "dec" => dec,
        "mag" => mag,
        "name" => name,
    ]?;

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/small_catalog.parquet"
    );
    let file = std::fs::File::create(path).expect("create fixture file");
    ParquetWriter::new(file).finish(&mut df)?;
    println!("wrote {path} ({} rows)", df.height());
    Ok(())
}
