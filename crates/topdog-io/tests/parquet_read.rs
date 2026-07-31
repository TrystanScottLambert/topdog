//! Integration tests for the parquet reader against the checked-in fixture.
//! Regenerate the fixture with `cargo run -p topdog-io --example gen_fixture`.

use std::path::Path;

use topdog_io::{open_table, ParquetSource, TableSource};

fn fixture() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/small_catalog.parquet"
    ))
}

#[test]
fn can_read_dispatches_on_extension() {
    let src = ParquetSource;
    assert!(src.can_read(Path::new("x.parquet")));
    assert!(src.can_read(Path::new("x.PARQUET")));
    assert!(src.can_read(Path::new("x.pq")));
    assert!(!src.can_read(Path::new("x.fits")));
    assert!(!src.can_read(Path::new("parquet")));
}

#[test]
fn unsupported_extension_is_rejected() {
    assert!(open_table(Path::new("table.csv")).is_err());
}

#[test]
fn opens_fixture_with_schema_and_counts() {
    let t = open_table(fixture()).unwrap();
    assert_eq!(t.name(), "small_catalog");
    assert_eq!(t.n_rows(), 300);
    let names: Vec<_> = t.columns().iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["id", "ra", "dec", "mag", "name"]);
}

#[test]
fn window_and_stats_work_end_to_end() {
    let t = open_table(fixture()).unwrap();
    let w = t.fetch_window(0, 3).unwrap();
    assert_eq!(w.rows.len(), 3);
    assert_eq!(w.rows[0][0], "0");
    assert_eq!(w.rows[0][4], "GAL-0000");

    let s = t.stats("mag").unwrap();
    assert!(s.null_count > 0);
    assert!(s.mean.is_some());

    let ra = t.stats("ra").unwrap();
    assert!(ra.null_count == 0);
}
