# topdog — CLAUDE.md

This file is the standing brief for Claude Code on this repository. Read it in full before writing any code, and re-read the relevant section before starting each new piece of work. Keep this file up to date as decisions are made — if you make an architectural choice not covered here, add it.

## 1. What we're building

`topdog` is a from-scratch Rust rewrite of [TOPCAT](https://www.star.bris.ac.uk/~mbt/topcat/) (source: https://github.com/Starlink/starjava/, under `topcat/`, `table/`, and `ttools/` — the rest of the starjava monorepo is irrelevant and should not be studied). TOPCAT is a Java desktop tool for interactively exploring, plotting, and manipulating tabular astronomical data. We are keeping its good ideas (fast interactive plotting, rich per-column stats, an intuitive table browser) and deliberately dropping everything related to the Virtual Observatory.

Read the TOPCAT docs for *feature and interaction* inspiration only — not for architecture. We are not porting Java code or its object model. Every subsystem gets redesigned around Rust idioms, Arrow/Parquet, and an Elm-architecture GUI.

### Philosophy (in priority order — when requirements conflict, higher wins)

1. **Parquet (and later FITS) are first-class citizens.** Reading a file and getting a usable, interactive dataframe should take seconds, not require configuration.
2. **Minimal RAM footprint via lazy evaluation.** Never materialize a full column/table in memory unless the user's operation genuinely requires it (e.g. a full sort or a plot that needs every point). Prefer streaming/chunked scans, predicate/projection pushdown, and windowed reads. A 50 GB parquet file should open in under a second and be immediately explorable.
3. **Paper-ready plots.** Default plot styling should look like something you'd put in a journal figure without touching it further: clean typography, sensible margins, vector export (SVG/PDF/PNG).
4. **Modern, responsive GUI.** No modal dialog spam, no Swing-era look. Native, GPU-accelerated, resizes gracefully.
5. **Everything about a plot's appearance is directly and discoverably editable.** If it's drawn on screen, it's a candidate for the user to click and change: axis labels, legend text, tick label size/rotation/format, marker size/shape/color, line width, plot title, axis limits, gridlines. Clicking a label opens inline editing (a floating text input positioned over the label), not a separate "preferences" panel buried three menus deep. A properties side panel (for the less "click directly on it" attributes: colors, scales, error bar style, binning) complements — but never replaces — direct manipulation.
6. **Faster than a matplotlib script.** The whole point of existing is that opening topdog, loading a parquet file, and getting a publication-candidate scatter plot must be quicker than writing `import matplotlib...` boilerplate. Every added click or dialog is a cost against this goal — justify it.
7. **Scripting, if we ever add it, is Python, not STILTS/Jython.** Not in scope for v1; don't build for it prematurely, but don't paint us into a corner (e.g. keep table operations as composable functions with clear names, not GUI-only logic) either.

### In scope (feature parity + beyond)

- Parquet reading (primary format).
- FITS table reading (binary tables; images out of scope unless later requested) — phase 2, see roadmap.
- Column statistics (min/max/mean/stddev/nulls/cardinality/percentiles) computed lazily/streamed.
- Row and column selection, filtering (predicate expressions), sorting, visibility toggling.
- Scatter plots with error bars (x and/or y).
- Histograms (linear/log binning).
- 3D plots (rotate/pan/zoom).
- Sky plots with multiple projections (Aitoff, Mollweide, orthographic — start with Aitoff + Mollweide).
- Line plots, done more intuitively than TOPCAT's (sensible defaults for connecting sorted vs unsorted data, multi-series).
- Model fitting overlaid on plots (start with linear + polynomial + a generic least-squares fit against a user-supplied expression).
- Per-column metadata: physical quantity/UCD-like tag and unit, editable in a header/metadata panel.
- Dynamic, click-to-edit legends and axis labels/titles/ticks (see philosophy §5).
- Dynamic editing of axis ranges and the plot bounding box/frame style.
- Table joins (at minimum: inner/left join on key column(s) between two loaded tables) — nice-to-have, not v1-blocking.
- Export: figures to SVG/PNG/PDF; tables to Parquet/CSV.

### Explicitly out of scope — do not build, do not leave stubs suggesting future VO work

- Everything Virtual Observatory: TAP/SIA/SSA/SCS clients, VOTable I/O, cone search, VO registries, datalink, SAMP.
- Cross-matching (positional or otherwise) between catalogues.
- STILTS / any bespoke scripting language.
- Any dependency on the `starjava` Java code or JVM.

If you find yourself importing a VO-related concept "just in case," stop — it's not needed.

## 2. Tech stack

- **Language:** Rust, edition 2021, stable toolchain only (no nightly features).
- **GUI:** [`iced`](https://iced.rs/) (Elm-architecture). Use `iced::widget::canvas` for all plot rendering — plots are custom-drawn geometry, not a wrapped charting library, because we need pixel-level control for click-to-edit label hit-testing. Use `iced_aw` only if a specific widget (e.g. color picker, tabs) is missing from core `iced` — don't take on the dependency speculatively.
- **Dataframes / lazy engine:** [`polars`](https://pola.rs/) with the `lazy` and `parquet` features. `polars::LazyFrame::scan_parquet` gives us predicate/projection pushdown and streaming out of the box — build the internal `topdog-core` dataframe abstraction as a thin wrapper around `LazyFrame`, not a reimplementation. Only `.collect()` the slice of rows/columns actually needed for the current view or plot (viewport-based paging for the table view; downsampled/binned aggregates for plots over huge tables — see §5).
- **FITS reading (phase 2):** start with the `fitsio` crate (mature cfitsio binding) for correctness; if the C dependency becomes a packaging problem, evaluate pure-Rust alternatives (`astrors`, `fits-io`) but don't block phase 1 on this decision.
- **Plot math/geometry:** hand-rolled (linear/log scale transforms, tick generation, Aitoff/Mollweide projections) in `topdog-core` or a `topdog-plot` crate — small and dependency-light, since correctness and control over tick placement matters more than reuse here.
- **Fitting:** start with closed-form linear/polynomial least squares (hand-rolled or via `nalgebra`); consider `argmin` only if/when nonlinear fitting is actually requested.
- **Python plotting escape hatch:** not built until a concrete need appears. If added later, it's invoked as an optional subprocess (e.g. write data + a matplotlib script, shell out), never a hard dependency — most users should never need it because native plots already look publication-ready.
- **Error handling:** `thiserror` for library-crate error enums, `anyhow` only at binary/application boundaries.
- **Testing:** `cargo test` with unit tests per module; integration tests for the parquet/FITS readers against small fixture files checked into `tests/fixtures/`.
- **Linting:** code must pass `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` before being considered done.

## 3. Workspace layout

Use a Cargo workspace. Suggested crate split (adjust as needed, but keep the GUI crate free of file-format-specific code and keep the core data engine free of GUI code — this boundary matters for testability):

```
topdog/
  Cargo.toml                 # workspace
  crates/
    topdog-core/              # DataTable abstraction over polars LazyFrame, column metadata (unit/UCD-like tags), selection/filter model, stats engine
    topdog-io/                 # parquet + (phase 2) fits readers/writers, all behind a common `TableSource` trait
    topdog-plot/                # scale/tick/projection math, plot "spec" data model (what's drawn), fitting routines
    topdog-gui/                # iced application: table browser, plot canvas + editable overlays, panels, app state/messages
    topdog-cli/ (optional)      # thin binary for headless stats/export, useful for testing topdog-core without the GUI
  tests/
    fixtures/                  # small .parquet (and later .fits) sample files
```

The `topdog-gui` binary is the product; `topdog-cli` is a convenience for testing the engine without spinning up a window, and is not a priority.

## 4. Core data model notes

- A loaded table is a `DataTable` in `topdog-core`: a `polars::LazyFrame` + per-column metadata (display name, physical quantity, unit, format) + the current row selection/filter expression + a materialized-row cache for whatever window the table view is currently showing.
- Table view materializes only the visible row range (virtualized/paginated rendering) plus whatever's needed for on-screen stats.
- Plots do not iterate every row through iced's canvas draw call for huge tables. For scatter/line plots beyond some threshold (make this configurable, default something like 200k points), use `polars` to compute a screen-space-aware downsample or density aggregate (e.g. hexbin/2D histogram) lazily, so the "large dataset" story from the philosophy actually holds at plot time, not just at load time.
- Column stats (min/max/mean/etc.) are computed via `LazyFrame` aggregation expressions, not manual Rust loops over collected `Vec`s, so they benefit from polars' streaming execution.

## 5. GUI / plotting interaction notes

- Plot rendering: an `iced::widget::canvas::Program` that draws frame, axes, ticks, gridlines, data, legend, title from a `PlotSpec` struct (owned by app state). `PlotSpec` is the single source of truth for a plot's appearance — every editable property (label text, font size, colors, marker style, axis limits, tick count/format) lives there.
- Click-to-edit: hit-test canvas mouse events against the last-drawn bounding boxes of text elements (axis title, axis labels, tick labels, legend entries, plot title). On click, overlay an `iced::widget::text_input` positioned at that element's screen rect, pre-filled with the current value; on submit, update `PlotSpec` and re-render. This is the mechanism behind "clickable and editable labels" — build it once as a reusable pattern, not per-label special cases.
- A properties panel (dockable side panel) handles attributes that don't make sense to "click on directly" — marker shape/size, error bar style, binning parameters, color scales, projection choice for sky plots.
- Changing any `PlotSpec` field triggers a redraw only (never a full data re-scan) unless the change affects what data is needed (e.g. new binning resolution, new column selected for an axis).
- Export renders the same `PlotSpec` + data through an SVG/PDF backend so exported figures match on-screen exactly — don't build a second rendering path.

## 5a. Decisions made during implementation (keep current)

- **Tabbed UI.** The main window is a tab strip: `Tables | Scatter | Line | Histogram | Sky | 3D`. Each plot tab exposes only the inputs that plot type needs (histogram: one column + binning; sky: lon/lat + projection; 3D: x/y/z; etc.). The table browser lives in the Tables tab, not permanently on screen.
- **Multiple tables.** Any number of parquet files can be open at once (`TopDog::tables`). Opening a file appends; nothing is a singleton.
- **Subsets (subsamples).** A subset is a named filter over a base table, typed as a SQL predicate (`mag < 20 AND dec > 0`) and parsed with `polars::sql::sql_expr` (polars `sql` feature). Core: `DataTable::subset(name, predicate)` returns a cheap filtered lazy clone with resolved row count. Subsets are created/deleted in the Tables tab sidebar and are browsable like tables.
- **Plot layers.** A plot is 1..N layers, each bound to a (table, subset) plus columns, drawn in order with palette colors (`Rgba::PALETTE`, Paul Tol bright). `PlotSpec.layers: Vec<LayerStyle>` styles them; the GUI holds parallel `Vec<PlotData>`. A legend appears at 2+ layers; legend entry labels are click-to-edit like all plot text (`TextElement::Legend(i)`).
- **Layer data is a snapshot.** Fetched when the layer is added; editing/deleting a subset later does not re-fetch existing layers.
- **Per-layer styling.** `LayerStyle` carries color, marker shape/size, filled/open, line width, and histogram style (bars/steps) per layer; edited in each plot tab's sidebar. Spec-level marker fields remain only as legacy defaults.
- **Plot tab layout.** Every plot tab is sidebar (Data inputs + Add layer + per-layer editors) + canvas; controls never live in a horizontal toolbar row (they overflow and hide buttons).
- **Histogram binning** is per layer via `BinSpec`: bin count (linear/log), fixed width (edges snapped to width multiples so subsets compare exactly), or explicit comma-separated edges.
- **Row-selection subsets.** `fetch_window` threads a stable row-index column (`__topdog_row__`) through filter/sort, so clicking rows in the Tables tab selects base-row identities; `DataTable::subset_from_rows` (polars `is_in`) turns a selection into a subset. Subsets can also project a chosen column list.
- **Sphere plots.** `PlotKind::Sphere` scatters lon/lat on a wireframe unit sphere (TOPCAT-style), or at true 3D positions when a distance column is given (r normalized to the max distance); far-hemisphere points are alpha-faded. Shares the 3D camera interaction.
- **3D camera state** (yaw/pitch/zoom/pan) lives in the GUI pane, not `PlotSpec` — orientation is interaction state, not figure appearance.
- **Plots draw on white regardless of app theme** so the on-screen figure matches the future exported (paper) version.

## 6. Suggested build order (phases)

Work in this order; get each phase compiling, tested, and demoable before moving to the next. Don't jump ahead to plotting polish while parquet loading is still flaky.

1. **Foundation:** workspace scaffold, `topdog-core::DataTable` wrapping a scanned parquet `LazyFrame`, column stats engine, a minimal iced app that opens a file dialog, loads a parquet file, and shows a virtualized table view with sortable/filterable columns.
2. **Core plots:** `PlotSpec` model + canvas renderer + scatter plot (with error bars) + histogram, both reading from `DataTable`. Get click-to-edit working on scatter plot labels first — treat it as the hardest and most important interaction to prove out early, not a final-polish item.
3. **More plot types:** line plots, 3D plot (rotate/pan/zoom), sky plot with Aitoff/Mollweide projections.
4. **Beyond-parity features:** model fitting overlays, per-column physical-quantity/unit metadata editing, legend/axis dynamic editing polish, table join.
5. **FITS support:** add `fitsio`-backed `TableSource` implementation behind the same trait parquet uses, so the rest of the app is format-agnostic.
6. **Export & polish:** SVG/PDF/PNG figure export, CSV/Parquet table export, theming pass.

## 7. Working conventions for Claude Code on this repo

- Before implementing a feature, check whether `topdog-core` already exposes what's needed — don't duplicate dataframe logic inside `topdog-gui`.
- Every new public function/struct gets a doc comment explaining *why*, not just *what*, when the rationale isn't obvious from the name.
- Prefer small, reviewable commits scoped to one phase/feature at a time; run `cargo test && cargo clippy --all-targets -- -D warnings && cargo fmt --check` before considering a change done.
- When adding a fixture-based test (parquet/FITS reading), keep fixture files small (a few KB, a few hundred rows) — we're testing correctness, not performance, in unit tests. Performance on large files gets validated with a separate, explicitly-large local fixture that's not committed to the repo (document how to generate it instead).
- If a TOPCAT feature is ambiguous or under-specified here, make the simplest reasonable choice, note the assumption in a code comment or commit message, and move on rather than blocking — this file can be amended later if the choice needs revisiting.
- Do not add VO/TAP/SAMP/cross-match code paths, dependencies, or even placeholder enum variants "for later." If TOPCAT source is consulted for a UI idea, skip its VO-adjacent panels entirely.
