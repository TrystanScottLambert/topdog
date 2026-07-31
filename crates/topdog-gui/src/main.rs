//! topdog desktop application.
//!
//! Tab-based UI: a Tables tab (browse loaded files, select rows, define
//! subsets) and one tab per plot type, each with a sidebar exposing only the
//! inputs that plot needs plus per-layer style editors. Multiple parquet
//! files can be open at once; any (table, subset) pair can be added as a
//! layer, so subsamples overplot on shared axes with a legend. All data
//! access goes through `topdog_core::DataTable`, so only the rows/aggregates
//! a view needs are ever materialized.

mod plot;

use std::collections::HashSet;
use std::path::PathBuf;

use iced::widget::{
    button, checkbox, column, container, mouse_area, pick_list, row, scrollable, slider, stack,
    text, text_input, Space,
};
use iced::{Background, Element, Font, Length, Padding, Task};
use topdog_core::{BinSpec, DataTable, RowWindow};
use topdog_plot::{HistStyle, MarkerShape, PlotSpec, RectF, Rgba, SkyProjection, TextElement};

use plot::{EditState, PlotData, PlotMessage, PlotPane};

/// Fixed table-view geometry. Uniform row height is what makes
/// virtualization trivial: row index <-> pixel offset is a multiplication.
const ROW_HEIGHT: f32 = 24.0;
const COL_WIDTH: f32 = 150.0;
const CELL_TEXT_SIZE: f32 = 13.0;
/// Rows fetched beyond the visible range on each side, so small scrolls hit
/// the cache instead of the file.
const OVERSCAN_ROWS: usize = 60;
/// Refetch when the viewport gets within this many rows of the buffer edge.
const REFETCH_MARGIN: usize = 20;

/// Point budget per scatter layer; beyond this the fetch stride-downsamples
/// (CLAUDE.md §4 — configurable in spirit; a settings UI can expose it later).
const MAX_PLOT_POINTS: usize = 200_000;
/// pick_list sentinel for "no column selected".
const NONE_OPTION: &str = "(none)";
/// widget id of the floating label editor, for focusing it on click.
const EDIT_INPUT_ID: &str = "plot-edit";
const SIDEBAR_WIDTH: f32 = 290.0;

fn main() -> iced::Result {
    iced::application(TopDog::default, update, view)
        .title(title)
        .window_size((1400.0, 850.0))
        .run()
}

/// The main tabs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Tables,
    Scatter,
    Line,
    Histogram,
    Sky,
    Sphere,
    ThreeD,
}

impl Tab {
    const ALL: [Tab; 7] = [
        Tab::Tables,
        Tab::Scatter,
        Tab::Line,
        Tab::Histogram,
        Tab::Sky,
        Tab::Sphere,
        Tab::ThreeD,
    ];

    fn label(&self) -> &'static str {
        match self {
            Tab::Tables => "Tables",
            Tab::Scatter => "Scatter",
            Tab::Line => "Line",
            Tab::Histogram => "Histogram",
            Tab::Sky => "Sky",
            Tab::Sphere => "Sphere",
            Tab::ThreeD => "3D",
        }
    }

    /// Index into `TopDog::plots` for plot tabs.
    fn plot_index(&self) -> Option<usize> {
        match self {
            Tab::Tables => None,
            Tab::Scatter => Some(0),
            Tab::Line => Some(1),
            Tab::Histogram => Some(2),
            Tab::Sky => Some(3),
            Tab::Sphere => Some(4),
            Tab::ThreeD => Some(5),
        }
    }
}

/// How the histogram tab derives its bins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinMode {
    Auto,
    Width,
    Edges,
}

impl BinMode {
    const ALL: [BinMode; 3] = [BinMode::Auto, BinMode::Width, BinMode::Edges];
}

impl std::fmt::Display for BinMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinMode::Auto => write!(f, "bin count"),
            BinMode::Width => write!(f, "bin width"),
            BinMode::Edges => write!(f, "custom edges"),
        }
    }
}

/// One opened parquet file plus its user-defined subsets.
struct LoadedTable {
    /// Where the file lives on disk — needed by the Python script generator.
    path: PathBuf,
    base: DataTable,
    numeric_columns: Vec<String>,
    subsets: Vec<SubsetEntry>,
}

/// One reproducible transformation step of a subset, from the base file.
/// The chain lets the Python generator rebuild any subset exactly.
#[derive(Debug, Clone)]
enum LineageStep {
    /// SQL predicate filter.
    Filter(String),
    /// Column projection.
    Cols(Vec<String>),
    /// Hand-picked row indices (relative to the previous step).
    Rows(Vec<u64>),
}

/// A named subsample. The filtered `DataTable` is a cheap lazy clone with
/// its row count already resolved.
struct SubsetEntry {
    name: String,
    /// Human-readable definition (predicate, column list, or "n picked rows").
    definition: String,
    /// Machine-readable derivation from the base file, for codegen.
    lineage: Vec<LineageStep>,
    table: DataTable,
}

impl LoadedTable {
    /// The `DataTable` for subset index: 0 = all rows, 1.. = subsets.
    fn subset_table(&self, ix: usize) -> &DataTable {
        if ix == 0 || ix > self.subsets.len() {
            &self.base
        } else {
            &self.subsets[ix - 1].table
        }
    }

    /// Display label for subset index (used for legend seeds).
    fn subset_label(&self, ix: usize) -> String {
        if ix == 0 || ix > self.subsets.len() {
            self.base.name().to_string()
        } else {
            self.subsets[ix - 1].name.clone()
        }
    }
}

/// State of the Tables tab: which (table, subset) is being browsed, the
/// virtualized view of it, row selection, and the new-subset form.
#[derive(Default)]
struct TablesTab {
    selected_table: usize,
    /// 0 = all rows, i = subsets[i-1].
    selected_subset: usize,
    view: Option<TableView>,
    /// Selected row indices (stable within the browsed table); cleared when
    /// the browse target changes.
    selection: HashSet<u64>,
    new_name: String,
    new_expr: String,
    /// Comma-separated column names for the subset; empty = all columns.
    new_cols: String,
    creating: bool,
    error: Option<String>,
}

/// Virtualized browse state over one `DataTable`.
struct TableView {
    table: DataTable,
    window: RowWindow,
    first_visible: usize,
    rows_in_view: usize,
    /// Bumped when sort or target table changes, so stale fetches drop.
    generation: u64,
    fetch_in_flight: bool,
}

impl TableView {
    fn new(table: DataTable) -> Self {
        Self {
            table,
            window: RowWindow {
                offset: 0,
                rows: Vec::new(),
                indices: Vec::new(),
            },
            first_visible: 0,
            rows_in_view: 40,
            generation: 0,
            fetch_in_flight: false,
        }
    }
}

/// Per-plot-tab input state + the plot itself.
struct PlotTab {
    tab: Tab,
    table_ix: usize,
    subset_ix: usize,
    x: Option<String>,
    y: Option<String>,
    z: Option<String>,
    x_err: Option<String>,
    y_err: Option<String>,
    /// Sphere tab: optional radial distance column.
    dist: Option<String>,
    /// Source of the layer currently being fetched (one in flight per tab).
    pending_source: Option<LayerSource>,
    log_bins: bool,
    bin_mode: BinMode,
    n_bins_text: String,
    width_text: String,
    edges_text: String,
    projection: SkyProjection,
    pane: Option<PlotPane>,
    /// Where each layer's data came from, parallel to `pane` layers.
    sources: Vec<LayerSource>,
    loading: bool,
}

/// Provenance of one layer: which (table, subset) and which fetch.
#[derive(Debug, Clone)]
struct LayerSource {
    table_ix: usize,
    subset_ix: usize,
    job: LayerJob,
}

impl PlotTab {
    fn new(tab: Tab) -> Self {
        Self {
            tab,
            table_ix: 0,
            subset_ix: 0,
            x: None,
            y: None,
            z: None,
            x_err: None,
            y_err: None,
            dist: None,
            pending_source: None,
            log_bins: false,
            bin_mode: BinMode::Auto,
            n_bins_text: "30".to_string(),
            width_text: String::new(),
            edges_text: String::new(),
            projection: SkyProjection::Aitoff,
            pane: None,
            sources: Vec::new(),
            loading: false,
        }
    }
}

struct TopDog {
    tables: Vec<LoadedTable>,
    active: Tab,
    tables_tab: TablesTab,
    plots: [PlotTab; 6],
    loading: bool,
    error: Option<String>,
    /// One-line success feedback (e.g. "saved figure.svg").
    status: Option<String>,
}

impl Default for TopDog {
    fn default() -> Self {
        Self {
            tables: Vec::new(),
            active: Tab::Tables,
            tables_tab: TablesTab::default(),
            plots: [
                PlotTab::new(Tab::Scatter),
                PlotTab::new(Tab::Line),
                PlotTab::new(Tab::Histogram),
                PlotTab::new(Tab::Sky),
                PlotTab::new(Tab::Sphere),
                PlotTab::new(Tab::ThreeD),
            ],
            loading: false,
            error: None,
            status: None,
        }
    }
}

impl TopDog {
    fn active_pane_mut(&mut self) -> Option<&mut PlotPane> {
        let ix = self.active.plot_index()?;
        self.plots[ix].pane.as_mut()
    }
}

/// A change to one layer's style, applied straight onto `spec.layers[i]`.
#[derive(Debug, Clone, Copy)]
enum StyleChange {
    Color(usize),
    Shape(MarkerShape),
    Filled(bool),
    Size(f32),
    LineWidth(f32),
    Hist(HistStyle),
}

/// Inputs on a plot tab (routed with the tab so every tab keeps its own).
#[derive(Debug, Clone)]
enum PlotInput {
    PickTable(String),
    PickSubset(String),
    PickX(String),
    PickY(String),
    PickZ(String),
    PickXErr(String),
    PickYErr(String),
    PickDist(String),
    ToggleLogBins(bool),
    PickBinMode(BinMode),
    NBinsChanged(String),
    WidthChanged(String),
    EdgesChanged(String),
    PickProjection(SkyProjection),
    Style { layer: usize, change: StyleChange },
    Figure(FigureChange),
    AddLayer,
    RemoveLayer(usize),
    ClearLayers,
}

/// Payload of a successfully created subset.
#[derive(Debug, Clone)]
struct NewSubset {
    table_ix: usize,
    name: String,
    definition: String,
    lineage: Vec<LineageStep>,
    table: DataTable,
}

/// What "export" means for a plot tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExportKind {
    Svg,
    Png,
    Python,
}

/// A tweak to the whole figure's appearance (fonts/ticks/grid).
#[derive(Debug, Clone, Copy)]
enum FigureChange {
    TitleSize(f32),
    LabelSize(f32),
    TickLabelSize(f32),
    TickLength(f32),
    TickWidth(f32),
    Grid(bool),
}

#[derive(Debug, Clone)]
enum Message {
    OpenFile,
    FileChosen(Option<PathBuf>),
    TableLoaded(Box<Result<(PathBuf, DataTable), String>>),
    SelectTab(Tab),
    // Tables tab
    BrowseTable(usize),
    BrowseSubset(usize),
    Scrolled(scrollable::Viewport),
    HeaderClicked(String),
    WindowFetched {
        generation: u64,
        result: Box<Result<RowWindow, String>>,
    },
    RowToggled(u64),
    ClearSelection,
    SubsetNameChanged(String),
    SubsetExprChanged(String),
    SubsetColsChanged(String),
    CreateSubset,
    CreateSubsetFromSelection,
    SubsetCreated(Box<Result<NewSubset, String>>),
    DeleteSubset(usize),
    ExportPlot {
        tab: Tab,
        kind: ExportKind,
    },
    Exported(Box<Result<Option<String>, String>>),
    // Plot tabs
    PlotInput {
        tab: Tab,
        input: PlotInput,
    },
    LayerBuilt {
        tab: Tab,
        result: Box<Result<(String, PlotSpec, PlotData), String>>,
    },
    Plot(PlotMessage),
}

fn title(state: &TopDog) -> String {
    match state.tables.len() {
        0 => "topdog".to_string(),
        1 => format!("topdog — {}", state.tables[0].base.name()),
        n => format!("topdog — {n} tables"),
    }
}

fn update(state: &mut TopDog, message: Message) -> Task<Message> {
    match message {
        Message::OpenFile => {
            if state.loading {
                return Task::none();
            }
            Task::perform(pick_file(), Message::FileChosen)
        }
        Message::FileChosen(None) => Task::none(),
        Message::FileChosen(Some(path)) => {
            state.loading = true;
            state.error = None;
            Task::perform(
                async move {
                    topdog_io::open_table(&path)
                        .map(|t| (path, t))
                        .map_err(|e| e.to_string())
                },
                |r| Message::TableLoaded(Box::new(r)),
            )
        }
        Message::TableLoaded(result) => {
            state.loading = false;
            match *result {
                Ok((path, table)) => {
                    let numeric_columns: Vec<String> = table
                        .columns()
                        .iter()
                        .filter(|c| c.is_numeric())
                        .map(|c| c.name.clone())
                        .collect();
                    state.tables.push(LoadedTable {
                        path,
                        base: table,
                        numeric_columns: numeric_columns.clone(),
                        subsets: Vec::new(),
                    });
                    let new_ix = state.tables.len() - 1;
                    // Seed unset plot-tab pickers so the first plot is one
                    // click away.
                    for pt in &mut state.plots {
                        if pt.x.is_none() {
                            pt.table_ix = new_ix;
                            pt.x = numeric_columns.first().cloned();
                            pt.y = numeric_columns.get(1).or(numeric_columns.first()).cloned();
                            pt.z = numeric_columns.get(2).cloned();
                        }
                    }
                    state.tables_tab.selected_table = new_ix;
                    state.tables_tab.selected_subset = 0;
                    state.active = Tab::Tables;
                    state.error = None;
                    rebuild_browse(state)
                }
                Err(e) => {
                    state.error = Some(e);
                    Task::none()
                }
            }
        }
        Message::SelectTab(tab) => {
            state.active = tab;
            Task::none()
        }
        Message::BrowseTable(ix) => {
            state.tables_tab.selected_table = ix;
            state.tables_tab.selected_subset = 0;
            rebuild_browse(state)
        }
        Message::BrowseSubset(ix) => {
            state.tables_tab.selected_subset = ix;
            rebuild_browse(state)
        }
        Message::Scrolled(viewport) => {
            let Some(view) = &mut state.tables_tab.view else {
                return Task::none();
            };
            view.first_visible = (viewport.absolute_offset().y / ROW_HEIGHT).floor() as usize;
            view.rows_in_view = (viewport.bounds().height / ROW_HEIGHT).ceil() as usize + 1;
            maybe_fetch(view)
        }
        Message::HeaderClicked(column) => {
            let Some(view) = &mut state.tables_tab.view else {
                return Task::none();
            };
            if let Err(e) = view.table.toggle_sort(&column) {
                state.error = Some(e.to_string());
                return Task::none();
            }
            view.generation += 1;
            view.fetch_in_flight = false;
            fetch_task(view)
        }
        Message::WindowFetched { generation, result } => {
            let Some(view) = &mut state.tables_tab.view else {
                return Task::none();
            };
            if generation != view.generation {
                return Task::none(); // stale fetch from before a sort/target change
            }
            view.fetch_in_flight = false;
            match *result {
                Ok(window) => {
                    view.window = window;
                    maybe_fetch(view)
                }
                Err(e) => {
                    state.error = Some(e);
                    Task::none()
                }
            }
        }
        Message::RowToggled(ix) => {
            if !state.tables_tab.selection.remove(&ix) {
                state.tables_tab.selection.insert(ix);
            }
            Task::none()
        }
        Message::ClearSelection => {
            state.tables_tab.selection.clear();
            Task::none()
        }
        Message::SubsetNameChanged(s) => {
            state.tables_tab.new_name = s;
            Task::none()
        }
        Message::SubsetExprChanged(s) => {
            state.tables_tab.new_expr = s;
            Task::none()
        }
        Message::SubsetColsChanged(s) => {
            state.tables_tab.new_cols = s;
            Task::none()
        }
        Message::CreateSubset => {
            let t = &state.tables_tab;
            let expr = t.new_expr.trim().to_string();
            let cols = parse_columns(&t.new_cols);
            if t.creating || state.tables.is_empty() || (expr.is_empty() && cols.is_none()) {
                return Task::none();
            }
            let table_ix = t.selected_table;
            let name = if t.new_name.trim().is_empty() {
                if expr.is_empty() {
                    "column subset".to_string()
                } else {
                    expr.clone()
                }
            } else {
                t.new_name.trim().to_string()
            };
            let definition = if expr.is_empty() {
                format!("columns: {}", t.new_cols.trim())
            } else if cols.is_some() {
                format!("{} | columns: {}", expr, t.new_cols.trim())
            } else {
                expr.clone()
            };
            let base = state.tables[table_ix].base.clone();
            let mut lineage = Vec::new();
            if !expr.is_empty() {
                lineage.push(LineageStep::Filter(expr.clone()));
            }
            if let Some(c) = &cols {
                lineage.push(LineageStep::Cols(c.clone()));
            }
            state.tables_tab.creating = true;
            state.tables_tab.error = None;
            Task::perform(
                async move {
                    let pred = (!expr.is_empty()).then_some(expr.as_str());
                    base.subset(name.clone(), pred, cols.as_deref())
                        .map(|t| NewSubset {
                            table_ix,
                            name,
                            definition,
                            lineage,
                            table: t,
                        })
                        .map_err(|e| e.to_string())
                },
                |r| Message::SubsetCreated(Box::new(r)),
            )
        }
        Message::CreateSubsetFromSelection => {
            let t = &state.tables_tab;
            if t.creating || t.selection.is_empty() {
                return Task::none();
            }
            let Some(view) = &t.view else {
                return Task::none();
            };
            let table_ix = t.selected_table;
            let mut rows: Vec<u64> = t.selection.iter().copied().collect();
            rows.sort_unstable();
            let cols = parse_columns(&t.new_cols);
            let name = if t.new_name.trim().is_empty() {
                format!("{} picked rows", rows.len())
            } else {
                t.new_name.trim().to_string()
            };
            let definition = format!("{} hand-picked rows", rows.len());
            // Selection is relative to the browsed table (which may itself
            // be a subset) — chain from it, not from the base.
            let source = view.table.clone();
            let mut lineage = if t.selected_subset == 0 {
                Vec::new()
            } else {
                state.tables[table_ix]
                    .subsets
                    .get(t.selected_subset - 1)
                    .map(|s| s.lineage.clone())
                    .unwrap_or_default()
            };
            lineage.push(LineageStep::Rows(rows.clone()));
            if let Some(c) = &cols {
                lineage.push(LineageStep::Cols(c.clone()));
            }
            state.tables_tab.creating = true;
            state.tables_tab.error = None;
            Task::perform(
                async move {
                    source
                        .subset_from_rows(name.clone(), &rows, cols.as_deref())
                        .map(|t| NewSubset {
                            table_ix,
                            name,
                            definition,
                            lineage,
                            table: t,
                        })
                        .map_err(|e| e.to_string())
                },
                |r| Message::SubsetCreated(Box::new(r)),
            )
        }
        Message::SubsetCreated(result) => {
            state.tables_tab.creating = false;
            match *result {
                Ok(new) => {
                    state.status = Some(format!("subset \"{}\" created", new.name));
                    if let Some(entry) = state.tables.get_mut(new.table_ix) {
                        entry.subsets.push(SubsetEntry {
                            name: new.name,
                            definition: new.definition,
                            lineage: new.lineage,
                            table: new.table,
                        });
                    }
                    state.tables_tab.new_name.clear();
                    state.tables_tab.new_expr.clear();
                    state.tables_tab.new_cols.clear();
                    state.tables_tab.selection.clear();
                }
                Err(e) => state.tables_tab.error = Some(e),
            }
            Task::none()
        }
        Message::ExportPlot { tab, kind } => export_plot(state, tab, kind),
        Message::Exported(result) => {
            match *result {
                Ok(Some(path)) => state.status = Some(format!("saved {path}")),
                Ok(None) => {} // dialog cancelled
                Err(e) => state.error = Some(e),
            }
            Task::none()
        }
        Message::DeleteSubset(ix) => {
            let t_ix = state.tables_tab.selected_table;
            if let Some(entry) = state.tables.get_mut(t_ix) {
                if ix < entry.subsets.len() {
                    entry.subsets.remove(ix);
                    // Keep browse selection valid; fall back to all rows.
                    if state.tables_tab.selected_subset == ix + 1 {
                        state.tables_tab.selected_subset = 0;
                        return rebuild_browse(state);
                    } else if state.tables_tab.selected_subset > ix + 1 {
                        state.tables_tab.selected_subset -= 1;
                    }
                }
            }
            Task::none()
        }
        Message::PlotInput { tab, input } => plot_input(state, tab, input),
        Message::LayerBuilt { tab, result } => {
            let Some(ix) = tab.plot_index() else {
                return Task::none();
            };
            let pt = &mut state.plots[ix];
            pt.loading = false;
            match *result {
                Ok((label, spec, data)) => {
                    match &mut pt.pane {
                        Some(pane) => pane.add_layer(label, data),
                        None => {
                            let mut pane = PlotPane::new(spec, Vec::new());
                            pane.add_layer(label, data);
                            pt.pane = Some(pane);
                        }
                    }
                    if let Some(source) = pt.pending_source.take() {
                        pt.sources.push(source);
                    }
                    state.error = None;
                }
                Err(e) => {
                    pt.pending_source = None;
                    state.error = Some(e);
                }
            }
            Task::none()
        }
        Message::Plot(PlotMessage::RegionSelected { x, y }) => region_subset(state, x, y),
        Message::Plot(msg) => {
            let Some(pane) = state.active_pane_mut() else {
                return Task::none();
            };
            match msg {
                PlotMessage::LabelClicked { element, rect } => {
                    // Commit any edit in progress, then start the new one.
                    pane.commit_edit();
                    pane.editing = Some(EditState {
                        element,
                        rect,
                        value: pane.text_of(element).to_string(),
                    });
                    pane.cache.clear();
                    iced::widget::operation::focus(EDIT_INPUT_ID)
                }
                PlotMessage::EditChanged(value) => {
                    if let Some(edit) = &mut pane.editing {
                        edit.value = value;
                    }
                    Task::none()
                }
                PlotMessage::EditSubmitted => {
                    pane.commit_edit();
                    Task::none()
                }
                PlotMessage::Rotate { dyaw, dpitch } => {
                    pane.camera.yaw += dyaw;
                    pane.camera.pitch = (pane.camera.pitch + dpitch).clamp(-1.55, 1.55);
                    pane.cache.clear();
                    Task::none()
                }
                PlotMessage::PanBy { dx, dy } => {
                    pane.camera.pan.0 += dx;
                    pane.camera.pan.1 += dy;
                    pane.cache.clear();
                    Task::none()
                }
                PlotMessage::ZoomBy(amount) => {
                    pane.camera.zoom = (pane.camera.zoom * 1.1f32.powf(amount)).clamp(0.1, 40.0);
                    pane.cache.clear();
                    Task::none()
                }
                PlotMessage::RegionSelected { .. } => unreachable!("handled above"),
            }
        }
    }
}

/// Turn a rubber-band selection into a new subset of the first layer's
/// source table: a BETWEEN predicate on that layer's plotted columns, so the
/// subset keeps every column of the source.
fn region_subset(state: &mut TopDog, x: (f64, f64), y: (f64, f64)) -> Task<Message> {
    let Some(ix) = state.active.plot_index() else {
        return Task::none();
    };
    let pt = &state.plots[ix];
    let Some(source) = pt.sources.first() else {
        return Task::none();
    };
    let (x_col, y_col): (&str, Option<&str>) = match &source.job {
        LayerJob::Scatter { x, y, .. } | LayerJob::Line { x, y } => (x, Some(y)),
        LayerJob::Histogram { column, .. } => (column, None),
        _ => return Task::none(),
    };
    let mut predicate = format!("{x_col} >= {} AND {x_col} <= {}", x.0, x.1);
    if let Some(y_col) = y_col {
        predicate.push_str(&format!(" AND {y_col} >= {} AND {y_col} <= {}", y.0, y.1));
    }

    let Some(entry) = state.tables.get(source.table_ix) else {
        return Task::none();
    };
    let table_ix = source.table_ix;
    let parent = entry.subset_table(source.subset_ix).clone();
    let mut lineage = if source.subset_ix == 0 {
        Vec::new()
    } else {
        entry
            .subsets
            .get(source.subset_ix - 1)
            .map(|s| s.lineage.clone())
            .unwrap_or_default()
    };
    lineage.push(LineageStep::Filter(predicate.clone()));
    let name = format!("region {}", entry.subsets.len() + 1);
    let definition = predicate.clone();

    Task::perform(
        async move {
            parent
                .subset(name.clone(), Some(&predicate), None)
                .map(|t| NewSubset {
                    table_ix,
                    name,
                    definition,
                    lineage,
                    table: t,
                })
                .map_err(|e| e.to_string())
        },
        |r| Message::SubsetCreated(Box::new(r)),
    )
}

/// "a, b, c" -> Some(["a","b","c"]); empty/whitespace -> None (= all).
fn parse_columns(s: &str) -> Option<Vec<String>> {
    let cols: Vec<String> = s
        .split(',')
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect();
    (!cols.is_empty()).then_some(cols)
}

/// Handle an input on a plot tab.
fn plot_input(state: &mut TopDog, tab: Tab, input: PlotInput) -> Task<Message> {
    let Some(ix) = tab.plot_index() else {
        return Task::none();
    };
    let n_tables = state.tables.len();
    let pt = &mut state.plots[ix];
    match input {
        PlotInput::PickTable(s) => {
            let new_ix = parse_ix(&s).min(n_tables.saturating_sub(1));
            if new_ix != pt.table_ix {
                pt.table_ix = new_ix;
                pt.subset_ix = 0;
                // Column picks likely don't exist in the new table; reseed.
                let cols = &state.tables[new_ix].numeric_columns;
                pt.x = cols.first().cloned();
                pt.y = cols.get(1).or(cols.first()).cloned();
                pt.z = cols.get(2).cloned();
                pt.x_err = None;
                pt.y_err = None;
                pt.dist = None;
            }
            Task::none()
        }
        PlotInput::PickSubset(s) => {
            pt.subset_ix = parse_ix(&s);
            Task::none()
        }
        PlotInput::PickX(c) => {
            pt.x = Some(c);
            Task::none()
        }
        PlotInput::PickY(c) => {
            pt.y = Some(c);
            Task::none()
        }
        PlotInput::PickZ(c) => {
            pt.z = Some(c);
            Task::none()
        }
        PlotInput::PickXErr(c) => {
            pt.x_err = (c != NONE_OPTION).then_some(c);
            Task::none()
        }
        PlotInput::PickYErr(c) => {
            pt.y_err = (c != NONE_OPTION).then_some(c);
            Task::none()
        }
        PlotInput::PickDist(c) => {
            pt.dist = (c != NONE_OPTION).then_some(c);
            Task::none()
        }
        PlotInput::ToggleLogBins(v) => {
            pt.log_bins = v;
            Task::none()
        }
        PlotInput::PickBinMode(m) => {
            pt.bin_mode = m;
            Task::none()
        }
        PlotInput::NBinsChanged(s) => {
            pt.n_bins_text = s;
            Task::none()
        }
        PlotInput::WidthChanged(s) => {
            pt.width_text = s;
            Task::none()
        }
        PlotInput::EdgesChanged(s) => {
            pt.edges_text = s;
            Task::none()
        }
        PlotInput::PickProjection(p) => {
            pt.projection = p;
            Task::none()
        }
        PlotInput::Style { layer, change } => {
            if let Some(pane) = &mut pt.pane {
                if let Some(style) = pane.spec.layers.get_mut(layer) {
                    match change {
                        StyleChange::Color(i) => style.color = Rgba::layer_color(i),
                        StyleChange::Shape(s) => style.marker = s,
                        StyleChange::Filled(f) => style.filled = f,
                        StyleChange::Size(s) => style.marker_size = s,
                        StyleChange::LineWidth(w) => style.line_width = w,
                        StyleChange::Hist(h) => style.hist_style = h,
                    }
                    pane.cache.clear();
                }
            }
            Task::none()
        }
        PlotInput::RemoveLayer(i) => {
            if let Some(pane) = &mut pt.pane {
                pane.remove_layer(i);
                if i < pt.sources.len() {
                    pt.sources.remove(i);
                }
                if pane.is_empty() {
                    pt.pane = None;
                    pt.sources.clear();
                }
            }
            Task::none()
        }
        PlotInput::ClearLayers => {
            pt.pane = None;
            pt.sources.clear();
            Task::none()
        }
        PlotInput::Figure(change) => {
            if let Some(pane) = &mut pt.pane {
                let spec = &mut pane.spec;
                match change {
                    FigureChange::TitleSize(v) => spec.title_size = v,
                    FigureChange::LabelSize(v) => spec.label_size = v,
                    FigureChange::TickLabelSize(v) => spec.tick_label_size = v,
                    FigureChange::TickLength(v) => spec.tick_length = v,
                    FigureChange::TickWidth(v) => spec.tick_width = v,
                    FigureChange::Grid(v) => spec.show_grid = v,
                }
                pane.cache.clear();
            }
            Task::none()
        }
        PlotInput::AddLayer => {
            if pt.loading || state.tables.is_empty() {
                return Task::none();
            }
            let Some(entry) = state.tables.get(pt.table_ix) else {
                return Task::none();
            };
            let table = entry.subset_table(pt.subset_ix).clone();
            let label = entry.subset_label(pt.subset_ix);

            // Capture this tab's bindings and build (spec, data) off-thread.
            let job: Option<(PlotSpec, LayerJob)> = match tab {
                Tab::Scatter => match (pt.x.clone(), pt.y.clone()) {
                    (Some(x), Some(y)) => Some((
                        PlotSpec::scatter(x.clone(), y.clone(), pt.x_err.clone(), pt.y_err.clone()),
                        LayerJob::Scatter {
                            x,
                            y,
                            x_err: pt.x_err.clone(),
                            y_err: pt.y_err.clone(),
                        },
                    )),
                    _ => None,
                },
                Tab::Line => match (pt.x.clone(), pt.y.clone()) {
                    (Some(x), Some(y)) => Some((
                        PlotSpec::line(x.clone(), y.clone()),
                        LayerJob::Line { x, y },
                    )),
                    _ => None,
                },
                Tab::Histogram => match (pt.x.clone(), parse_bins(pt)) {
                    (Some(x), Ok(bins)) => Some((
                        PlotSpec::histogram(x.clone(), 0, pt.log_bins),
                        LayerJob::Histogram { column: x, bins },
                    )),
                    (_, Err(e)) => {
                        state.error = Some(e);
                        return Task::none();
                    }
                    _ => None,
                },
                Tab::Sky => match (pt.x.clone(), pt.y.clone()) {
                    (Some(lon), Some(lat)) => Some((
                        PlotSpec::sky(lon.clone(), lat.clone(), pt.projection),
                        LayerJob::Sky { lon, lat },
                    )),
                    _ => None,
                },
                Tab::Sphere => match (pt.x.clone(), pt.y.clone()) {
                    (Some(lon), Some(lat)) => Some((
                        PlotSpec::sphere(lon.clone(), lat.clone(), pt.dist.clone()),
                        LayerJob::Sphere {
                            lon,
                            lat,
                            dist: pt.dist.clone(),
                        },
                    )),
                    _ => None,
                },
                Tab::ThreeD => match (pt.x.clone(), pt.y.clone(), pt.z.clone()) {
                    (Some(x), Some(y), Some(z)) => Some((
                        PlotSpec::scatter3d(x.clone(), y.clone(), z.clone()),
                        LayerJob::Xyz { x, y, z },
                    )),
                    _ => None,
                },
                Tab::Tables => None,
            };
            let Some((spec, job)) = job else {
                return Task::none();
            };
            pt.loading = true;
            pt.pending_source = Some(LayerSource {
                table_ix: pt.table_ix,
                subset_ix: pt.subset_ix,
                job: job.clone(),
            });
            Task::perform(
                async move {
                    job.run(&table)
                        .map(|data| (label, spec, data))
                        .map_err(|e| e.to_string())
                },
                move |result| Message::LayerBuilt {
                    tab,
                    result: Box::new(result),
                },
            )
        }
    }
}

/// Fixed export canvas: 4:3, sized so default fonts read like a figure.
const EXPORT_W: f32 = 1200.0;
const EXPORT_H: f32 = 900.0;
/// PNG oversampling factor (3600×2700 output ≈ 300 dpi at full column width).
const PNG_SCALE: f32 = 3.0;

/// Kick off an export of the given tab's plot (save dialog + write).
fn export_plot(state: &mut TopDog, tab: Tab, kind: ExportKind) -> Task<Message> {
    let Some(ix) = tab.plot_index() else {
        return Task::none();
    };
    let pt = &state.plots[ix];
    let Some(pane) = &pt.pane else {
        return Task::none();
    };
    match kind {
        ExportKind::Svg => {
            let svg = pane.to_svg(EXPORT_W, EXPORT_H);
            Task::perform(
                async move { save_bytes(svg.into_bytes(), "figure.svg", "SVG", &["svg"]).await },
                |r| Message::Exported(Box::new(r)),
            )
        }
        ExportKind::Png => {
            let svg = pane.to_svg(EXPORT_W, EXPORT_H);
            Task::perform(
                async move {
                    let png = rasterize_svg(&svg, PNG_SCALE)?;
                    save_bytes(png, "figure.png", "PNG", &["png"]).await
                },
                |r| Message::Exported(Box::new(r)),
            )
        }
        ExportKind::Python => {
            let script = python_script(state, pt);
            Task::perform(
                async move { save_bytes(script.into_bytes(), "plot.py", "Python", &["py"]).await },
                |r| Message::Exported(Box::new(r)),
            )
        }
    }
}

/// Save dialog + write. `Ok(None)` = user cancelled.
async fn save_bytes(
    bytes: Vec<u8>,
    default_name: &str,
    filter_name: &str,
    extensions: &[&str],
) -> Result<Option<String>, String> {
    let Some(handle) = rfd::AsyncFileDialog::new()
        .set_title("Save")
        .set_file_name(default_name)
        .add_filter(filter_name, extensions)
        .save_file()
        .await
    else {
        return Ok(None);
    };
    let path = handle.path().to_path_buf();
    std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
    Ok(Some(path.display().to_string()))
}

/// Rasterize the export SVG at `scale`× for print-quality PNG. Reuses the
/// exact SVG the vector export produces, so PNG and SVG cannot differ.
fn rasterize_svg(svg: &str, scale: f32) -> Result<Vec<u8>, String> {
    let mut options = resvg::usvg::Options::default();
    options.fontdb_mut().load_system_fonts();
    let tree = resvg::usvg::Tree::from_str(svg, &options).map_err(|e| e.to_string())?;
    let size = tree.size();
    let (w, h) = (
        (size.width() * scale).ceil() as u32,
        (size.height() * scale).ceil() as u32,
    );
    let mut pixmap =
        resvg::tiny_skia::Pixmap::new(w, h).ok_or_else(|| "bad export size".to_string())?;
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );
    pixmap.encode_png().map_err(|e| e.to_string())
}

/// Generate a standalone Python script (polars + matplotlib) reproducing
/// the current plot: same files, same subset lineage, same styles. This is
/// the scripting escape hatch — the data pipeline is real polars, so the
/// script is a starting point for further analysis, not just a picture.
fn python_script(state: &TopDog, pt: &PlotTab) -> String {
    use std::fmt::Write as _;
    let Some(pane) = &pt.pane else {
        return String::new();
    };
    let spec = &pane.spec;
    let mut out = String::new();
    let _ = writeln!(out, "# Generated by topdog — reproduces the current plot.");
    let _ = writeln!(out, "# Requires: pip install polars matplotlib numpy");
    let _ = writeln!(out, "import polars as pl");
    let _ = writeln!(out, "import numpy as np");
    let _ = writeln!(out, "import matplotlib.pyplot as plt");
    let _ = writeln!(out);

    // Figure setup depends on plot family.
    match &spec.kind {
        topdog_plot::PlotKind::Sky { projection, .. } => {
            let proj = match projection {
                SkyProjection::Aitoff => "aitoff",
                SkyProjection::Mollweide => "mollweide",
            };
            let _ = writeln!(out, "fig = plt.figure(figsize=(10, 6))");
            let _ = writeln!(out, "ax = fig.add_subplot(projection=\"{proj}\")");
        }
        topdog_plot::PlotKind::Sphere { .. } | topdog_plot::PlotKind::Scatter3D { .. } => {
            let _ = writeln!(out, "fig = plt.figure(figsize=(8, 8))");
            let _ = writeln!(out, "ax = fig.add_subplot(projection=\"3d\")");
        }
        _ => {
            let _ = writeln!(out, "fig, ax = plt.subplots(figsize=(8, 6))");
        }
    }
    let _ = writeln!(out);

    for (i, source) in pt.sources.iter().enumerate() {
        let style = match spec.layers.get(i) {
            Some(s) => s.clone(),
            None => continue,
        };
        let Some(entry) = state.tables.get(source.table_ix) else {
            continue;
        };
        let hex = {
            let c = style.color;
            format!(
                "#{:02x}{:02x}{:02x}",
                (c.r * 255.0) as u8,
                (c.g * 255.0) as u8,
                (c.b * 255.0) as u8
            )
        };
        let marker = match style.marker {
            MarkerShape::Circle => "o",
            MarkerShape::Square => "s",
            MarkerShape::Diamond => "D",
            MarkerShape::Cross => "+",
        };
        let label = style.label.replace('"', "'");
        let ms2 = style.marker_size * style.marker_size * 3.0; // ~area pts²

        let _ = writeln!(out, "# --- layer {i}: {label}");
        let _ = writeln!(
            out,
            "lf{i} = pl.scan_parquet(r\"{}\")",
            entry.path.display()
        );
        let lineage = if source.subset_ix == 0 {
            &[][..]
        } else {
            entry
                .subsets
                .get(source.subset_ix - 1)
                .map(|s| s.lineage.as_slice())
                .unwrap_or(&[])
        };
        for step in lineage {
            match step {
                LineageStep::Filter(p) => {
                    let _ = writeln!(out, "lf{i} = lf{i}.filter(pl.sql_expr(\"{}\"))", p);
                }
                LineageStep::Cols(cols) => {
                    let quoted: Vec<String> = cols.iter().map(|c| format!("\"{c}\"")).collect();
                    let _ = writeln!(out, "lf{i} = lf{i}.select([{}])", quoted.join(", "));
                }
                LineageStep::Rows(rows) => {
                    let list: Vec<String> = rows.iter().map(|r| r.to_string()).collect();
                    let _ = writeln!(
                        out,
                        "lf{i} = lf{i}.with_row_index(\"__row\").filter(pl.col(\"__row\").is_in([{}])).drop(\"__row\")",
                        list.join(", ")
                    );
                }
            }
        }

        match &source.job {
            LayerJob::Scatter { x, y, x_err, y_err } => {
                let mut cols = vec![x.clone(), y.clone()];
                cols.extend(x_err.clone());
                cols.extend(y_err.clone());
                let quoted: Vec<String> = cols.iter().map(|c| format!("\"{c}\"")).collect();
                let _ = writeln!(
                    out,
                    "df{i} = lf{i}.select([{}]).drop_nulls().collect()",
                    quoted.join(", ")
                );
                if x_err.is_some() || y_err.is_some() {
                    let xe = x_err
                        .as_ref()
                        .map(|c| format!("xerr=df{i}[\"{c}\"], "))
                        .unwrap_or_default();
                    let ye = y_err
                        .as_ref()
                        .map(|c| format!("yerr=df{i}[\"{c}\"], "))
                        .unwrap_or_default();
                    let _ = writeln!(
                        out,
                        "ax.errorbar(df{i}[\"{x}\"], df{i}[\"{y}\"], {xe}{ye}fmt=\"{marker}\", ms={:.1}, color=\"{hex}\", linestyle=\"none\", label=\"{label}\")",
                        style.marker_size * 2.0
                    );
                } else {
                    let fill = if style.filled {
                        String::new()
                    } else {
                        format!(", facecolors=\"none\", edgecolors=\"{hex}\"")
                    };
                    let _ = writeln!(
                        out,
                        "ax.scatter(df{i}[\"{x}\"], df{i}[\"{y}\"], s={ms2:.1}, c=\"{hex}\", marker=\"{marker}\", label=\"{label}\"{fill})"
                    );
                }
            }
            LayerJob::Line { x, y } => {
                let _ = writeln!(
                    out,
                    "df{i} = lf{i}.select([\"{x}\", \"{y}\"]).drop_nulls().sort(\"{x}\").collect()"
                );
                let _ = writeln!(
                    out,
                    "ax.plot(df{i}[\"{x}\"], df{i}[\"{y}\"], color=\"{hex}\", lw={:.1}, label=\"{label}\")",
                    style.line_width
                );
            }
            LayerJob::Histogram { column, .. } => {
                // Use the exact edges topdog computed so figures match.
                let edges = match pane.layers.get(i) {
                    Some(PlotData::Histogram(h)) => h
                        .edges
                        .iter()
                        .map(|e| format!("{e}"))
                        .collect::<Vec<_>>()
                        .join(", "),
                    _ => String::new(),
                };
                let histtype = match style.hist_style {
                    HistStyle::Bars => "bar",
                    HistStyle::Steps => "step",
                };
                let _ = writeln!(
                    out,
                    "df{i} = lf{i}.select([\"{column}\"]).drop_nulls().collect()"
                );
                let _ = writeln!(
                    out,
                    "ax.hist(df{i}[\"{column}\"], bins=[{edges}], histtype=\"{histtype}\", color=\"{hex}\", alpha=0.7, lw={:.1}, label=\"{label}\")",
                    style.line_width
                );
            }
            LayerJob::Sky { lon, lat } => {
                let _ = writeln!(
                    out,
                    "df{i} = lf{i}.select([\"{lon}\", \"{lat}\"]).drop_nulls().collect()"
                );
                let _ = writeln!(
                    out,
                    "lon{i} = np.radians(180.0 - (df{i}[\"{lon}\"].to_numpy() % 360.0))"
                );
                let _ = writeln!(
                    out,
                    "ax.scatter(lon{i}, np.radians(df{i}[\"{lat}\"].to_numpy()), s={ms2:.1}, c=\"{hex}\", marker=\"{marker}\", label=\"{label}\")"
                );
            }
            LayerJob::Sphere { lon, lat, dist } => {
                let mut cols = vec![lon.clone(), lat.clone()];
                cols.extend(dist.clone());
                let quoted: Vec<String> = cols.iter().map(|c| format!("\"{c}\"")).collect();
                let _ = writeln!(
                    out,
                    "df{i} = lf{i}.select([{}]).drop_nulls().collect()",
                    quoted.join(", ")
                );
                let r_expr = match dist {
                    Some(d) => format!("df{i}[\"{d}\"].to_numpy()"),
                    None => "1.0".to_string(),
                };
                let _ = writeln!(out, "lon{i} = np.radians(df{i}[\"{lon}\"].to_numpy())");
                let _ = writeln!(out, "lat{i} = np.radians(df{i}[\"{lat}\"].to_numpy())");
                let _ = writeln!(out, "r{i} = {r_expr}");
                let _ = writeln!(
                    out,
                    "ax.scatter(r{i}*np.cos(lat{i})*np.cos(lon{i}), r{i}*np.cos(lat{i})*np.sin(lon{i}), r{i}*np.sin(lat{i}), s={ms2:.1}, c=\"{hex}\", marker=\"{marker}\", label=\"{label}\")"
                );
            }
            LayerJob::Xyz { x, y, z } => {
                let _ = writeln!(
                    out,
                    "df{i} = lf{i}.select([\"{x}\", \"{y}\", \"{z}\"]).drop_nulls().collect()"
                );
                let _ = writeln!(
                    out,
                    "ax.scatter(df{i}[\"{x}\"], df{i}[\"{y}\"], df{i}[\"{z}\"], s={ms2:.1}, c=\"{hex}\", marker=\"{marker}\", label=\"{label}\")"
                );
            }
        }
        let _ = writeln!(out);
    }

    // Axes cosmetics mirroring the on-screen figure.
    if spec.kind.has_axes() {
        let _ = writeln!(out, "ax.set_xlabel(\"{}\")", spec.x_axis.label);
        let _ = writeln!(out, "ax.set_ylabel(\"{}\")", spec.y_axis.label);
        let _ = writeln!(
            out,
            "ax.tick_params(direction=\"in\", top=True, right=True, length={:.1}, width={:.1})",
            spec.tick_length, spec.tick_width
        );
        if spec.show_grid {
            let _ = writeln!(out, "ax.grid(alpha=0.3)");
        }
    }
    if !spec.title.is_empty() {
        let _ = writeln!(out, "ax.set_title(\"{}\")", spec.title);
    }
    if spec.layers.len() >= 2 {
        let _ = writeln!(out, "ax.legend(frameon=True, edgecolor=\"black\")");
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "plt.savefig(\"figure.png\", dpi=300, bbox_inches=\"tight\")"
    );
    let _ = writeln!(out, "plt.show()");
    out
}

/// Build the histogram `BinSpec` from the tab's inputs.
fn parse_bins(pt: &PlotTab) -> Result<BinSpec, String> {
    match pt.bin_mode {
        BinMode::Auto => {
            let n: usize = pt
                .n_bins_text
                .trim()
                .parse()
                .map_err(|_| format!("bad bin count: {:?}", pt.n_bins_text))?;
            if n == 0 || n > 100_000 {
                return Err("bin count must be between 1 and 100000".to_string());
            }
            Ok(BinSpec::Count {
                n,
                log: pt.log_bins,
            })
        }
        BinMode::Width => {
            let w: f64 = pt
                .width_text
                .trim()
                .parse()
                .map_err(|_| format!("bad bin width: {:?}", pt.width_text))?;
            Ok(BinSpec::Width(w))
        }
        BinMode::Edges => {
            let edges: Result<Vec<f64>, _> = pt
                .edges_text
                .split(',')
                .map(|s| s.trim().parse::<f64>())
                .collect();
            let edges = edges.map_err(|_| {
                "bin edges must be comma-separated numbers, e.g. 0, 0.5, 1, 2".to_string()
            })?;
            Ok(BinSpec::Edges(edges))
        }
    }
}

/// The data fetch for one layer, run on a background task.
#[derive(Debug, Clone)]
enum LayerJob {
    Scatter {
        x: String,
        y: String,
        x_err: Option<String>,
        y_err: Option<String>,
    },
    Line {
        x: String,
        y: String,
    },
    Histogram {
        column: String,
        bins: BinSpec,
    },
    Sky {
        lon: String,
        lat: String,
    },
    Sphere {
        lon: String,
        lat: String,
        dist: Option<String>,
    },
    Xyz {
        x: String,
        y: String,
        z: String,
    },
}

impl LayerJob {
    fn run(&self, table: &DataTable) -> Result<PlotData, topdog_core::CoreError> {
        match self {
            LayerJob::Scatter { x, y, x_err, y_err } => table
                .scatter_data(x, y, x_err.as_deref(), y_err.as_deref(), MAX_PLOT_POINTS)
                .map(PlotData::Scatter),
            LayerJob::Line { x, y } => table.line_data(x, y, MAX_PLOT_POINTS).map(PlotData::Line),
            LayerJob::Histogram { column, bins } => {
                table.histogram(column, bins).map(PlotData::Histogram)
            }
            LayerJob::Sky { lon, lat } => table
                .scatter_data(lon, lat, None, None, MAX_PLOT_POINTS)
                .map(PlotData::Sky),
            LayerJob::Sphere { lon, lat, dist } => {
                let mut names: Vec<&str> = vec![lon, lat];
                if let Some(d) = dist {
                    names.push(d);
                }
                table
                    .columns_data(&names, MAX_PLOT_POINTS)
                    .map(PlotData::Sphere)
            }
            LayerJob::Xyz { x, y, z } => table
                .columns_data(&[x, y, z], MAX_PLOT_POINTS)
                .map(PlotData::Xyz),
        }
    }
}

/// Point the Tables-tab browse view at the current (table, subset) and kick
/// off the first window fetch.
fn rebuild_browse(state: &mut TopDog) -> Task<Message> {
    let t = &state.tables_tab;
    let Some(entry) = state.tables.get(t.selected_table) else {
        state.tables_tab.view = None;
        return Task::none();
    };
    let generation = state
        .tables_tab
        .view
        .as_ref()
        .map(|v| v.generation + 1)
        .unwrap_or(0);
    let mut view = TableView::new(entry.subset_table(t.selected_subset).clone());
    view.generation = generation;
    let task = fetch_task(&mut view);
    state.tables_tab.view = Some(view);
    state.tables_tab.selection.clear();
    task
}

/// Start a fetch if the viewport is nearing (or outside) the cached window.
fn maybe_fetch(view: &mut TableView) -> Task<Message> {
    if view.fetch_in_flight {
        return Task::none();
    }
    let needed_start = view.first_visible;
    let needed_end = (view.first_visible + view.rows_in_view).min(view.table.n_rows());
    let buffered_start = view.window.offset;
    let buffered_end = view.window.offset + view.window.rows.len();

    let near_top = needed_start < buffered_start + REFETCH_MARGIN && buffered_start > 0;
    let near_bottom =
        needed_end + REFETCH_MARGIN > buffered_end && buffered_end < view.table.n_rows();
    if near_top || near_bottom {
        fetch_task(view)
    } else {
        Task::none()
    }
}

/// Fetch a window centered around the current viewport, on a background task.
fn fetch_task(view: &mut TableView) -> Task<Message> {
    view.fetch_in_flight = true;
    let generation = view.generation;
    let offset = view.first_visible.saturating_sub(OVERSCAN_ROWS);
    let len = view.rows_in_view + 2 * OVERSCAN_ROWS;
    let table = view.table.clone();
    Task::perform(
        async move { table.fetch_window(offset, len).map_err(|e| e.to_string()) },
        move |result| Message::WindowFetched {
            generation,
            result: Box::new(result),
        },
    )
}

async fn pick_file() -> Option<PathBuf> {
    rfd::AsyncFileDialog::new()
        .add_filter("Parquet", &["parquet", "pq"])
        .set_title("Open table")
        .pick_file()
        .await
        .map(|handle| handle.path().to_path_buf())
}

/// "3: name" -> 3. Options carry their index so duplicate names stay
/// unambiguous.
fn parse_ix(s: &str) -> usize {
    s.split(':')
        .next()
        .and_then(|p| p.parse().ok())
        .unwrap_or(0)
}

fn table_option(i: usize, entry: &LoadedTable) -> String {
    format!("{i}: {}", entry.base.name())
}

fn subset_option(i: usize, entry: &LoadedTable) -> String {
    if i == 0 {
        format!("0: all rows ({})", entry.base.n_rows())
    } else {
        format!(
            "{i}: {} ({})",
            entry.subsets[i - 1].name,
            entry.subsets[i - 1].table.n_rows()
        )
    }
}

fn view(state: &TopDog) -> Element<'_, Message> {
    let open_label = if state.loading {
        "Loading…"
    } else {
        "Open…"
    };
    let mut toolbar = row![button(open_label).on_press(Message::OpenFile)]
        .spacing(12)
        .align_y(iced::Alignment::Center)
        .padding(8);
    if let Some(error) = &state.error {
        toolbar = toolbar.push(text(error).size(13).style(text::danger));
    } else if let Some(status) = &state.status {
        toolbar = toolbar.push(text(status).size(13).style(text::success));
    }

    let tabs = row(Tab::ALL.iter().map(|t| {
        let style = if *t == state.active {
            button::primary
        } else {
            button::secondary
        };
        button(text(t.label()).size(13))
            .style(style)
            .on_press(Message::SelectTab(*t))
            .into()
    }))
    .spacing(4)
    .padding(Padding {
        top: 0.0,
        bottom: 6.0,
        left: 8.0,
        right: 8.0,
    });

    let content: Element<'_, Message> = if state.tables.is_empty() {
        container(
            text(if state.loading {
                "Loading…"
            } else {
                "Open a parquet file to get started"
            })
            .size(16),
        )
        .center(Length::Fill)
        .into()
    } else {
        match state.active {
            Tab::Tables => tables_tab(state),
            plot_tab => {
                let ix = plot_tab.plot_index().expect("plot tabs have an index");
                plot_tab_view(state, &state.plots[ix])
            }
        }
    };

    column![toolbar, tabs, content].into()
}

/// Small labelled section header for sidebars.
fn section(label: &'static str) -> Element<'static, Message> {
    text(label).size(14).into()
}

/// Tables tab: sidebar (tables, subsets, selection, new-subset form) +
/// virtualized, row-selectable table view.
fn tables_tab(state: &TopDog) -> Element<'_, Message> {
    let t = &state.tables_tab;
    let mut sidebar = column![section("Tables")].spacing(6);

    for (i, entry) in state.tables.iter().enumerate() {
        let style = if i == t.selected_table {
            button::primary
        } else {
            button::secondary
        };
        sidebar = sidebar.push(
            button(
                text(format!(
                    "{} ({} rows)",
                    entry.base.name(),
                    entry.base.n_total_rows()
                ))
                .size(13),
            )
            .style(style)
            .width(Length::Fill)
            .on_press(Message::BrowseTable(i)),
        );
    }

    if let Some(entry) = state.tables.get(t.selected_table) {
        sidebar = sidebar.push(Space::new().height(10));
        sidebar = sidebar.push(section("Subsets"));

        let all_style = if t.selected_subset == 0 {
            button::primary
        } else {
            button::secondary
        };
        sidebar = sidebar.push(
            button(text(format!("all rows ({})", entry.base.n_rows())).size(13))
                .style(all_style)
                .width(Length::Fill)
                .on_press(Message::BrowseSubset(0)),
        );
        for (i, sub) in entry.subsets.iter().enumerate() {
            let style = if t.selected_subset == i + 1 {
                button::primary
            } else {
                button::secondary
            };
            sidebar = sidebar.push(
                row![
                    button(text(format!("{} ({})", sub.name, sub.table.n_rows())).size(13))
                        .style(style)
                        .width(Length::Fill)
                        .on_press(Message::BrowseSubset(i + 1)),
                    button(text("✕").size(12)).on_press(Message::DeleteSubset(i)),
                ]
                .spacing(4),
            );
            // The defining expression, dimmed, so subsets stay auditable.
            sidebar = sidebar.push(
                text(sub.definition.clone())
                    .size(11)
                    .font(Font::MONOSPACE)
                    .style(text::secondary),
            );
        }

        sidebar = sidebar.push(Space::new().height(10));
        sidebar = sidebar.push(section("New subset"));
        sidebar = sidebar.push(
            text_input("name", &t.new_name)
                .on_input(Message::SubsetNameChanged)
                .size(13),
        );
        sidebar = sidebar.push(
            text_input("filter, e.g. mag < 20 AND dec > 0", &t.new_expr)
                .on_input(Message::SubsetExprChanged)
                .on_submit(Message::CreateSubset)
                .size(13),
        );
        sidebar = sidebar.push(
            text_input("columns, e.g. ra, dec, mag (empty = all)", &t.new_cols)
                .on_input(Message::SubsetColsChanged)
                .size(13),
        );
        if t.creating {
            sidebar = sidebar.push(text("creating…").size(13));
        } else {
            let n_sel = t.selection.len();
            let from_sel = (n_sel > 0).then_some(Message::CreateSubsetFromSelection);
            sidebar = sidebar.push(
                row![
                    button(text("From filter").size(13)).on_press(Message::CreateSubset),
                    button(text(format!("From {n_sel} selected")).size(13))
                        .on_press_maybe(from_sel),
                ]
                .spacing(4),
            );
            if n_sel > 0 {
                sidebar = sidebar.push(
                    button(text("clear selection").size(12)).on_press(Message::ClearSelection),
                );
            } else {
                sidebar = sidebar.push(
                    text("click rows in the table to select them")
                        .size(11)
                        .style(text::secondary),
                );
            }
        }
        if let Some(err) = &t.error {
            sidebar = sidebar.push(text(err).size(12).style(text::danger));
        }
    }

    let sidebar = container(scrollable(sidebar.padding(8)).height(Length::Fill))
        .width(Length::Fixed(SIDEBAR_WIDTH));

    let table_area: Element<'_, Message> = match &t.view {
        Some(view) => table_view(view, &t.selection),
        None => container(text("No table selected").size(14))
            .center(Length::Fill)
            .into(),
    };

    row![sidebar, table_area].spacing(6).into()
}

/// A plot tab: sidebar with kind-specific inputs + per-layer style editors,
/// canvas on the right.
fn plot_tab_view<'a>(state: &'a TopDog, pt: &'a PlotTab) -> Element<'a, Message> {
    let tab = pt.tab;
    let msg = move |input: PlotInput| Message::PlotInput { tab, input };

    let table_ix = pt.table_ix.min(state.tables.len() - 1);
    let entry = &state.tables[table_ix];
    let cols = entry.numeric_columns.clone();
    let mut optional_cols = vec![NONE_OPTION.to_string()];
    optional_cols.extend(cols.iter().cloned());

    let table_options: Vec<String> = state
        .tables
        .iter()
        .enumerate()
        .map(|(i, e)| table_option(i, e))
        .collect();
    let subset_options: Vec<String> = (0..=entry.subsets.len())
        .map(|i| subset_option(i, entry))
        .collect();
    let subset_ix = pt.subset_ix.min(entry.subsets.len());

    let labelled = |label: &'static str, el: Element<'a, Message>| -> Element<'a, Message> {
        row![text(label).size(13).width(Length::Fixed(52.0)), el]
            .spacing(4)
            .align_y(iced::Alignment::Center)
            .into()
    };
    let col_pick = |label: &'static str,
                    selected: &Option<String>,
                    to_input: fn(String) -> PlotInput|
     -> Element<'a, Message> {
        labelled(
            label,
            pick_list(cols.clone(), selected.clone(), move |s| msg(to_input(s)))
                .placeholder("column")
                .text_size(13)
                .width(Length::Fill)
                .into(),
        )
    };
    let opt_pick = |label: &'static str,
                    selected: &Option<String>,
                    to_input: fn(String) -> PlotInput|
     -> Element<'a, Message> {
        labelled(
            label,
            pick_list(
                optional_cols.clone(),
                Some(selected.clone().unwrap_or_else(|| NONE_OPTION.to_string())),
                move |s| msg(to_input(s)),
            )
            .text_size(13)
            .width(Length::Fill)
            .into(),
        )
    };

    let mut sidebar = column![
        section("Data"),
        labelled(
            "table",
            pick_list(
                table_options,
                Some(table_option(table_ix, entry)),
                move |s| msg(PlotInput::PickTable(s))
            )
            .text_size(13)
            .width(Length::Fill)
            .into()
        ),
        labelled(
            "subset",
            pick_list(
                subset_options,
                Some(subset_option(subset_ix, entry)),
                move |s| msg(PlotInput::PickSubset(s))
            )
            .text_size(13)
            .width(Length::Fill)
            .into()
        ),
    ]
    .spacing(6);

    match tab {
        Tab::Scatter => {
            sidebar = sidebar
                .push(col_pick("x", &pt.x, PlotInput::PickX))
                .push(col_pick("y", &pt.y, PlotInput::PickY))
                .push(opt_pick("x error", &pt.x_err, PlotInput::PickXErr))
                .push(opt_pick("y error", &pt.y_err, PlotInput::PickYErr));
        }
        Tab::Line => {
            sidebar = sidebar
                .push(col_pick("x", &pt.x, PlotInput::PickX))
                .push(col_pick("y", &pt.y, PlotInput::PickY));
        }
        Tab::Histogram => {
            sidebar = sidebar.push(col_pick("column", &pt.x, PlotInput::PickX));
            sidebar = sidebar.push(labelled(
                "bins",
                pick_list(BinMode::ALL, Some(pt.bin_mode), move |m| {
                    msg(PlotInput::PickBinMode(m))
                })
                .text_size(13)
                .width(Length::Fill)
                .into(),
            ));
            match pt.bin_mode {
                BinMode::Auto => {
                    sidebar = sidebar.push(labelled(
                        "count",
                        text_input("30", &pt.n_bins_text)
                            .on_input(move |s| msg(PlotInput::NBinsChanged(s)))
                            .size(13)
                            .into(),
                    ));
                    sidebar = sidebar.push(
                        checkbox(pt.log_bins)
                            .label("log-spaced bins")
                            .on_toggle(move |v| msg(PlotInput::ToggleLogBins(v)))
                            .text_size(13),
                    );
                }
                BinMode::Width => {
                    sidebar = sidebar.push(labelled(
                        "width",
                        text_input("e.g. 0.5", &pt.width_text)
                            .on_input(move |s| msg(PlotInput::WidthChanged(s)))
                            .size(13)
                            .into(),
                    ));
                }
                BinMode::Edges => {
                    sidebar = sidebar.push(labelled(
                        "edges",
                        text_input("e.g. 0, 0.5, 1, 2, 5", &pt.edges_text)
                            .on_input(move |s| msg(PlotInput::EdgesChanged(s)))
                            .size(13)
                            .into(),
                    ));
                }
            }
        }
        Tab::Sky => {
            sidebar = sidebar
                .push(col_pick("lon (RA)", &pt.x, PlotInput::PickX))
                .push(col_pick("lat (Dec)", &pt.y, PlotInput::PickY))
                .push(labelled(
                    "proj",
                    pick_list(SkyProjection::ALL, Some(pt.projection), move |p| {
                        msg(PlotInput::PickProjection(p))
                    })
                    .text_size(13)
                    .width(Length::Fill)
                    .into(),
                ));
        }
        Tab::Sphere => {
            sidebar = sidebar
                .push(col_pick("lon (RA)", &pt.x, PlotInput::PickX))
                .push(col_pick("lat (Dec)", &pt.y, PlotInput::PickY))
                .push(opt_pick("distance", &pt.dist, PlotInput::PickDist));
        }
        Tab::ThreeD => {
            sidebar = sidebar
                .push(col_pick("x", &pt.x, PlotInput::PickX))
                .push(col_pick("y", &pt.y, PlotInput::PickY))
                .push(col_pick("z", &pt.z, PlotInput::PickZ));
        }
        Tab::Tables => {}
    }

    let can_add = !pt.loading
        && match tab {
            Tab::Histogram => pt.x.is_some(),
            Tab::ThreeD => pt.x.is_some() && pt.y.is_some() && pt.z.is_some(),
            _ => pt.x.is_some() && pt.y.is_some(),
        };
    sidebar = sidebar.push(
        button(text(if pt.loading { "adding…" } else { "Add layer" }).size(13))
            .on_press_maybe(can_add.then(|| msg(PlotInput::AddLayer))),
    );

    if let Some(pane) = &pt.pane {
        sidebar = sidebar.push(Space::new().height(10));
        sidebar = sidebar.push(section("Layers"));
        for (i, layer) in pane.spec.layers.iter().enumerate() {
            sidebar = sidebar.push(layer_editor(tab, i, layer));
        }
        if !pane.spec.layers.is_empty() {
            sidebar = sidebar.push(
                button(text("clear all layers").size(12)).on_press(msg(PlotInput::ClearLayers)),
            );
        }

        // Figure-wide cosmetics.
        let spec = &pane.spec;
        let fig = move |change: FigureChange| msg(PlotInput::Figure(change));
        sidebar = sidebar.push(Space::new().height(10));
        sidebar = sidebar.push(section("Figure"));
        let slider_row = |label: String,
                          range: std::ops::RangeInclusive<f32>,
                          value: f32,
                          on: fn(f32) -> FigureChange|
         -> Element<'a, Message> {
            row![
                text(label).size(12).width(Length::Fixed(84.0)),
                slider(range, value, move |v| fig(on(v))).step(0.5_f32),
            ]
            .spacing(6)
            .align_y(iced::Alignment::Center)
            .into()
        };
        sidebar = sidebar.push(slider_row(
            format!("title {:.0}", spec.title_size),
            10.0..=32.0,
            spec.title_size,
            FigureChange::TitleSize,
        ));
        sidebar = sidebar.push(slider_row(
            format!("labels {:.0}", spec.label_size),
            8.0..=26.0,
            spec.label_size,
            FigureChange::LabelSize,
        ));
        sidebar = sidebar.push(slider_row(
            format!("ticks {:.0}", spec.tick_label_size),
            6.0..=20.0,
            spec.tick_label_size,
            FigureChange::TickLabelSize,
        ));
        if spec.kind.has_axes() {
            sidebar = sidebar.push(slider_row(
                format!("tick len {:.1}", spec.tick_length),
                0.0..=15.0,
                spec.tick_length,
                FigureChange::TickLength,
            ));
            sidebar = sidebar.push(slider_row(
                format!("tick wd {:.1}", spec.tick_width),
                0.5..=4.0,
                spec.tick_width,
                FigureChange::TickWidth,
            ));
            sidebar = sidebar.push(
                checkbox(spec.show_grid)
                    .label("grid")
                    .on_toggle(move |v| fig(FigureChange::Grid(v)))
                    .text_size(12),
            );
        }

        // Export.
        sidebar = sidebar.push(Space::new().height(10));
        sidebar = sidebar.push(section("Export"));
        let export = |kind: ExportKind| Message::ExportPlot { tab, kind };
        sidebar = sidebar.push(
            row![
                button(text("SVG").size(13)).on_press(export(ExportKind::Svg)),
                button(text("PNG").size(13)).on_press(export(ExportKind::Png)),
                button(text("Python").size(13)).on_press(export(ExportKind::Python)),
            ]
            .spacing(6),
        );
        sidebar = sidebar.push(
            text("PNG saves at 3× for print; Python writes a polars+matplotlib script")
                .size(11)
                .style(text::secondary),
        );
        if pane.supports_region_select() {
            sidebar = sidebar.push(
                text("drag on the plot to select a region as a new subset")
                    .size(11)
                    .style(text::secondary),
            );
        }
    }

    let sidebar = container(scrollable(sidebar.padding(8)).height(Length::Fill))
        .width(Length::Fixed(SIDEBAR_WIDTH));

    let canvas_area: Element<'a, Message> = match &pt.pane {
        Some(pane) => match &pane.editing {
            None => pane.canvas(),
            Some(edit) => stack![pane.canvas(), edit_overlay(pane, edit)].into(),
        },
        None => container(text("Pick a table, subset, and columns, then Add layer").size(14))
            .center(Length::Fill)
            .into(),
    };

    row![sidebar, canvas_area].spacing(6).into()
}

/// Style controls for one layer: color palette, shape/fill for point plots,
/// widths and bin style where they apply.
fn layer_editor(tab: Tab, i: usize, layer: &topdog_plot::LayerStyle) -> Element<'_, Message> {
    let msg = move |change: StyleChange| Message::PlotInput {
        tab,
        input: PlotInput::Style { layer: i, change },
    };

    let mut editor = column![row![
        text(format!("■ {}", layer.label)).size(12).style({
            let c = layer.color;
            move |_: &iced::Theme| text::Style {
                color: Some(iced::Color::from_rgba(c.r, c.g, c.b, c.a)),
            }
        }),
        Space::new().width(Length::Fill),
        button(text("✕").size(10)).on_press(Message::PlotInput {
            tab,
            input: PlotInput::RemoveLayer(i)
        }),
    ]
    .align_y(iced::Alignment::Center)]
    .spacing(4);

    // Color swatches from the shared palette.
    editor = editor.push(
        row(Rgba::PALETTE.iter().enumerate().map(|(ci, c)| {
            let color = iced::Color::from_rgba(c.r, c.g, c.b, c.a);
            button(Space::new().width(14).height(14))
                .style(move |_theme: &iced::Theme, _status| button::Style {
                    background: Some(Background::Color(color)),
                    ..button::Style::default()
                })
                .on_press(msg(StyleChange::Color(ci)))
                .into()
        }))
        .spacing(4),
    );

    let uses_markers = matches!(tab, Tab::Scatter | Tab::Sky | Tab::Sphere | Tab::ThreeD);
    if uses_markers {
        editor = editor.push(
            row![
                pick_list(MarkerShape::ALL, Some(layer.marker), move |s| msg(
                    StyleChange::Shape(s)
                ))
                .text_size(12)
                .width(Length::Fill),
                checkbox(layer.filled)
                    .label("filled")
                    .on_toggle(move |f| msg(StyleChange::Filled(f)))
                    .text_size(12),
            ]
            .spacing(6)
            .align_y(iced::Alignment::Center),
        );
        editor = editor.push(
            row![
                text(format!("size {:.1}", layer.marker_size)).size(12),
                slider(0.5..=12.0, layer.marker_size, move |v| msg(
                    StyleChange::Size(v)
                ))
                .step(0.5_f32),
            ]
            .spacing(6)
            .align_y(iced::Alignment::Center),
        );
    }
    let uses_lines = matches!(tab, Tab::Line | Tab::Histogram) || (uses_markers && !layer.filled);
    if uses_lines {
        editor = editor.push(
            row![
                text(format!("line {:.1}", layer.line_width)).size(12),
                slider(0.5..=8.0, layer.line_width, move |v| msg(
                    StyleChange::LineWidth(v)
                ))
                .step(0.5_f32),
            ]
            .spacing(6)
            .align_y(iced::Alignment::Center),
        );
    }
    if tab == Tab::Histogram {
        editor = editor.push(
            pick_list(HistStyle::ALL, Some(layer.hist_style), move |h| {
                msg(StyleChange::Hist(h))
            })
            .text_size(12)
            .width(Length::Fill),
        );
    }

    container(editor)
        .padding(6)
        .width(Length::Fill)
        .style(container::bordered_box)
        .into()
}

fn edit_overlay<'a>(pane: &'a PlotPane, edit: &'a EditState) -> Element<'a, Message> {
    let size = match edit.element {
        TextElement::Title => pane.spec.title_size,
        TextElement::XLabel | TextElement::YLabel => pane.spec.label_size,
        TextElement::Legend(_) => pane.spec.tick_label_size,
    };
    // The y-label is drawn rotated; its overlay input is horizontal at the
    // same spot — a pragmatic tradeoff that keeps the pattern uniform.
    let rect: RectF = edit.rect;
    let input = text_input("label…", &edit.value)
        .id(EDIT_INPUT_ID)
        .on_input(|s| Message::Plot(PlotMessage::EditChanged(s)))
        .on_submit(Message::Plot(PlotMessage::EditSubmitted))
        .size(size)
        .width(Length::Fixed(rect.width.max(160.0)));

    container(input)
        .padding(Padding {
            top: rect.y,
            left: rect.x,
            right: 0.0,
            bottom: 0.0,
        })
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

/// Header row + virtualized body. Both live inside one horizontal scrollable
/// so the header stays vertically pinned but scrolls sideways with the data.
/// Rows are clickable to toggle selection (highlighted).
fn table_view<'a>(view: &'a TableView, selection: &HashSet<u64>) -> Element<'a, Message> {
    let header = row(view.table.columns().iter().map(|c| {
        let marker = match view.table.sort_spec() {
            Some(s) if s.column == c.name && s.descending => " ▼",
            Some(s) if s.column == c.name => " ▲",
            _ => "",
        };
        button(
            text(format!("{}{marker}", c.name))
                .size(CELL_TEXT_SIZE)
                .font(Font::MONOSPACE),
        )
        .width(Length::Fixed(COL_WIDTH))
        .on_press(Message::HeaderClicked(c.name.clone()))
        .into()
    }));

    let numeric: Vec<bool> = view
        .table
        .columns()
        .iter()
        .map(|c| c.is_numeric())
        .collect();
    let n_rows = view.table.n_rows();
    let window = &view.window;

    // Virtualization: real widgets only for the buffered window, blank
    // spacers stand in for everything above and below so the scrollbar
    // geometry matches the full table.
    let top_pad = window.offset as f32 * ROW_HEIGHT;
    let below = n_rows.saturating_sub(window.offset + window.rows.len());
    let bottom_pad = below as f32 * ROW_HEIGHT;

    let rows_col = column(
        std::iter::once(Space::new().height(top_pad).into())
            .chain(
                window
                    .rows
                    .iter()
                    .zip(window.indices.iter())
                    .map(|(r, &base_ix)| {
                        data_row(r, &numeric, base_ix, selection.contains(&base_ix))
                    }),
            )
            .chain(std::iter::once(Space::new().height(bottom_pad).into())),
    );

    let body = scrollable(rows_col)
        .on_scroll(Message::Scrolled)
        .height(Length::Fill)
        .width(Length::Shrink);

    let table_width = COL_WIDTH * view.table.n_columns() as f32;
    scrollable(container(column![header, body]).width(Length::Fixed(table_width)))
        .direction(scrollable::Direction::Horizontal(
            scrollable::Scrollbar::new(),
        ))
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn data_row<'a>(
    cells: &'a [String],
    numeric: &[bool],
    base_ix: u64,
    selected: bool,
) -> Element<'a, Message> {
    let content = row(cells.iter().zip(numeric).map(|(value, &is_num)| {
        text(value)
            .size(CELL_TEXT_SIZE)
            .font(Font::MONOSPACE)
            .width(Length::Fixed(COL_WIDTH))
            .align_x(if is_num {
                iced::alignment::Horizontal::Right
            } else {
                iced::alignment::Horizontal::Left
            })
            .into()
    }))
    .height(Length::Fixed(ROW_HEIGHT));

    let styled = container(content).style(move |theme: &iced::Theme| {
        if selected {
            container::Style {
                background: Some(Background::Color(
                    theme.extended_palette().primary.weak.color,
                )),
                ..container::Style::default()
            }
        } else {
            container::Style::default()
        }
    });

    mouse_area(styled)
        .on_press(Message::RowToggled(base_ix))
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rasterize_produces_scaled_png() {
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100\" height=\"50\" \
                   viewBox=\"0 0 100 50\"><rect x=\"0\" y=\"0\" width=\"100\" height=\"50\" \
                   fill=\"rgb(255,255,255)\"/><circle cx=\"50\" cy=\"25\" r=\"10\" \
                   fill=\"rgb(0,0,0)\"/></svg>";
        let png = rasterize_svg(svg, 3.0).expect("rasterizes");
        // PNG magic number, and dimensions scaled 3x (IHDR: bytes 16..24).
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        let width = u32::from_be_bytes([png[16], png[17], png[18], png[19]]);
        let height = u32::from_be_bytes([png[20], png[21], png[22], png[23]]);
        assert_eq!((width, height), (300, 150));
    }

    #[test]
    fn rasterize_rejects_malformed_svg() {
        assert!(rasterize_svg("not an svg at all", 1.0).is_err());
    }

    #[test]
    fn parse_columns_splits_and_trims() {
        assert_eq!(
            parse_columns(" ra, dec ,mag "),
            Some(vec!["ra".into(), "dec".into(), "mag".into()])
        );
        assert_eq!(parse_columns("   "), None);
        assert_eq!(parse_columns(""), None);
        assert_eq!(parse_columns("a,,b"), Some(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn parse_ix_reads_option_prefix() {
        assert_eq!(parse_ix("0: all rows (300)"), 0);
        assert_eq!(parse_ix("12: my subset (7)"), 12);
        assert_eq!(parse_ix("nonsense"), 0);
    }

    #[test]
    fn bin_spec_parsed_from_tab_inputs() {
        let mut pt = PlotTab::new(Tab::Histogram);
        assert!(matches!(
            parse_bins(&pt),
            Ok(BinSpec::Count { n: 30, log: false })
        ));

        pt.log_bins = true;
        assert!(matches!(
            parse_bins(&pt),
            Ok(BinSpec::Count { n: 30, log: true })
        ));

        pt.n_bins_text = "0".into();
        assert!(parse_bins(&pt).is_err());
        pt.n_bins_text = "abc".into();
        assert!(parse_bins(&pt).is_err());

        pt.bin_mode = BinMode::Width;
        pt.width_text = "0.25".into();
        assert!(matches!(parse_bins(&pt), Ok(BinSpec::Width(w)) if (w - 0.25).abs() < 1e-12));

        pt.bin_mode = BinMode::Edges;
        pt.edges_text = "0, 1.5, 3".into();
        match parse_bins(&pt) {
            Ok(BinSpec::Edges(e)) => assert_eq!(e, vec![0.0, 1.5, 3.0]),
            other => panic!("expected edges, got {other:?}"),
        }
        pt.edges_text = "0, oops".into();
        assert!(parse_bins(&pt).is_err());
    }
    /// Build a minimal app state with one table and one scatter layer.
    fn state_with_scatter_layer() -> (TopDog, PlotTab) {
        use topdog_core::DataTable;
        let df = polars::df![
            "ra" => [10.0f64, 20.0, 30.0],
            "dec" => [1.0f64, 2.0, 3.0],
        ]
        .unwrap();
        let table = DataTable::from_lazy("cat", polars::prelude::IntoLazy::lazy(df)).unwrap();
        let sub = table.subset("bright", Some("dec > 1.5"), None).unwrap();

        let mut state = TopDog::default();
        state.tables.push(LoadedTable {
            path: PathBuf::from("/data/cat.parquet"),
            base: table,
            numeric_columns: vec!["ra".into(), "dec".into()],
            subsets: vec![SubsetEntry {
                name: "bright".into(),
                definition: "dec > 1.5".into(),
                lineage: vec![LineageStep::Filter("dec > 1.5".into())],
                table: sub,
            }],
        });

        let mut pt = PlotTab::new(Tab::Scatter);
        let mut pane = PlotPane::new(PlotSpec::scatter("ra", "dec", None, None), Vec::new());
        pane.add_layer(
            "bright".into(),
            PlotData::Scatter(topdog_core::ScatterData {
                x: vec![20.0, 30.0],
                y: vec![2.0, 3.0],
                x_err: None,
                y_err: None,
                total_points: 2,
                stride: 1,
            }),
        );
        pane.spec.title = "My figure".into();
        pt.pane = Some(pane);
        pt.sources.push(LayerSource {
            table_ix: 0,
            subset_ix: 1,
            job: LayerJob::Scatter {
                x: "ra".into(),
                y: "dec".into(),
                x_err: None,
                y_err: None,
            },
        });
        (state, pt)
    }

    #[test]
    fn python_script_reproduces_source_and_style() {
        let (state, pt) = state_with_scatter_layer();
        let script = python_script(&state, &pt);
        println!("{script}");
        assert!(script.contains("import polars as pl"));
        assert!(script.contains("import matplotlib.pyplot as plt"));
        // Source file and the subset's filter lineage are both reproduced.
        assert!(script.contains("scan_parquet(r\"/data/cat.parquet\")"));
        assert!(script.contains("pl.sql_expr(\"dec > 1.5\")"));
        // Plot call, labels, inward ticks, and title.
        assert!(script.contains("ax.scatter("));
        assert!(script.contains("ax.set_xlabel(\"ra\")"));
        assert!(script.contains("ax.set_title(\"My figure\")"));
        assert!(script.contains("direction=\"in\", top=True, right=True"));
        assert!(script.contains("savefig"));
    }
}
