#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console on Windows release

use std::path::PathBuf;
use std::time::SystemTime;

use eframe::egui; // eframe re-exports egui, so no separate egui dependency is needed

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 600.0])
            .with_min_inner_size([480.0, 320.0])
            .with_title("topdog"),
        ..Default::default()
    };

    eframe::run_native(
        "topdog",
        options,
        Box::new(|_cc| Ok(Box::<TopDog>::default())),
    )
}

/// Everything we currently know about the open file.
/// Right now this is just filesystem metadata; the parquet schema/stats will
/// hang off here once we add the reader.
struct OpenFile {
    path: PathBuf,
    size_bytes: u64,
    modified: Option<SystemTime>,
    looks_like_parquet: bool,
}

impl OpenFile {
    fn from_path(path: PathBuf) -> std::io::Result<Self> {
        let meta = std::fs::metadata(&path)?;
        let looks_like_parquet = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("parquet"))
            .unwrap_or(false);
        Ok(Self {
            size_bytes: meta.len(),
            modified: meta.modified().ok(),
            looks_like_parquet,
            path,
        })
    }
}

#[derive(Default)]
struct TopDog {
    open_file: Option<OpenFile>,
    recent: Vec<PathBuf>,
    /// A transient message shown under the toolbar (warning / error / info).
    status: Option<String>,
}

/// Actions captured while drawing the UI, applied after the central panel
/// closes. This keeps us from borrowing `self` mutably while it's already
/// borrowed for rendering.
enum Action {
    OpenDialog,
    OpenPath(PathBuf),
    Close,
}

impl TopDog {
    fn open_path(&mut self, path: PathBuf) {
        match OpenFile::from_path(path.clone()) {
            Ok(file) => {
                self.status = if file.looks_like_parquet {
                    None
                } else {
                    Some(format!(
                        "Heads up: \"{}\" has no .parquet extension.",
                        path.display()
                    ))
                };

                // Move to the front of the recents list, dedup, cap at 8.
                self.recent.retain(|p| p != &path);
                self.recent.insert(0, path);
                self.recent.truncate(8);

                self.open_file = Some(file);
            }
            Err(err) => self.status = Some(format!("Couldn't open file: {err}")),
        }
    }

    fn pick_file_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Parquet", &["parquet"])
            .set_title("Open a parquet file")
            .pick_file()
        {
            self.open_path(path);
        }
    }

    /// Grab any files dropped onto the window this frame.
    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if let Some(path) = dropped.into_iter().find_map(|f| f.path) {
            self.open_path(path);
        }
    }
}

impl eframe::App for TopDog {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.handle_dropped_files(&ctx);

        let hovering_files = ctx.input(|i| !i.raw.hovered_files.is_empty());
        let mut action: Option<Action> = None;

        // In egui 0.35 the App is handed a `Ui` directly, and `CentralPanel::show`
        // now takes that `Ui` (the old `show_inside` was renamed to `show`).
        // The CentralPanel just gives us margins + a background.
        egui::CentralPanel::default().show(ui, |ui| {
            // ---- Toolbar ----
            ui.horizontal(|ui| {
                if ui.button("📂  Open parquet…").clicked() {
                    action = Some(Action::OpenDialog);
                }
                if self.open_file.is_some() && ui.button("✖  Close").clicked() {
                    action = Some(Action::Close);
                }
            });

            // ---- Status line ----
            if let Some(msg) = &self.status {
                let warn = ui.visuals().warn_fg_color;
                ui.colored_label(warn, msg);
            } else if hovering_files {
                ui.label("Release to open the file…");
            } else {
                ui.weak("Drag a .parquet file onto the window, or use Open.");
            }

            ui.separator();

            // ---- Content ----
            match self.open_file.as_ref() {
                None => draw_drop_zone(ui, &self.recent, hovering_files, &mut action),
                Some(file) => draw_file_view(ui, file),
            }
        });

        // ---- Apply captured actions ----
        match action {
            Some(Action::OpenDialog) => self.pick_file_dialog(),
            Some(Action::OpenPath(p)) => self.open_path(p),
            Some(Action::Close) => {
                self.open_file = None;
                self.status = None;
            }
            None => {}
        }
    }
}

/// Empty state: a drop target plus the recent-files list.
fn draw_drop_zone(
    ui: &mut egui::Ui,
    recent: &[PathBuf],
    hovering: bool,
    action: &mut Option<Action>,
) {
    ui.add_space(32.0);
    ui.vertical_centered(|ui| {
        ui.heading("🐕  topdog");
        ui.label("Browse and inspect parquet files.");
        ui.add_space(24.0);

        let (fill, stroke) = if hovering {
            (
                ui.visuals().selection.bg_fill,
                ui.visuals().selection.stroke,
            )
        } else {
            (
                ui.visuals().faint_bg_color,
                ui.visuals().widgets.noninteractive.bg_stroke,
            )
        };

        egui::Frame::group(ui.style())
            .fill(fill)
            .stroke(stroke)
            .inner_margin(egui::Margin::same(28))
            .show(ui, |ui| {
                ui.set_width(360.0);
                ui.vertical_centered(|ui| {
                    ui.label("Drop a .parquet file here");
                    ui.add_space(8.0);
                    if ui.button("Browse…").clicked() {
                        *action = Some(Action::OpenDialog);
                    }
                });
            });
    });

    if !recent.is_empty() {
        ui.add_space(24.0);
        ui.separator();
        ui.label(egui::RichText::new("Recent").strong());
        ui.add_space(4.0);
        for path in recent {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("<unknown>");
            // A link-styled label that reveals the full path on hover.
            let label =
                egui::Label::new(egui::RichText::new(name).underline()).sense(egui::Sense::click());
            if ui
                .add(label)
                .on_hover_text(path.display().to_string())
                .clicked()
            {
                *action = Some(Action::OpenPath(path.clone()));
            }
        }
    }
}

/// Loaded state: metadata card plus a placeholder for the data view to come.
fn draw_file_view(ui: &mut egui::Ui, file: &OpenFile) {
    let name = file
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<unknown>");

    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.heading(name);
            if !file.looks_like_parquet {
                let warn = ui.visuals().warn_fg_color;
                ui.colored_label(warn, "(not a .parquet file)");
            }
        });
        ui.add_space(4.0);

        egui::Grid::new("file_meta")
            .num_columns(2)
            .spacing([16.0, 6.0])
            .show(ui, |ui| {
                ui.label("Path");
                // Labels wrap by default in a vertical layout, so long paths are fine.
                ui.monospace(file.path.display().to_string());
                ui.end_row();

                ui.label("Size");
                ui.label(human_bytes(file.size_bytes));
                ui.end_row();

                if let Some(modified) = file.modified {
                    ui.label("Modified");
                    ui.label(format_elapsed(modified));
                    ui.end_row();
                }
            });
    });

    ui.add_space(16.0);

    // Where the schema + row preview will live once the parquet reader lands.
    let faint = ui.visuals().faint_bg_color;
    egui::Frame::group(ui.style()).fill(faint).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.vertical_centered(|ui| {
            ui.add_space(24.0);
            ui.weak("Schema and row preview will render here.");
            ui.add_space(24.0);
        });
    });
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{size:.1} {} ({bytes} bytes)", UNITS[unit])
    }
}

fn format_elapsed(modified: SystemTime) -> String {
    match modified.elapsed() {
        Ok(d) => {
            let s = d.as_secs();
            if s < 60 {
                format!("{s}s ago")
            } else if s < 3_600 {
                format!("{}m ago", s / 60)
            } else if s < 86_400 {
                format!("{}h ago", s / 3_600)
            } else {
                format!("{}d ago", s / 86_400)
            }
        }
        Err(_) => "just now".to_owned(),
    }
}
