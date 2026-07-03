#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::egui::{self, IconData};
use std::collections::BTreeSet;

#[derive(Default)]
struct MyApp {
    current_picked_path: Option<String>,
    file_paths: BTreeSet<String>,
}

impl eframe::App for MyApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("Select files").show(ui, |ui| {
            if ui.button("Load file").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("parquet", &["parquet", "parq", "pq"])
                    .pick_file()
            {
                self.current_picked_path = Some(path.display().to_string());
                self.file_paths.insert(path.display().to_string());
            }
        });

        egui::Panel::left("loaded_files").show(ui, |ui| {
            ui.label("Loaded files:");
            if !self.file_paths.is_empty() {
                for file_path in &self.file_paths {
                    if ui.selectable_label(false, file_path).clicked() {
                        egui::Panel::right("dataframe_options").show(ui, |ui| ui.label(file_path));
                    }
                }
            }
        });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("topdog")
            .with_icon(IconData::default())
            .with_inner_size([640., 240.])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "topdog",
        options,
        Box::new(|_cc| Ok(Box::<MyApp>::default())),
    )
}
