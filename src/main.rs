#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use iced::widget::{button, checkbox, column, container, row, scrollable, text};
use iced::{Element, Length, Task};
use polars::prelude::*;
use std::path::PathBuf;

#[derive(Default)]
struct App {
    files: Vec<LoadedFile>,
    selected: Option<usize>,
}

struct LoadedFile {
    path: PathBuf,
    schema: Result<Vec<ColumnInfo>, String>,
}

struct ColumnInfo {
    name: String,
    dtype: String,
    selected: bool,
}

#[derive(Debug, Clone)]
enum Message {
    OpenFile,
    FilePicked(Option<PathBuf>),
    SelectFile(usize),
    ToggleColumn(usize, bool),
    SelectAll,
    SelectNone,
}

/// Reads only the parquet schema/metadata — does not load data.
fn read_columns(path: &PathBuf) -> Result<Vec<ColumnInfo>, String> {
    let path_str = path
        .to_str()
        .ok_or_else(|| "path is not valid UTF-8".to_string())?;
    let mut lf = LazyFrame::scan_parquet(PlRefPath::new(path_str), ScanArgsParquet::default())
        .map_err(|e| e.to_string())?;
    let schema = lf.collect_schema().map_err(|e| e.to_string())?;
    Ok(schema
        .iter()
        .map(|(name, dtype)| ColumnInfo {
            name: name.to_string(),
            dtype: dtype.to_string(),
            selected: true,
        })
        .collect())
}

impl App {
    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            // Fire off the (async) file dialog; its result comes back as FilePicked.
            Message::OpenFile => {
                return Task::perform(
                    async {
                        rfd::AsyncFileDialog::new()
                            .add_filter("parquet", &["parquet", "parq", "pq"])
                            .pick_file()
                            .await
                            .map(|h| h.path().to_path_buf())
                    },
                    Message::FilePicked,
                );
            }
            Message::FilePicked(Some(path)) => {
                if !self.files.iter().any(|f| f.path == path) {
                    let schema = read_columns(&path);
                    self.files.push(LoadedFile { path, schema });
                    self.selected = Some(self.files.len() - 1);
                }
            }
            Message::FilePicked(None) => {} // user cancelled
            Message::SelectFile(i) => self.selected = Some(i),
            Message::ToggleColumn(i, checked) => {
                if let Some(Ok(cols)) = self.selected.map(|s| &mut self.files[s].schema) {
                    if let Some(c) = cols.get_mut(i) {
                        c.selected = checked;
                    }
                }
            }
            Message::SelectAll => self.set_all(true),
            Message::SelectNone => self.set_all(false),
        }
        Task::none()
    }

    fn set_all(&mut self, value: bool) {
        if let Some(Ok(cols)) = self.selected.map(|s| &mut self.files[s].schema) {
            cols.iter_mut().for_each(|c| c.selected = value);
        }
    }

    fn view(&self) -> Element<'_, Message> {
        // --- left: loaded files ---
        let mut file_list = column![text("Loaded files:").size(16)].spacing(6);
        for (i, f) in self.files.iter().enumerate() {
            let name = f
                .path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| f.path.display().to_string());
            // Coerce the two style fns to a common fn-pointer type.
            let style: fn(&iced::Theme, button::Status) -> button::Style =
                if self.selected == Some(i) {
                    button::primary
                } else {
                    button::text
                };
            file_list = file_list.push(
                button(text(name))
                    .width(Length::Fill)
                    .style(style)
                    .on_press(Message::SelectFile(i)),
            );
        }

        let sidebar = container(
            column![
                button("Load file").on_press(Message::OpenFile),
                scrollable(file_list).height(Length::Fill),
            ]
            .spacing(10),
        )
        .width(Length::Fixed(260.0))
        .padding(10);

        // --- right: column picker for the selected file ---
        let right: Element<Message> = match self.selected {
            Some(i) => column_picker(&self.files[i]),
            None => container(text("Load a parquet file and select it to choose columns."))
                .padding(10)
                .into(),
        };

        row![sidebar, right].into()
    }
}

fn column_picker(file: &LoadedFile) -> Element<'_, Message> {
    let mut content = column![text(file.path.display().to_string()).size(14)]
        .spacing(8)
        .padding(10);

    match &file.schema {
        Ok(cols) => {
            content = content.push(
                row![
                    button("All").on_press(Message::SelectAll),
                    button("None").on_press(Message::SelectNone),
                ]
                .spacing(6),
            );

            let mut list = column![].spacing(4);
            for (i, c) in cols.iter().enumerate() {
                list = list.push(
                    checkbox(c.selected)
                        .label(format!("{}  ({})", c.name, c.dtype))
                        .on_toggle(move |checked| Message::ToggleColumn(i, checked)),
                );
            }
            content = content.push(scrollable(list).height(Length::Fill));

            let n = cols.iter().filter(|c| c.selected).count();
            content = content.push(text(format!("{n} column(s) selected")));
        }
        Err(e) => {
            content = content.push(text(format!("Failed to read schema:\n{e}")));
        }
    }

    container(content).width(Length::Fill).into()
}

fn main() -> iced::Result {
    iced::application(App::default, App::update, App::view)
        .title("topdog")
        .window_size(iced::Size::new(800.0, 400.0))
        .run()
}
