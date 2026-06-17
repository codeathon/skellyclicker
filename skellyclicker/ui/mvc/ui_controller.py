import time
import os
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, simpledialog, messagebox, NORMAL, DISABLED

import pandas as pd
from deeplabcut.utils import auxiliaryfunctions
from pydantic import ValidationError

from skellyclicker.core.deeplabcut_handler.create_deeplabcut.deelabcut_project_config import (
    DeeplabcutTrainingConfig,
)
from skellyclicker.core.session_validation import (
    bodypart_names_from_csv_columns,
    validate_bodypart_overlap,
    validate_label_csv_against_videos,
)
from skellyclicker.ui.mvc.ui_model import SkellyClickerUIModel
from skellyclicker.core.deeplabcut_handler.deeplabcut_handler import DeeplabcutHandler
from skellyclicker.ui.mvc.ui_view import SkellyClickerUIView
from skellyclicker.core.video_handler.video_viewer import VideoViewer

DEEPLABCUT_CONFIG_FILE_NAME = "config.yaml"


@dataclass
class SkellyClickerUIController:
    ui_view: SkellyClickerUIView
    ui_model: SkellyClickerUIModel

    video_viewer: VideoViewer | None = None
    deeplabcut_handler: DeeplabcutHandler | None = None
    _background_job_running: bool = False

    def _run_background_job(
        self,
        job_name: str,
        worker,
        on_success,
        on_error=None,
    ) -> None:
        if self._background_job_running:
            messagebox.showinfo(
                "Job In Progress",
                f"Another background job is already running ({job_name}).",
            )
            return

        self._background_job_running = True
        self._set_dlc_buttons_state(DISABLED)
        print(f"Starting background job: {job_name}")

        def _worker_wrapper():
            try:
                result = worker()
                self.ui_view.root.after(0, lambda: self._finish_background_job(on_success, result))
            except Exception as error:
                tb = traceback.format_exc()
                self.ui_view.root.after(
                    0,
                    lambda: self._finish_background_job(
                        on_error or self._default_background_error,
                        (job_name, error, tb),
                    ),
                )

        threading.Thread(target=_worker_wrapper, daemon=True).start()

    def _finish_background_job(self, callback, result) -> None:
        self._background_job_running = False
        self._set_dlc_buttons_state(NORMAL)
        callback(result)

    def _default_background_error(self, result) -> None:
        job_name, error, tb = result
        print(tb)
        messagebox.showerror(
            f"{job_name} Failed",
            f"{job_name} failed:\n{error}",
        )

    def _set_dlc_buttons_state(self, state) -> None:
        self.ui_view.train_deeplabcut_model_button.config(state=state)
        self.ui_view.analyze_videos_button.config(state=state)

    def _set_machine_labels_path(self, machine_labels_path: str) -> None:
        self.ui_model.machine_labels_path = machine_labels_path
        self.ui_view.machine_labels_path_var.set(machine_labels_path)

    def _validate_before_open_videos(self) -> bool:
        warnings: list[str] = []
        if self.ui_model.csv_saved_path:
            warnings.extend(
                validate_label_csv_against_videos(
                    self.ui_model.csv_saved_path,
                    self.ui_model.video_files or [],
                    "Human labels",
                )
            )
        if self.ui_model.machine_labels_path:
            warnings.extend(
                validate_label_csv_against_videos(
                    self.ui_model.machine_labels_path,
                    self.ui_model.video_files or [],
                    "Machine labels",
                )
            )
            if self.ui_model.csv_saved_path:
                human_parts = bodypart_names_from_csv_columns(
                    list(pd.read_csv(self.ui_model.csv_saved_path, nrows=0).columns)
                )
                machine_parts = bodypart_names_from_csv_columns(
                    list(pd.read_csv(self.ui_model.machine_labels_path, nrows=0).columns)
                )
                warnings.extend(validate_bodypart_overlap(human_parts, machine_parts))

        if not warnings:
            return True

        detail = "\n".join(f"- {warning}" for warning in warnings)
        proceed = messagebox.askyesno(
            "Label CSV Mismatch",
            "Some label files do not align with the loaded videos.\n\n"
            f"{detail}\n\nOpen videos anyway?",
        )
        return proceed

    def load_deeplabcut_project(self) -> None:
        project_path = filedialog.askdirectory(
            title="Select DeepLabCut Project Directory",
            initialdir="/home/scholl-lab/deeplabcut_data"
        )
        if project_path:
            self.ui_model.project_path = project_path
            self.ui_view.deeplabcut_project_path_var.set(project_path)
            self.deeplabcut_handler = DeeplabcutHandler.load_deeplabcut_project(
                project_config_path=str(
                    Path(project_path) / DEEPLABCUT_CONFIG_FILE_NAME
                )
            )
            self.ui_view.current_iteration_var.set(
                str(self.deeplabcut_handler.iteration)
            )
            print(f"DeepLabCut project loaded from: {project_path}")

    def create_deeplabcut_project(self) -> None:
        project_path = filedialog.askdirectory(
            title="Select Directory for New DeepLabCut Project",
            initialdir="/home/scholl-lab/deeplabcut_data"
        )
        if project_path:
            project_name = simpledialog.askstring(
                "DeepLabCut Project Name", "Enter name for new deeplabcut project:"
            )
            if project_name:
                if (
                    self.ui_model.tracked_point_names is None
                    or len(self.ui_model.tracked_point_names) == 0
                ):
                    messagebox.showinfo(
                        "No Tracked Points",
                        "Load and label videos before creating a DeepLabCut project.",
                    )
                    return

                full_project_path = os.path.join(project_path, project_name)
                self.deeplabcut_handler = DeeplabcutHandler.create_deeplabcut_project(
                    project_name=project_name,
                    project_parent_directory=project_path,
                    tracked_point_names=self.ui_model.tracked_point_names,
                    connections=None,  # TODO: Handle connections somehow
                )
                self.ui_model.project_path = full_project_path
                self.ui_view.deeplabcut_project_path_var.set(full_project_path)
                self.ui_view.current_iteration_var.set(
                    str(self.deeplabcut_handler.iteration)
                )

                print(f"Creating new deeplabcut project: {full_project_path}")

    def load_videos(self) -> None:
        video_files = filedialog.askopenfilenames(
            title="Select Videos",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")],
            initialdir="/home/scholl-lab/ferret_recordings"
        )
        if video_files:
            self.ui_model.video_files = list(video_files)
            self.ui_view.open_videos_button.config(state=NORMAL)
            self.open_videos()

    def load_labels_csv(self) -> None:
        csv_file = filedialog.askopenfilename(
            title="Select Labels CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir="/home/scholl-lab/ferret_recordings"
        )
        if (
            csv_file
            and Path(csv_file).exists()
            and Path(csv_file).is_file()
            and Path(csv_file).suffix == ".csv"
        ):
            self.ui_model.csv_saved_path = csv_file
            self.ui_view.click_save_path_var.set(csv_file)
            self.ui_model.tracked_point_names = bodypart_names_from_csv_columns(
                list(pd.read_csv(csv_file, nrows=0).columns)
            )
            print(f"Labels CSV loaded from: {csv_file}")
        else:
            print("Invalid CSV file selected or file does not exist")

    def load_machine_labels_csv(self) -> None:
        machine_labels_file = filedialog.askopenfilename(
            title="Select Machine Labels CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir="/home/scholl-lab/ferret_recordings"
        )
        if (
            machine_labels_file
            and Path(machine_labels_file).exists()
            and Path(machine_labels_file).is_file()
            and Path(machine_labels_file).suffix == ".csv"
        ):
            self.ui_model.machine_labels_path = machine_labels_file
            print(f"Machine labels CSV loaded from: {machine_labels_file}")
            self.ui_view.machine_labels_path_var.set(machine_labels_file)
        else:
            print("Invalid CSV file selected or file does not exist")

    def clear_labels_csv(self) -> None:
        confirmation = messagebox.askyesno(
            "Clear Labels CSV",
            "Are you sure you want to clear the labels CSV?",
        )
        if not confirmation:
            return
        print(f"Clearing labels CSV {self.ui_model.csv_saved_path}")
        self.ui_model.csv_saved_path = None
        self.ui_view.click_save_path_var.set("")
        print("Labels CSV cleared")

    def clear_machine_labels_csv(self) -> None:
        confirmation = messagebox.askyesno(
            "Clear Machine Labels CSV",
            "Are you sure you want to clear the machine labels CSV?",
        )
        if not confirmation:
            return
        print(f"Clearing machine labels CSV {self.ui_model.machine_labels_path}")
        self.ui_model.machine_labels_path = None
        self.ui_view.machine_labels_path_var.set("")
        print("Machine labels CSV cleared")

    def open_videos(self) -> None:
        if self.ui_model.video_files:
            if not self._validate_before_open_videos():
                return

            self.ui_view.videos_directory_path_var.set(
                ", ".join(self.ui_model.video_files)
            )
            print(f"Videos loaded: {len(self.ui_model.video_files)} files")
            if self.video_viewer:
                print("Stopping previous video viewer")
                self.video_viewer.stop()
                print("Previous video viewer stopped")

            if self.ui_model.csv_saved_path:
                self.video_viewer = VideoViewer.from_videos(
                    video_paths=self.ui_model.video_files,
                    data_handler_path=self.ui_model.csv_saved_path,
                    machine_labels_path=self.ui_model.machine_labels_path,
                )
            else:
                self.video_viewer = VideoViewer.from_videos(
                    video_paths=self.ui_model.video_files,
                    machine_labels_path=self.ui_model.machine_labels_path,
                )
            self.ui_model.tracked_point_names = (
                self.video_viewer.video_handler.data_handler.config.tracked_point_names
            )
            self.ui_model.frame_count = self.video_viewer.video_handler.frame_count
            self.video_viewer.on_complete = self.video_viewer_on_complete
            print("launching videos")
            # Pass Tk root so OpenCV pumps on macOS and completion callbacks stay thread-safe.
            self.video_viewer.launch_video_thread(tk_root=self.ui_view.root)

    def video_viewer_on_complete(self) -> None:
        if self.video_viewer is None:
            print("Video viewer closed while not initialized")
            return
        save_data = messagebox.askyesno("Save Data", "Would you like to save the data?")
        if save_data is False:
            save_data = messagebox.askyesno(
                "Save Data Confirmation",
                "Confirm your choice: Click 'yes' to prevent data loss or 'no' to discard the labeled data:",
            )
        save_path = self.video_viewer.video_handler.close(
            save_data=save_data,
            save_path=self.ui_model.csv_saved_path,
        )

        if save_data and save_path:
            self.ui_model.csv_saved_path = save_path
            self.ui_view.click_save_path_var.set(save_path)
            self.update_progress()
            messagebox.showinfo("Data Saved", f"Data saved to: {save_path}")
        else:
            messagebox.showinfo("Data Not Saved", "Data not saved.")

        self.video_viewer = None

    def train_model(self) -> None:
        if not self.ui_model.project_path:
            messagebox.showinfo("No Project", "Please load or create a project first")
            return
        if self.deeplabcut_handler is None:
            messagebox.showinfo(
                "No DeepLabCut Handler", "DeepLabCut handler not initialized"
            )
            return
        if not self.ui_model.video_files:
            messagebox.showinfo(
                "No Videos",
                "Attempted to train model without loading videos, must load videos and label before training",
            )
            return
        if not self.ui_model.csv_saved_path:
            messagebox.showinfo(
                "No Data",
                "Attempted to train model without saving data, must label videos before training",
            )
            return

        training_config = DeeplabcutTrainingConfig(
            epochs=self.ui_model.training_epochs,
            save_epochs=self.ui_model.training_save_epochs,
            batch_size=self.ui_model.training_batch_size,
            hflip_augmentation=self.ui_model.hflip_augmentation,
        )

        def worker():
            self.deeplabcut_handler.train_model(
                labels_csv_path=self.ui_model.csv_saved_path,
                video_paths=self.ui_model.video_files,
                training_config=training_config,
            )
            return self.deeplabcut_handler.iteration

        def on_success(iteration: int) -> None:
            self.ui_view.current_iteration_var.set(str(iteration))
            print("Model completed training")
            messagebox.showinfo("Training Complete", "DeepLabCut training finished.")

        self._run_background_job("Train DLC Model", worker, on_success)

    def analyze_videos(self) -> None:
        if not self.ui_model.project_path:
            messagebox.showinfo("No Project", "Please load or create a project first")
            return
        if self.deeplabcut_handler is None:
            messagebox.showinfo(
                "No DeepLabCut Handler", "DeepLabCut handler not initialized"
            )
            return

        analyze_training_videos_dialog = messagebox.askyesnocancel(
            "Analyze training videos",
            "Would you like to analyze the training videos?",
            detail="Click 'yes' to analyze training videos, 'no' to select videos to analyze, or 'cancel' to cancel the operation.",
        )

        if analyze_training_videos_dialog is None:
            return
        elif analyze_training_videos_dialog is True:
            print("Analyzing videos...")
            video_paths = self.ui_model.video_files
            copy_to_machine_labels = True
            config = auxiliaryfunctions.read_config(self.deeplabcut_handler.project_config_path)
            output_folder = (
                Path(config["project_path"])
                / "model_outputs"
                / f"model_outputs_iteration_{config['iteration']}"
            )
        else:
            print("Analyzing videos...")
            copy_to_machine_labels = False
            video_paths = filedialog.askopenfilenames(
                title="Select Videos",
                filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")],
                initialdir="/home/scholl-lab/ferret_recordings"
            )
            if video_paths is not None and len(video_paths) > 0:
                config = auxiliaryfunctions.read_config(self.deeplabcut_handler.project_config_path)
                project_name = config.get("Task", "")
                output_folder = Path(video_paths[0]).parent / f"{project_name}_model_outputs_iteration_{self.deeplabcut_handler.iteration}"

        if video_paths is None or len(video_paths) == 0:
            messagebox.showinfo("No Videos", "No videos selected for analysis")
            return

        video_paths = list(video_paths)
        copy_flag = copy_to_machine_labels
        annotate_videos = self.ui_model.annotate_videos
        filter_predictions = self.ui_model.filter_predictions
        output_folder_path = output_folder
        handler = self.deeplabcut_handler

        def worker():
            machine_labels_path = handler.analyze_videos(
                video_paths=video_paths,
                annotate_videos=annotate_videos,
                filter_videos=filter_predictions,
                output_folder=output_folder_path,
            )
            return machine_labels_path, copy_flag

        def on_success(result) -> None:
            machine_labels_path, should_copy = result
            if should_copy:
                self._set_machine_labels_path(machine_labels_path)
            print("Videos analyzed")
            messagebox.showinfo(
                "Analyze Complete",
                f"Analysis finished.\nMachine labels CSV:\n{machine_labels_path}",
            )
            if should_copy and messagebox.askyesno(
                "Open Videos",
                "Re-open videos now to overlay machine labels?\n"
                "Press 'm' in the viewer to toggle overlays.",
            ):
                self.open_videos()

        self._run_background_job("Analyze Videos", worker, on_success)

    def set_save_path(self) -> None:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if file_path:
            self.ui_model.csv_saved_path = file_path
            self.ui_view.click_save_path_var.set(file_path)
            print(f"New save path set to: {file_path}")

    def save_session(self) -> None:
        output_directory = (
            Path(self.ui_model.session_saved_path).parent
            if self.ui_model.session_saved_path
            else None
        )
        output_filename = (
            Path(self.ui_model.session_saved_path).name
            if self.ui_model.session_saved_path
            else None
        )

        output_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=output_directory,
            initialfile=output_filename,
        )
        json_data = self.ui_model.model_dump_json(indent=4)

        if output_path is None or output_path == "":
            print("No valid path selected, session not saved")
            return

        with open(output_path, "w") as f:
            f.write(json_data)

        self.ui_model.session_saved_path = output_path
        print(f"Session successfully saved to: {output_path}")

    def load_session(self) -> None:
        json_file = filedialog.askopenfilename(
            title="Select Session File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="/home/scholl-lab/deeplabcut_data"
        )
        if json_file is None or json_file == "":
            return
        with open(json_file, "r") as f:
            json_data = f.read()

        try:
            self.ui_model = SkellyClickerUIModel.model_validate_json(json_data)
            self.ui_model.session_saved_path = json_file
        except ValidationError as e:
            print(f"Error loading session: {e}")
            return

        if self.deeplabcut_handler:
            print(
                "WARNING: Project loaded while deeplabcut handler already initialized, closing deeplabcut project"
            )
        if self.ui_model.project_path:
            self.ui_view.deeplabcut_project_path_var.set(self.ui_model.project_path)
            self.deeplabcut_handler = DeeplabcutHandler.load_deeplabcut_project(
                project_config_path=str(
                    Path(self.ui_model.project_path) / DEEPLABCUT_CONFIG_FILE_NAME
                )
            )
            self.ui_view.current_iteration_var.set(
                str(self.deeplabcut_handler.iteration)
            )
        else:
            self.deeplabcut_handler = None
            self.ui_view.current_iteration_var.set("None")

        self.sync_ui_with_model()

        print(f"Session successfully loaded from: {json_file}")

    def sync_ui_with_model(self) -> None:
        self.ui_view.autosave_boolean_var.set(self.ui_model.auto_save)
        self.ui_view.show_help_boolean_var.set(self.ui_model.show_help)
        self.ui_view.annotate_videos_boolean_var.set(self.ui_model.annotate_videos)
        self.ui_view.deeplabcut_filter_predictions_var.set(
            self.ui_model.filter_predictions
        )
        if self.ui_model.video_files:
            self.ui_view.videos_directory_path_var.set(
                ", ".join(self.ui_model.video_files)
            )
            self.ui_view.open_videos_button.config(state=NORMAL)
        else:
            self.ui_view.open_videos_button.config(state=DISABLED)
        if self.ui_model.csv_saved_path:
            self.ui_view.click_save_path_var.set(self.ui_model.csv_saved_path)
        if self.ui_model.machine_labels_path:
            self.ui_view.machine_labels_path_var.set(self.ui_model.machine_labels_path)
        if self.ui_model.project_path:
            self.ui_view.deeplabcut_project_path_var.set(self.ui_model.project_path)
        if self.ui_model.training_epochs:
            self.ui_view.deeplabcut_epochs_var.set(self.ui_model.training_epochs)
        if self.ui_model.training_save_epochs:
            self.ui_view.deeplabcut_save_epochs_var.set(
                self.ui_model.training_save_epochs
            )
        if self.ui_model.training_batch_size:
            self.ui_view.deeplabcut_batch_size_var.set(
                self.ui_model.training_batch_size
            )
        if self.ui_model.frame_count > 0:
            self.ui_view.labeling_progress.update(
                self.ui_model.frame_count, self.ui_model.labeled_frames
            )
        else:
            self.ui_view.labeling_progress.update(0, [])

    def on_autosave_toggle(self) -> None:
        self.ui_model.auto_save = self.ui_view.autosave_boolean_var.get()
        print(f"Auto-save set to: {self.ui_model.auto_save}")

    def on_show_help_toggle(self) -> None:
        self.ui_model.show_help = self.ui_view.show_help_boolean_var.get()
        print(f"Show help set to: {self.ui_model.show_help}")

    def on_annotate_videos_toggle(self) -> None:
        self.ui_model.annotate_videos = self.ui_view.annotate_videos_boolean_var.get()
        print(f"Annotate videos set to: {self.ui_model.annotate_videos}")

    def on_filter_predictions_toggle(self) -> None:
        self.ui_model.filter_predictions = self.ui_view.deeplabcut_filter_predictions_var.get()
        print(f"Filter predictions set to: {self.ui_model.filter_predictions}")

    def on_training_epochs_change(self) -> None:
        try:
            training_epochs = int(self.ui_view.deeplabcut_epochs_var.get())
            if training_epochs < 1:
                raise ValueError("Training epochs must be at least 1")
            self.ui_model.training_epochs = training_epochs
        except ValueError:
            messagebox.showerror(
                "Invalid Input", "Please enter a valid integer for epochs"
            )
            return
        print(f"Training epochs set to: {self.ui_model.training_epochs}")

    def on_training_save_epochs_change(self) -> None:
        try:
            save_epochs = int(self.ui_view.deeplabcut_save_epochs_var.get())
            if save_epochs < 1:
                raise ValueError("Save epochs must be at least 1")
            self.ui_model.training_save_epochs = save_epochs
        except ValueError:
            messagebox.showerror(
                "Invalid Input", "Please enter a valid integer for save epochs"
            )
            return
        print(f"Training save epochs set to: {self.ui_model.training_save_epochs}")

    def on_training_batch_size_change(self) -> None:
        try:
            batch_size = int(self.ui_view.deeplabcut_batch_size_var.get())
            if batch_size < 1:
                raise ValueError("Batch size must be at least 1")
            self.ui_model.training_batch_size = batch_size
        except ValueError:
            messagebox.showerror(
                "Invalid Input", "Please enter a valid integer for batch size"
            )
            return
        print(f"Training batch size set to: {self.ui_model.training_batch_size}")

    def update_progress(self) -> None:
        if self.video_viewer:
            total_frames = self.video_viewer.video_handler.frame_count
            labeled_frames = self.video_viewer.video_handler.data_handler.get_nonempty_frames()
            self.ui_model.frame_count = total_frames
            self.ui_model.labeled_frames = labeled_frames

        self.ui_view.labeling_progress.update(
            self.ui_model.frame_count, self.ui_model.labeled_frames
        )

    def clear_session(self) -> None:
        response = messagebox.askyesno(
            "Confirmation", "Are you sure you want to clear the session?"
        )
        if response:
            self.ui_model = SkellyClickerUIModel()
            self.sync_ui_with_model()
            print("Session cleared")

    def finish_and_close(self):
        if self.video_viewer:
            self.video_viewer.stop()

        if self.ui_model.auto_save:
            self.save_session()
            return

        save_session_answer = messagebox.askyesno(
            "Save Session", "Would you like to save this session?"
        )
        if save_session_answer is False:
            save_session_answer = messagebox.askyesno(
                "Save Session Confirmation",
                "Confirm your choice: Click 'yes' to prevent data loss or 'no' to discard session data:",
            )

        if save_session_answer:
            self.save_session()
