from datetime import datetime
import json
import logging
import math
from collections.abc import Callable
import cv2
import deeplabcut
from deeplabcut import DEBUG
from deeplabcut.utils import auxiliaryfunctions
from multiprocessing import Pool
from pathlib import Path
from pydantic import BaseModel
from time import perf_counter_ns

from skellyclicker.core.click_data_handler.data_handler import DataHandler
from skellyclicker.core.video_handler.image_annotator import ImageAnnotator, ImageAnnotatorConfig
from skellyclicker.core.deeplabcut_handler.create_deeplabcut.create_deeplabcut_config import (
    create_new_deeplabcut_project,
)
from skellyclicker.core.deeplabcut_handler.create_deeplabcut.create_deeplabcut_project_data import (
    fill_in_labelled_data_folder,
)
from skellyclicker.core.deeplabcut_handler.create_deeplabcut.deelabcut_project_config import (
    DeeplabcutTrainingConfig,
)
from skellyclicker.core.deeplabcut_handler.analyze_videos_dlc import analyze_videos_dlc
from skellyclicker.core.deeplabcut_handler.partial_analyze_dlc import partial_analyze_human_labels
from skellyclicker.core.deeplabcut_handler.dlc_csv_io import (
	merge_dlc_csvs_for_skellyclicker,
)


logger = logging.getLogger(__name__)


class PointConnection(BaseModel):
    parent: str
    child: str

    @classmethod
    def from_tuple(cls, points: tuple[str, str]):
        return cls(parent=points[0], child=points[1])

    @property
    def as_tuple(self) -> tuple[str, str]:
        return self.parent, self.child

    @property
    def as_list(self) -> list[str]:
        return list(self.as_tuple)


class DeeplabcutHandler(BaseModel):
    project_name: str
    project_config_path: str
    iteration: int
    tracked_point_names: list[str]
    connections: list[PointConnection] | None

    @classmethod
    def create_deeplabcut_project(
        cls,
        project_name: str,
        project_parent_directory: str,
        tracked_point_names: list[str],
        connections: list[PointConnection] | None = None,
    ):
        logger.info("Creating deeplabcut project structure...")

        if connections is None:
            connections = []
        return cls(
            project_name=project_name,
            connections=connections,
            iteration=0,
            tracked_point_names=tracked_point_names,
            project_config_path=create_new_deeplabcut_project(
                project_name=project_name,
                project_parent_directory=project_parent_directory,
                bodyparts=tracked_point_names,
                skeleton=[connection.as_list for connection in connections],
            ),
        )

    @classmethod
    def load_deeplabcut_project(cls, project_config_path: str):
        logger.info(f"Loading deeplabcut project from config: {project_config_path}")
        config = auxiliaryfunctions.read_config(project_config_path)

        return cls(
            project_name=config["Task"],
            tracked_point_names=config["bodyparts"],
            iteration=config["iteration"],
            connections=[
                PointConnection.from_tuple(connection)
                for connection in config["skeleton"]
            ],
            project_config_path=project_config_path,
        )

    def _bump_iteration(self):
        config = auxiliaryfunctions.read_config(self.project_config_path)

        shuffles_path = Path(self.project_config_path).parent / "training-datasets"
        results_path = Path(self.project_config_path).parent / "dlc-models"

        # bump the iteration in the config file
        config["iteration"] += 1
        iteration_count = int(config["iteration"])
        logger.info(f"Bumped iteration to: {iteration_count}")

        # Create common subdirectories for training-datasets
        iteration_path = shuffles_path / f"iteration-{iteration_count}"
        iteration_path.mkdir(parents=True, exist_ok=bool(DEBUG))
        logger.info(f"Created training dataset directory: {iteration_path}")

        # Create common subdirectories for dlc-models
        model_iteration_path = results_path / f"iteration-{iteration_count}"
        model_iteration_path.mkdir(parents=True, exist_ok=bool(DEBUG))
        logger.info(f"Created model directory: {model_iteration_path}")

        auxiliaryfunctions.write_config(self.project_config_path, config)
        self.iteration = iteration_count
        logger.info(f"Saved updated config file: {self.project_config_path}")

    def train_model(
        self,
        labels_csv_path: str | None = None,
        video_paths: list[str] | None = None,
        training_config: DeeplabcutTrainingConfig | None = None,
        progress_callback: Callable[[float | None, str], None] | None = None,
    ):
        def report(fraction: float | None, message: str) -> None:
            if progress_callback:
                progress_callback(fraction, message)

        if training_config is None:
            training_config = DeeplabcutTrainingConfig()

        if not video_paths:
            raise ValueError("Select videos before training")

        # Path registry allows videos in different folders (cross-experiment corpus).
        from skellyclicker.services.video_path_registry import build_video_path_registry
        from skellyclicker.core.deeplabcut_handler.labeled_data_io import (
            has_human_labels_for_videos,
            is_legacy_skellyclicker_csv,
            labeled_data_dir,
            labeled_data_session_subset,
            regenerate_all_collected_data_h5,
            resolve_human_labels_root,
        )

        registry = build_video_path_registry(video_paths)
        # Legacy single-folder arg still used as fallback prefix when registry unused.
        video_folder = Path(next(iter(registry.values()))).parent

        parent_directory = Path(self.project_config_path).parent
        labeled_root = labeled_data_dir(parent_directory)

        if (
            parent_directory / "dlc-models-pytorch" / f"iteration-{self.iteration}"
        ).exists():
            logger.info(
                "Model detected for current iteration, bumping to next iteration"
            )
            self._bump_iteration()

        # Human labels live in labeled-data. Only convert when a legacy flat CSV
        # is still passed (CLI / one-shot import); web train skips this.
        if labels_csv_path and Path(labels_csv_path).is_file() and is_legacy_skellyclicker_csv(
            labels_csv_path
        ):
            report(0.05, "Processing labeled frames…")
            fill_in_labelled_data_folder(
                path_to_videos_for_training=str(video_folder),
                path_to_dlc_project_folder=str(parent_directory),
                path_to_image_labels_csv=labels_csv_path,
                video_paths=video_paths,
            )
        elif labels_csv_path:
            try:
                labeled_root = resolve_human_labels_root(labels_csv_path)
            except ValueError:
                labeled_root = labeled_data_dir(parent_directory)

        # Only UI-selected videos — ignore other labeled-data folders in the project.
        if not has_human_labels_for_videos(labeled_root, video_paths):
            raise ValueError(
                "No human labels for the videos in this session. "
                "Open the labeler, label the selected videos, and save before training."
            )

        report(0.08, "Refreshing labeled-data H5 for session videos…")
        regenerate_all_collected_data_h5(labeled_root, video_paths=video_paths)

        report(0.15, f"Creating training dataset ({training_config.model_type})…")
        # Hide non-session labeled-data folders so DLC does not train on them.
        with labeled_data_session_subset(labeled_root, video_paths):
            deeplabcut.create_training_dataset(
                self.project_config_path, net_type=training_config.model_type
            )
        # deeplabcut.create_training_model_comparison(self.project_config_path, net_types=["resnet_50", "rtmpose_x"])


        batch_size = training_config.batch_size
        learning_rate = training_config.learning_rate
        if batch_size > 1:
            learning_rate *= math.floor(math.sqrt(batch_size))
            training_config.learning_rate = learning_rate
            print(f"Adjusted default learning rate to scale with square root of batch size")
            print("See https://stackoverflow.com/questions/64105986/in-2020-what-is-the-optimal-way-to-train-a-model-in-pytorch-on-more-than-one-gpu")

        pytorch_cfg_updates = {
            "runner.optimizer.params.lr": training_config.learning_rate
        }
        if training_config.hflip_augmentation:
            pytorch_cfg_updates["data.train.hflip"] = True
        logger.info("Training model...")
        logger.info(f"With config: epochs={training_config.epochs}, save epochs={training_config.save_epochs}, batch_size={training_config.batch_size}, learning_rate={training_config.learning_rate}")
        report(
            0.15,
            f"Training network ({training_config.epochs} epochs)…",
        )
        start_time = perf_counter_ns()
        from skellyclicker.core.deeplabcut_handler.dlc_progress import (
            hook_dlc_training_progress,
        )

        with hook_dlc_training_progress(report):
            deeplabcut.train_network(
                self.project_config_path,
                epochs=training_config.epochs,
                save_epochs=training_config.save_epochs,
                batch_size=training_config.batch_size,
                pytorch_cfg_updates=pytorch_cfg_updates
            )
        end_time = perf_counter_ns()
        report(1.0, "Training finished")
        print(f"Model training took {(end_time-start_time)/1e9} seconds over {training_config.epochs} epochs ({(end_time-start_time)/(1e9*training_config.epochs)} s per epoch)")

    def analyze_videos(
        self,
        video_paths: list[str],
        output_folder: str | Path,
        annotate_videos: bool = False,
        filter_videos: bool = True,
        max_parallel_videos: int | None = None,
        progress_callback: Callable[[float | None, str], None] | None = None,
    ) -> str:
        from skellyclicker.services.dlc_paths import (
            dlc_project_dir,
            resolve_analyze_iteration,
        )
        from skellyclicker.core.deeplabcut_handler.parallel_analyze import (
            analyze_videos_parallel,
            resolve_worker_count,
        )

        def report(fraction: float | None, message: str) -> None:
            if progress_callback:
                progress_callback(fraction, message)

        project_dir = dlc_project_dir(self.project_config_path)
        config = auxiliaryfunctions.read_config(self.project_config_path)
        analyze_iteration = resolve_analyze_iteration(project_dir, config)
        self.iteration = analyze_iteration
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        n_videos = max(len(video_paths), 1)
        # Auto (None/0) -> one worker per GPU; single GPU / one video -> sequential.
        workers = resolve_worker_count(len(video_paths), max_parallel_videos)
        analyze_kwargs = dict(
            config=str(self.project_config_path),
            videotype=".mp4",
            save_as_csv=True,
            destfolder=str(output_folder),
            batch_size=8,
            overwrite=True,
        )

        if workers > 1:
            report(0.05, f"Analyzing {n_videos} video(s) across {workers} GPU worker(s)…")
            analyze_videos_parallel(
                video_paths=video_paths,
                analyze_kwargs=analyze_kwargs,
                worker_count=workers,
                progress_callback=progress_callback,
            )
        else:
            report(0.05, f"Analyzing {n_videos} video(s)…")
            # multiprocess only when no progress callback (legacy script path).
            analyze_videos_dlc(
                videos=video_paths,
                multiprocess=progress_callback is None,
                progress_callback=progress_callback,
                **analyze_kwargs,
            )
        report(0.75, "Inference complete")

        if filter_videos:
            report(0.78, "Filtering predictions…")
            deeplabcut.filterpredictions(
                str(self.project_config_path),
                video_paths,
                videotype=".mp4",
                filtertype="median",
                windowlength=5,
                destfolder=str(output_folder),
            )

        csv_path = Path(output_folder) / f"skellyclicker_machine_labels_iteration_{analyze_iteration}.csv"

        # Cross-folder analyze is allowed; merge still keys rows by video basename.
        config = auxiliaryfunctions.read_config(self.project_config_path)
        metadata = {
            "model_name": self.project_name,
            "scorer": config.get("scorer"),
            "iteration": self.iteration,
            "project_creation_date": config.get("date"),
            "processing_datetime": datetime.now().isoformat(),
            "project_config_path": str(self.project_config_path),
            "tracked_point_names": self.tracked_point_names,
            "connections": [c.model_dump() for c in self.connections] if self.connections else [],
            "video_paths": [str(v) for v in video_paths],
            "csv_path": str(csv_path),
            "output_path": str(output_folder),
        }
        metadata_path = Path(output_folder) / f"skellyclicker_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved annotation metadata to {metadata_path}")

        # Merge + sidecars before plot/annotate so large multi-video runs still
        # produce machine CSVs if matplotlib/OpenCV later aborts.
        def on_merge_progress(done: int, total: int, video_label: str) -> None:
            if total <= 0:
                return
            # Reserve 0.82–0.94 for one-video-at-a-time merge/export.
            frac = 0.82 + 0.12 * (done / total)
            report(frac, f"Merging machine labels ({done}/{total}): {video_label}")

        report(0.82, "Merging machine labels CSV…")
        per_video = self.merge_csvs_for_skellyclicker(
            csv_folder_path=str(output_folder),
            output_path=str(csv_path),
            filtered=filter_videos,
            video_paths=video_paths,
            on_video_progress=on_merge_progress,
        )
        logger.info(
            "Wrote %d per-video machine CSV(s) next to source videos",
            len(per_video),
        )

        # Optional plots — must not block or undo CSV outputs on failure.
        try:
            report(0.95, "Plotting trajectories…")
            deeplabcut.plot_trajectories(
                config=self.project_config_path,
                videos=video_paths,
                filtered=filter_videos,
                destfolder=str(output_folder),
            )
        except Exception as exc:
            logger.exception("plot_trajectories failed after machine CSVs were written: %s", exc)
            report(0.95, f"Plotting skipped ({exc})")

        if annotate_videos:
            report(0.96, "Annotating videos…")
            self.annotate_videos(
                output_path=str(output_folder),
                csv_path=str(csv_path),
                video_paths=[Path(video) for video in video_paths]
            )
        else:
            print("Skipping video annotation")

        return str(csv_path)

    def partial_analyze_videos(
        self,
        human_labels_csv: str,
        video_paths: list[str],
        machine_labels_csv: str,
        progress_callback: Callable[[float | None, str], None] | None = None,
    ) -> str:
        """Re-infer frames listed in the human labels CSV and patch machine labels."""
        return partial_analyze_human_labels(
            config=str(self.project_config_path),
            human_labels_csv=human_labels_csv,
            video_paths=video_paths,
            machine_labels_csv=machine_labels_csv,
            progress_callback=progress_callback,
        )

    def merge_csvs_for_skellyclicker(
        self,
        csv_folder_path: str | Path,
        output_path: str | Path,
        filtered: bool = False,
        video_paths: list[str] | None = None,
        on_video_progress: Callable[[int, int, str], None] | None = None,
    ) -> list[Path]:
        """Stream-merge DLC CSVs; optionally write ``{stem}.csv`` beside each video."""
        return merge_dlc_csvs_for_skellyclicker(
            csv_folder_path,
            output_path,
            filtered=filtered,
            video_paths=video_paths,
            on_video_progress=on_video_progress,
        )

    def annotate_videos(self, output_path: str | Path, video_paths: list[Path], csv_path: str | Path):
        print(
            f"Annotating videos {video_paths}, saving to {output_path}"
        )
        args = [
            (output_path, csv_path, video) for video in video_paths
        ]
        with Pool(processes=len(video_paths)) as pool:
            pool.starmap(self.annotate_single_video, args)

    def annotate_single_video(self, output_path: str | Path, csv_path: str | Path, video: Path):
        data_handler = DataHandler.from_csv(csv_path)
        annotator_config = ImageAnnotatorConfig(
                marker_thickness=3,
                show_names=False,
                tracked_points=sorted(data_handler.tracked_points),
                show_clicks=False,
                show_help=False
            )
        image_annotator = ImageAnnotator(config=annotator_config)


        video_name = video.stem
        cap = cv2.VideoCapture(str(video))

        framerate = cap.get(cv2.CAP_PROP_FPS)
        framesize = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # need to deal with higher frame rates
        print(f"writing video to {str(Path(output_path) / video.name)}")

        video_writer_object = cv2.VideoWriter(
                str(Path(output_path) / video.name), fourcc, round(framerate, 2), framesize
            )

        frame_number = -1
        while True:
            ret, frame = cap.read()
            frame_number += 1
            if not ret:
                print(f"failed to read frame {frame_number}")
                break

            click_data = data_handler.get_data_by_video_name_and_frame(video_name=video_name, frame_number=frame_number)

            annotated_frame = image_annotator.annotate_single_image(image=frame, click_data=click_data)
            video_writer_object.write(annotated_frame)
        video_writer_object.release()
        cap.release()
