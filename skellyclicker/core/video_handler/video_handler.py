import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel

from skellyclicker import VideoPathString
from skellyclicker.core.human_labels_io import human_labels_csv_filename
from skellyclicker.core.click_data_handler.click_handler import ClickHandler
from skellyclicker.core.click_data_handler.data_handler import (
    DataHandler,
    DataHandlerConfig,
)
from skellyclicker.core.video_handler.image_annotator import (
    ImageAnnotator,
    ImageAnnotatorConfig,
)
from skellyclicker.core.video_handler.video_models import (
    VideoPlaybackState,
    GridParameters,
    VideoMetadata,
    VideoScalingParameters,
)

logger = logging.getLogger(__name__)
from copy import deepcopy


class VideoHandler(BaseModel):
    video_folder: str
    videos: dict[VideoPathString, VideoPlaybackState] = {}
    click_handler: ClickHandler
    data_handler: DataHandler
    grid_parameters: GridParameters
    preview_grid_parameters: GridParameters | None = None
    preview_scaling_params: list[VideoScalingParameters] | None = None
    image_annotator: ImageAnnotator = ImageAnnotator()
    frame_count: int
    show_machine_labels: bool = False
    machine_labels_path: str | None = None
    machine_labels_handler: DataHandler | None
    machine_labels_annotator: ImageAnnotator | None
    # Display-only live scrub predictions — never written to machine CSV.
    # Callable[[video_name, frame], dict[str, tuple[float, float]] | None]
    live_points_lookup: Any | None = None

    @classmethod
    def from_videos(
        cls,
        video_paths: list[str],
        max_window_size: tuple[int, int],
        data_handler_path: str | None = None,
        tracked_point_names: list[str] | None = None,
        machine_labels_path: str | None = None,
    ):
        video_paths = sorted(video_paths)
        for path in video_paths:
            if not Path(path).is_file():
                raise ValueError(f"File {path} does not exist.")
        videos, grid_parameters, frame_count = cls._load_videos(
            video_paths, max_window_size
        )
        return cls._assemble_handler(
            video_paths=video_paths,
            videos=videos,
            grid_parameters=grid_parameters,
            frame_count=frame_count,
            data_handler_path=data_handler_path,
            tracked_point_names=tracked_point_names,
            machine_labels_path=machine_labels_path,
        )

    @classmethod
    def from_videos_for_labeler(
        cls,
        video_paths: list[str],
        data_handler_path: str | None = None,
        tracked_point_names: list[str] | None = None,
        machine_labels_path: str | None = None,
    ):
        """Web labeler: native-resolution grid for clicks; capped grid for scrub preview."""
        from skellyclicker import PREVIEW_MAX_WINDOW_SIZE

        video_paths = sorted(video_paths)
        for path in video_paths:
            if not Path(path).is_file():
                raise ValueError(f"File {path} does not exist.")

        videos, frame_count = cls._open_videos(video_paths)
        grid_parameters = GridParameters.calculate_native(videos)
        cls._apply_grid_scaling(videos, grid_parameters)

        preview_grid = GridParameters.calculate(videos, PREVIEW_MAX_WINDOW_SIZE)
        preview_scaling = [
            cls._calculate_scaling_parameters(
                video.metadata.width, video.metadata.height, preview_grid.cell_size
            )
            for video in videos.values()
        ]

        return cls._assemble_handler(
            video_paths=video_paths,
            videos=videos,
            grid_parameters=grid_parameters,
            frame_count=frame_count,
            data_handler_path=data_handler_path,
            tracked_point_names=tracked_point_names,
            machine_labels_path=machine_labels_path,
            preview_grid=preview_grid,
            preview_scaling=preview_scaling,
        )

    @classmethod
    def _assemble_handler(
        cls,
        *,
        video_paths: list[str],
        videos: dict[VideoPathString, VideoPlaybackState],
        grid_parameters: GridParameters,
        frame_count: int,
        data_handler_path: str | None,
        tracked_point_names: list[str] | None,
        machine_labels_path: str | None,
        preview_grid: GridParameters | None = None,
        preview_scaling: list[VideoScalingParameters] | None = None,
    ) -> "VideoHandler":
        if data_handler_path and Path(data_handler_path).suffix == ".json":
            data_handler = DataHandler.from_config(
                DataHandlerConfig.from_config_file(
                    videos=videos, config_path=data_handler_path
                )
            )
        elif data_handler_path and Path(data_handler_path).suffix == ".csv":
            resolved_video_names = sorted(v.name for v in videos.values())
            data_handler = DataHandler.from_csv(
                data_handler_path,
                video_names=resolved_video_names,
                num_frames=frame_count,
                tracked_point_names=tracked_point_names,
            )
        elif tracked_point_names:
            data_handler = DataHandler.from_config(
                DataHandlerConfig(
                    num_frames=frame_count,
                    video_names=sorted(v.name for v in videos.values()),
                    tracked_point_names=tracked_point_names,
                )
            )
        else:
            raise ValueError(
                "Provide a labels CSV/JSON path or tracked_point_names for bodyparts"
            )

        if machine_labels_path:
            machine_labels_handler = None
            machine_labels_annotator = None
        else:
            machine_labels_path = None
            machine_labels_handler = None
            machine_labels_annotator = None

        image_annotator = ImageAnnotator(
            config=ImageAnnotatorConfig(
                tracked_points=data_handler.config.tracked_point_names,
            )
        )

        return cls(
            video_folder=str(Path(list(videos.keys())[0]).parent),
            videos=videos,
            click_handler=ClickHandler(
                output_path=str(Path(video_paths[0]).parent / "clicks.csv"),
                grid_helper=grid_parameters,
                videos=list(videos.values()),
            ),
            data_handler=data_handler,
            grid_parameters=grid_parameters,
            preview_grid_parameters=preview_grid,
            preview_scaling_params=preview_scaling,
            frame_count=frame_count,
            image_annotator=image_annotator,
            show_machine_labels=False,
            machine_labels_path=machine_labels_path,
            machine_labels_handler=machine_labels_handler,
            machine_labels_annotator=machine_labels_annotator,
        )

    def ensure_machine_labels_loaded(self) -> None:
        """Parse machine-label CSV on first overlay use (skipped at labeler open)."""
        if self.machine_labels_handler is not None or not self.machine_labels_path:
            return
        video_names = sorted(v.name for v in self.videos.values())
        handler = DataHandler.from_csv_overlay(
            self.machine_labels_path,
            video_names=video_names,
            num_frames=self.frame_count,
            tracked_point_names=self.data_handler.config.tracked_point_names,
        )
        self.machine_labels_handler = handler
        self._ensure_machine_annotator()

    def _ensure_machine_annotator(self) -> None:
        if self.machine_labels_annotator is not None:
            return
        self.machine_labels_annotator = ImageAnnotator(
            config=ImageAnnotatorConfig(
                marker_type=cv2.MARKER_CROSS,
                marker_size=10,
                marker_thickness=1,
                tracked_points=self.data_handler.config.tracked_point_names,
                show_clicks=False,
            )
        )

    def _machine_click_data_for_frame(
        self, video_index: int, frame_number: int
    ) -> dict:
        """CSV machine labels for this frame, else display-only live cache (never saved)."""
        from skellyclicker.core.video_handler.video_models import ClickData

        if self.machine_labels_path:
            self.ensure_machine_labels_loaded()
        if self.machine_labels_handler is not None:
            csv_data = self.machine_labels_handler.get_data_by_video_frame(
                video_index=video_index, frame_number=frame_number
            )
            if csv_data:
                return csv_data
        # Preview-only live predictions — in-memory cache, not the machine CSV.
        lookup = self.live_points_lookup
        if lookup is None:
            return {}
        video_name = sorted(v.name for v in self.videos.values())[video_index]
        points = lookup(video_name, frame_number)
        if not points:
            return {}
        return {
            name: ClickData(
                video_index=video_index,
                frame_number=frame_number,
                video_x=int(x),
                video_y=int(y),
                window_x=int(x),
                window_y=int(y),
            )
            for name, (x, y) in points.items()
        }

    @classmethod
    def _open_videos(
        cls, video_paths: list[str]
    ) -> tuple[dict[VideoPathString, VideoPlaybackState], int]:
        """Open VideoCapture handles and validate equal frame counts."""
        videos: dict[VideoPathString, VideoPlaybackState] = {}
        image_counts: set[int] = set()

        for video_path in video_paths:
            video_name = Path(video_path).name
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            metadata = VideoMetadata(
                path=video_path,
                name=video_name,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            )
            image_counts.add(metadata.frame_count)
            videos[video_path] = VideoPlaybackState(
                metadata=metadata, cap=cap, scaling_params=None
            )

        if len(image_counts) > 1:
            raise ValueError("All videos must have the same number of images")

        return videos, image_counts.pop()

    @classmethod
    def _apply_grid_scaling(
        cls,
        videos: dict[VideoPathString, VideoPlaybackState],
        grid_parameters: GridParameters,
    ) -> None:
        for video in videos.values():
            video.scaling_params = cls._calculate_scaling_parameters(
                video.metadata.width,
                video.metadata.height,
                grid_parameters.cell_size,
            )

    @classmethod
    def _load_videos(
        cls, video_paths: list[str], max_window_size: tuple[int, int]
    ) -> tuple[dict[VideoPathString, VideoPlaybackState], GridParameters, int]:
        """Load videos and fit them into a capped window grid."""
        videos, frame_count = cls._open_videos(video_paths)
        grid_parameters = GridParameters.calculate(
            videos=videos, max_window_size=max_window_size
        )
        cls._apply_grid_scaling(videos, grid_parameters)
        return videos, grid_parameters, frame_count

    @staticmethod
    def _calculate_scaling_parameters(
        orig_width: int, orig_height: int, cell_size: tuple[int, int]
    ) -> VideoScalingParameters:
        """Calculate scaling parameters for a video to fit in a grid cell."""
        cell_width, cell_height = cell_size

        # Calculate scale factor preserving aspect ratio
        scale = min(cell_width / orig_width, cell_height / orig_height)

        scaled_width = int(orig_width * scale)
        scaled_height = int(orig_height * scale)

        # Calculate offsets to center the video
        x_offset = (cell_width - scaled_width) // 2
        y_offset = (cell_height - scaled_height) // 2

        return VideoScalingParameters(
            scale=scale,
            x_offset=x_offset,
            y_offset=y_offset,
            scaled_width=scaled_width,
            scaled_height=scaled_height,
            original_width=orig_width,
            original_height=orig_height,
        )

    def prepare_single_image(
        self,
        image: np.ndarray,
        frame_number: int,
        scaling_params: VideoScalingParameters,
    ) -> np.ndarray:
        """Process a video image - resize and add overlays."""
        if image is None:
            return np.zeros(self.grid_parameters.cell_size + (3,), dtype=np.uint8)

        # Resize image
        resized = cv2.resize(
            image, (scaling_params.scaled_width, scaling_params.scaled_height)
        )

        # Create padded image
        padded = np.zeros(
            (self.grid_parameters.cell_height, self.grid_parameters.cell_width, 3),
            dtype=np.uint8,
        )
        padded[
            scaling_params.y_offset : scaling_params.y_offset
            + scaling_params.scaled_height,
            scaling_params.x_offset : scaling_params.x_offset
            + scaling_params.scaled_width,
        ] = resized

        # Add frame number
        cv2.putText(
            padded,
            f"Frame {frame_number}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return padded

    def handle_clicks(
        self, x: int, y: int, frame_number: int, auto_next_point: bool = False
    ):
        click_data = self.click_handler.process_click(x, y, frame_number)
        if click_data is None:
            return
        self.data_handler.update_dataframe(click_data)

        if auto_next_point:
            self.data_handler.move_active_point_by_index(index_change=1)

    def move_active_point_by_index(self, index_change: int):
        self.data_handler.move_active_point_by_index(index_change=index_change)

    def copy_frame_data_from_machine_labels(
        self, frame_number: int, video_index: int
    ) -> None:
        self.ensure_machine_labels_loaded()
        if self.machine_labels_handler is not None:
            machine_labels_data = self.machine_labels_handler.get_data_by_video_frame(
                    video_index=video_index, frame_number=frame_number
                )
            for name, click_data in machine_labels_data.items():
                try:
                    self.data_handler.update_dataframe(
                        click_data=click_data,
                        point_name=name,
                    )
                except (ValueError, KeyError) as e:
                    logger.error(f"Error updating data with point name {name}: {e}")

    def create_grid_image(
        self,
        frame_number: int,
        annotate_images: bool = True,
        *,
        preview: bool = False,
    ) -> np.ndarray:
        """Create a grid of video images."""
        use_preview = (
            preview
            and self.preview_grid_parameters is not None
            and self.preview_scaling_params is not None
        )
        grid = (
            self.preview_grid_parameters if use_preview else self.grid_parameters
        )
        video_states = [deepcopy(video.zoom_state) for video in self.videos.values()]

        grid_image = np.zeros(
            (grid.total_height, grid.total_width, 3),
            dtype=np.uint8,
        )

        for video_index, (video, zoom_state) in enumerate(
            zip(self.videos.values(), video_states)
        ):
            scaling = (
                self.preview_scaling_params[video_index]
                if use_preview
                else video.scaling_params
            )
            row = video_index // grid.columns
            col = video_index % grid.columns

            video.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, image = video.cap.read()

            if success:
                image = cv2.convertScaleAbs(image, alpha=video.contrast, beta=video.brightness)
                if annotate_images:
                    image = self.image_annotator.annotate_single_image(
                        image,
                        active_point=self.data_handler.active_point,
                        click_data=self.data_handler.get_data_by_video_frame(
                            video_index=video_index, frame_number=frame_number
                        ),
                    )
                if self.show_machine_labels:
                    click_data = self._machine_click_data_for_frame(
                        video_index, frame_number
                    )
                    if click_data:
                        self._ensure_machine_annotator()
                        image = self.machine_labels_annotator.annotate_single_image(
                            image,
                            click_data=click_data,
                        )

                if zoom_state.scale > 1.0:
                    zoomed_width = int(scaling.scaled_width * zoom_state.scale)
                    zoomed_height = int(scaling.scaled_height * zoom_state.scale)
                    zoomed = cv2.resize(image, (zoomed_width, zoomed_height))

                    relative_x = (
                        zoom_state.center_x - scaling.x_offset
                    ) / scaling.scaled_width
                    relative_y = (
                        zoom_state.center_y - scaling.y_offset
                    ) / scaling.scaled_height
                    center_x = int(relative_x * zoomed_width)
                    center_y = int(relative_y * zoomed_height)

                    x1 = max(0, center_x - scaling.scaled_width // 2)
                    y1 = max(0, center_y - scaling.scaled_height // 2)
                    x2 = min(zoomed_width, x1 + scaling.scaled_width)
                    y2 = min(zoomed_height, y1 + scaling.scaled_height)

                    if x2 == zoomed_width:
                        x1 = zoomed_width - scaling.scaled_width
                    if y2 == zoomed_height:
                        y1 = zoomed_height - scaling.scaled_height

                    scaled_image = zoomed[y1:y2, x1:x2]

                else:
                    scaled_image = cv2.resize(
                        image,
                        (scaling.scaled_width, scaling.scaled_height),
                    )

                y_start = row * grid.cell_height + scaling.y_offset
                x_start = col * grid.cell_width + scaling.x_offset

                try:
                    grid_image[
                        y_start : y_start + scaled_image.shape[0],
                        x_start : x_start + scaled_image.shape[1],
                    ] = scaled_image
                except ValueError as e:
                    logger.error(f"Error placing image in grid: {e}")

        if use_preview:
            return grid_image

        return self.image_annotator.annotate_image_grid(
            image=grid_image,
            active_point=self.data_handler.active_point,
            frame_number=frame_number,
        )

    def close(
        self, save_data: bool | None = None, save_path: str | None = None
    ) -> str | None:
        """Clean up resources."""
        logger.info("VideoHandler closing")
        saved_path: str | None = None
        if save_data is True:
            saved_path = self._save_data(save_pathstring=save_path)
        elif save_data is None:
            while True:
                save_data_input = input("Save data? (yes/no): ")
                if save_data_input.lower() == "yes" or save_data_input.lower() == "y":
                    saved_path = self._save_data(save_pathstring=save_path)
                    break
                else:
                    confirmation = input(
                        "Confirm your choice: Type 'yes' to prevent data loss or 'no' to delete this session forever (yes/no): "
                    )
                    if confirmation == "no" or confirmation == "n":
                        logger.info("Data not saved.")
                        saved_path = None
                        break
        for video in self.videos.values():
            video.cap.release()

        return saved_path

    def save_labels(self, save_path: str | None = None) -> str:
        """Persist human labels without releasing video captures."""
        return self._save_data(save_pathstring=save_path)

    def _save_data(self, save_pathstring: str | None = None) -> str:
        if save_pathstring is None:
            # Default next to the loaded videos so labels are easy to find.
            save_path = Path(self.video_folder) / "skellyclicker_data"
            save_path.mkdir(exist_ok=True, parents=True)
        else:
            save_path = Path(save_pathstring)

        if save_path.is_dir():
            save_path.mkdir(exist_ok=True, parents=True)
            video_names = sorted(v.name for v in self.videos.values())
            csv_path = save_path / human_labels_csv_filename(video_names)
        else:
            csv_path = save_path

        self.data_handler.save_csv(output_path=str(csv_path))

        return str(csv_path)
