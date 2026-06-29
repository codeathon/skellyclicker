"""FastAPI routes for session, labeling, and DLC workflow."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from skellyclicker.services.models import AppSession, WorkflowState
from skellyclicker.services.errors import SessionConflictError, SessionError
from skellyclicker.services.session_store import store

# Repo root is two levels above skellyclicker/api/app.py
REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"

_logger = logging.getLogger("skellyclicker.dialog")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
	from skellyclicker.services.native_dialog import dialog_startup_warning

	warning = dialog_startup_warning()
	if warning:
		_logger.warning(warning)
	yield


app = FastAPI(title="SkellyClicker", version="0.2.0", lifespan=_lifespan)


@app.exception_handler(SessionError)
async def session_error_handler(_request, exc: SessionError):
	status = 409 if isinstance(exc, SessionConflictError) else 400
	return JSONResponse(status_code=status, content={"detail": exc.message})


app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["*"],
	allow_headers=["*"],
)


class PathListBody(BaseModel):
	paths: list[str]


class PathBody(BaseModel):
	path: str


class CreateProjectBody(BaseModel):
	parent_directory: str
	project_name: str
	bodyparts: list[str] | None = None


class CloseLabelerBody(BaseModel):
	save: bool
	save_path: str | None = None


class ClickBody(BaseModel):
	x: int
	y: int


class FrameBody(BaseModel):
	frame_number: int


class AnalyzeBody(BaseModel):
	video_paths: list[str]
	use_training_videos: bool = True


class ToggleBody(BaseModel):
	enabled: bool


class TrainingSettingsBody(BaseModel):
	epochs: int | None = None
	save_epochs: int | None = None
	batch_size: int | None = None


class AnalyzeOptionsBody(BaseModel):
	filter_predictions: bool | None = None
	annotate_videos: bool | None = None


class DialogBody(BaseModel):
	title: str = "Select"
	extensions: list[str] = []


class SaveDialogBody(BaseModel):
	title: str = "Save"
	extensions: list[str] = []
	default_name: str = ""


def _dialog_http(handler, body: DialogBody):
	from skellyclicker.services.dialog_errors import DialogCancelled, DialogUnavailable

	try:
		return handler()
	except DialogCancelled:
		return {"paths": []}
	except DialogUnavailable as exc:
		raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/dialog/open-file")
def dialog_open_file(body: DialogBody):
	from skellyclicker.services.native_dialog import pick_file

	def run():
		return {"paths": [pick_file(body.title, body.extensions or ["*"])]}

	return _dialog_http(run, body)


@app.post("/api/dialog/open-files")
def dialog_open_files(body: DialogBody):
	from skellyclicker.services.native_dialog import pick_files

	def run():
		return {"paths": pick_files(body.title, body.extensions or ["*"])}

	return _dialog_http(run, body)


@app.post("/api/dialog/open-directory")
def dialog_open_directory(body: DialogBody):
	from skellyclicker.services.native_dialog import pick_directory

	def run():
		return {"paths": [pick_directory(body.title)]}

	return _dialog_http(run, body)


@app.post("/api/dialog/save-file")
def dialog_save_file(body: SaveDialogBody):
	from skellyclicker.services.native_dialog import save_file

	def run():
		return {"paths": [save_file(body.title, body.extensions or ["*"], body.default_name)]}

	return _dialog_http(run, body)


@app.get("/api/dialog/status")
def dialog_status():
	from skellyclicker.services.native_dialog import check_dialog_availability

	available, detail = check_dialog_availability()
	return {"available": available, "detail": detail}


@app.post("/api/upload/files")
async def upload_files(files: list[UploadFile] = File(...)):
	"""Store browser-selected files on the server; return absolute paths."""
	from skellyclicker.services.upload_store import save_upload

	if not files:
		raise HTTPException(status_code=400, detail="No files uploaded")

	session = store.get_session()
	paths: list[str] = []
	for upload in files:
		content = await upload.read()
		paths.append(save_upload(session.session_id, upload.filename or "upload", content))
	return {"paths": paths}


@app.get("/api/session", response_model=AppSession)
def get_session() -> AppSession:
	return store.get_session()


@app.get("/api/workflow/hints")
def workflow_hints():
	from skellyclicker.services.workflow import build_workflow_hints

	return build_workflow_hints(store.get_session())


@app.post("/api/session/new", response_model=AppSession)
def new_session() -> AppSession:
	return store.start_new_session()


@app.post("/api/session/clear", response_model=AppSession)
def clear_session() -> AppSession:
	return store.clear_session()


@app.post("/api/session/save", response_model=AppSession)
def save_session(body: PathBody) -> AppSession:
	return store.save_session_json(body.path)


@app.post("/api/session/load", response_model=AppSession)
def load_session(body: PathBody) -> AppSession:
	return store.load_session_json(body.path)


@app.post("/api/videos", response_model=AppSession)
def set_videos(body: PathListBody) -> AppSession:
	return store.set_videos(body.paths)


@app.post("/api/videos/add", response_model=AppSession)
def add_videos(body: PathListBody) -> AppSession:
	return store.add_videos(body.paths)


@app.post("/api/videos/remove", response_model=AppSession)
def remove_video(body: PathBody) -> AppSession:
	return store.remove_video(body.path)


@app.post("/api/labels/human", response_model=AppSession)
def set_human_labels(body: PathBody) -> AppSession:
	return store.set_human_labels_path(body.path)


@app.post("/api/labels/machine", response_model=AppSession)
def set_machine_labels(body: PathBody) -> AppSession:
	return store.set_machine_labels_path(body.path)


@app.post("/api/labels/train-on-machine", response_model=AppSession)
def set_train_on_machine(body: ToggleBody) -> AppSession:
	if body.enabled and not store.session.machine_labels_path:
		raise HTTPException(
			status_code=400,
			detail="Load machine labels CSV before enabling train-on-machine mode",
		)
	store.session.train_on_machine_labels = body.enabled
	return store.get_session()


@app.post("/api/training/settings", response_model=AppSession)
def set_training_settings(body: TrainingSettingsBody) -> AppSession:
	if body.epochs is None and body.save_epochs is None and body.batch_size is None:
		raise HTTPException(status_code=400, detail="Provide at least one training setting")
	return store.set_training_settings(
		epochs=body.epochs,
		save_epochs=body.save_epochs,
		batch_size=body.batch_size,
	)


@app.post("/api/analyze/options", response_model=AppSession)
def set_analyze_options(body: AnalyzeOptionsBody) -> AppSession:
	if body.filter_predictions is None and body.annotate_videos is None:
		raise HTTPException(status_code=400, detail="Provide at least one analyze option")
	return store.set_analyze_options(
		filter_predictions=body.filter_predictions,
		annotate_videos=body.annotate_videos,
	)


@app.post("/api/dlc/load", response_model=AppSession)
def load_dlc(body: PathBody) -> AppSession:
	return store.load_dlc_project(body.path)


@app.post("/api/dlc/create", response_model=AppSession)
def create_dlc(body: CreateProjectBody) -> AppSession:
	from skellyclicker.core.deeplabcut_handler.deeplabcut_handler import (
		DeeplabcutHandler,
	)

	bodyparts = body.bodyparts or store.session.tracked_point_names
	if not bodyparts:
		raise HTTPException(
			status_code=400,
			detail="Provide bodyparts when creating a DLC project, or import labels first.",
		)
	store._assert_no_active_job()
	handler = DeeplabcutHandler.create_deeplabcut_project(
		project_name=body.project_name,
		project_parent_directory=body.parent_directory,
		tracked_point_names=bodyparts,
	)
	store.dlc_handler = handler
	full_path = str(Path(body.parent_directory) / body.project_name)
	store.session.dlc_project_path = full_path
	store.session.dlc_iteration = handler.iteration
	store.session.tracked_point_names = bodyparts
	store.session.workflow_state = WorkflowState.ready_to_train
	return store.get_session()


@app.post("/api/labeling/open", response_model=AppSession)
def open_labeler() -> AppSession:
	return store.open_labeler()


@app.post("/api/labeling/close", response_model=AppSession)
def close_labeler(body: CloseLabelerBody) -> AppSession:
	return store.close_labeler(save=body.save, save_path=body.save_path)


@app.get("/api/labeling/state")
def labeling_state():
	if not store.labeling_engine:
		raise HTTPException(status_code=400, detail="Labeler is not open")
	return store.labeling_engine.state_dict()


@app.get("/api/labeling/frame/{frame_number}")
def labeling_frame(frame_number: int, preview: bool = False) -> Response:
	if not store.labeling_engine:
		raise HTTPException(status_code=400, detail="Labeler is not open")
	jpeg = store.labeling_engine.render_frame_jpeg(frame_number, preview=preview)
	return Response(content=jpeg, media_type="image/jpeg")


@app.post("/api/labeling/click")
def labeling_click(body: ClickBody):
	if not store.labeling_engine:
		raise HTTPException(status_code=400, detail="Labeler is not open")
	store.labeling_engine.handle_click(body.x, body.y)
	return store.labeling_engine.state_dict()


@app.post("/api/labeling/frame")
def set_frame(body: FrameBody):
	if not store.labeling_engine:
		raise HTTPException(status_code=400, detail="Labeler is not open")
	store.labeling_engine.frame_number = body.frame_number
	store.labeling_engine.sync_active_point()
	return store.labeling_engine.state_dict()


@app.post("/api/labeling/toggle-machine-overlay")
def toggle_machine_overlay():
	if not store.labeling_engine:
		raise HTTPException(status_code=400, detail="Labeler is not open")
	eng = store.labeling_engine
	eng.show_machine_labels = not eng.show_machine_labels
	return eng.state_dict()


@app.post("/api/labeling/toggle-help")
def toggle_help_overlay():
	if not store.labeling_engine:
		raise HTTPException(status_code=400, detail="Labeler is not open")
	eng = store.labeling_engine
	eng.show_help = not eng.show_help
	return eng.state_dict()


@app.post("/api/dlc/train")
def train_network():
	try:
		job = store.job_runner.start_train()
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	return {"job_id": job.job_id}


@app.post("/api/dlc/analyze")
def analyze_videos(body: AnalyzeBody):
	try:
		paths = body.video_paths if not body.use_training_videos else (store.session.videos or [])
		job = store.job_runner.start_analyze(paths, body.use_training_videos)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	return {"job_id": job.job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
	job = store.job_runner.get_job(job_id)
	if not job:
		raise HTTPException(status_code=404, detail="Job not found")
	return job


@app.websocket("/ws/jobs/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str):
	await websocket.accept()
	last_len = 0
	last_progress: float | None = -1.0
	last_message = ""
	try:
		while True:
			job = store.job_runner.get_job(job_id)
			if job and len(job.log_lines) > last_len:
				for line in job.log_lines[last_len:]:
					await websocket.send_json({"type": "log", "message": line})
				last_len = len(job.log_lines)
			if job and (
				job.progress_percent != last_progress or job.message != last_message
			):
				await websocket.send_json(
					{
						"type": "progress",
						"percent": job.progress_percent,
						"message": job.message,
					}
				)
				last_progress = job.progress_percent
				last_message = job.message
			if job and job.status.value in ("completed", "failed"):
				await websocket.send_json(
					{
						"type": "done",
						"status": job.status.value,
						"percent": job.progress_percent,
						"message": job.message,
					}
				)
				break
			await asyncio.sleep(0.15)
	except WebSocketDisconnect:
		pass


if FRONTEND_DIST.is_dir():
	app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static")
else:
	@app.get("/")
	def frontend_not_built():
		raise HTTPException(
			status_code=503,
			detail=(
				f"Frontend not built. Run: cd {REPO_ROOT / 'frontend'} && npm install && npm run build"
			),
		)
