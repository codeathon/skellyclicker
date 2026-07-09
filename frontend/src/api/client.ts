export type WorkflowState =
  | "idle"
  | "needs_videos"
  | "ready_to_label"
  | "labeling"
  | "needs_project"
  | "ready_to_train"
  | "training"
  | "ready_to_analyze"
  | "analyzing"
  | "review";

export interface AssetPathCheck {
  kind: string;
  path: string;
  exists: boolean;
}

export type LabelingMode = "single" | "synced" | "corpus";

export interface AppSession {
  session_id: string;
  generation: number;
  workflow_state: WorkflowState;
  session_saved_path: string | null;
  videos: string[] | null;
  labeling_mode: LabelingMode;
  active_video_path: string | null;
  human_labels_path: string | null;
  machine_labels_path: string | null;
  dlc_project_path: string | null;
  dlc_iteration: number | null;
  tracked_point_names: string[];
  labeled_frame_count: number;
  frame_count: number;
  train_on_machine_labels: boolean;
  auto_save_session: boolean;
  labeling_session_id: string | null;
  active_job_id: string | null;
  status_message: string;
  training_epochs: number;
  training_save_epochs: number;
  training_batch_size: number;
  filter_predictions: boolean;
  annotate_videos: boolean;
  asset_path_checks: AssetPathCheck[];
}

export interface SessionVideoOption {
  path: string;
  name: string;
}

export type NavFrameKind = "human" | "machine" | "both";

export interface NavFrameItem {
  frame: number;
  kind: NavFrameKind;
}

export interface LabelingState {
  session_id: string;
  frame_number: number;
  frame_count: number;
  active_point: string;
  tracked_points: string[];
  point_colors: Record<string, [number, number, number]>;
  placed_points: string[];
  available_points: string[];
  labeled_frames: number;
  labeled_frame_list: number[];
  nav_frame_list?: NavFrameItem[];
  show_machine_labels: boolean;
  show_help: boolean;
  show_names: boolean;
  has_machine_labels: boolean;
  live_inference_ready?: boolean;
  auto_next_point: boolean;
  grid_width: number;
  grid_height: number;
  labeling_mode?: LabelingMode;
  session_videos?: SessionVideoOption[];
  active_video_path?: string | null;
  active_video_name?: string | null;
}

export type JobStatus = "pending" | "running" | "completed" | "failed";

export interface BackgroundJob {
  job_id: string;
  name: string;
  status: JobStatus;
  message: string;
  log_lines: string[];
  progress_percent: number | null;
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = err.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join("; ")
          : res.statusText;
    throw new Error(message || res.statusText);
  }
  return res.json();
}

async function dialogApi(body: object, path: string): Promise<{ paths: string[] }> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (res.status === 503) {
    throw new Error("DIALOG_UNAVAILABLE");
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      typeof err.detail === "string" ? err.detail : res.statusText,
    );
  }
  return res.json();
}

export const client = {
  getSession: () => api<AppSession>("/api/session"),
  newSession: () => api<AppSession>("/api/session/new", { method: "POST" }),
  saveSession: (path: string) =>
    api<AppSession>("/api/session/save", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
  loadSession: (path: string) =>
    api<AppSession>("/api/session/load", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
  setVideos: (paths: string[]) =>
    api<AppSession>("/api/videos", {
      method: "POST",
      body: JSON.stringify({ paths }),
    }),
  addVideos: (paths: string[]) =>
    api<AppSession>("/api/videos/add", {
      method: "POST",
      body: JSON.stringify({ paths }),
    }),
  removeVideo: (path: string) =>
    api<AppSession>("/api/videos/remove", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
  setHumanLabels: (path: string) =>
    api<AppSession>("/api/labels/human", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
  setMachineLabels: (path: string) =>
    api<AppSession>("/api/labels/machine", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
  setTrainingSettings: (settings: {
    epochs?: number;
    save_epochs?: number;
    batch_size?: number;
  }) =>
    api<AppSession>("/api/training/settings", {
      method: "POST",
      body: JSON.stringify(settings),
    }),
  setAnalyzeOptions: (options: {
    filter_predictions?: boolean;
    annotate_videos?: boolean;
  }) =>
    api<AppSession>("/api/analyze/options", {
      method: "POST",
      body: JSON.stringify(options),
    }),
  openLabeler: () => api<AppSession>("/api/labeling/open", { method: "POST" }),
  closeLabeler: (save: boolean, savePath?: string) =>
    api<AppSession>("/api/labeling/close", {
      method: "POST",
      body: JSON.stringify({ save, save_path: savePath ?? null }),
    }),
  saveLabeler: (savePath?: string) =>
    api<AppSession>("/api/labeling/save", {
      method: "POST",
      body: JSON.stringify({ save_path: savePath ?? null }),
    }),
  labelingState: () => api<LabelingState>("/api/labeling/state"),
  setFrame: (frame_number: number) =>
    api<LabelingState>("/api/labeling/frame", {
      method: "POST",
      body: JSON.stringify({ frame_number }),
    }),
  click: (x: number, y: number) =>
    api<LabelingState>("/api/labeling/click", {
      method: "POST",
      body: JSON.stringify({ x, y }),
    }),
  toggleMachineOverlay: () =>
    api<LabelingState>("/api/labeling/toggle-machine-overlay", { method: "POST" }),
  toggleHelp: () =>
    api<LabelingState>("/api/labeling/toggle-help", { method: "POST" }),
  toggleLabelNames: () =>
    api<LabelingState>("/api/labeling/toggle-names", { method: "POST" }),
  setActivePoint: (point_name: string) =>
    api<LabelingState>("/api/labeling/active-point", {
      method: "POST",
      body: JSON.stringify({ point_name }),
    }),
  setActiveLabelingVideo: (path: string) =>
    api<AppSession>("/api/labeling/active-video", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
  undoLabel: () =>
    api<LabelingState>("/api/labeling/undo", { method: "POST" }),
  loadDlc: (path: string) =>
    api<AppSession>("/api/dlc/load", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
  createDlc: (
    parent_directory: string,
    project_name: string,
    bodyparts?: string[],
  ) =>
    api<AppSession>("/api/dlc/create", {
      method: "POST",
      body: JSON.stringify({ parent_directory, project_name, bodyparts }),
    }),
  train: () => api<{ job_id: string }>("/api/dlc/train", { method: "POST" }),
  analyze: (video_paths: string[], use_training_videos: boolean) =>
    api<{ job_id: string }>("/api/dlc/analyze", {
      method: "POST",
      body: JSON.stringify({ video_paths, use_training_videos }),
    }),
  dialogOpenFile: (title: string, extensions: string[]) =>
    dialogApi({ title, extensions }, "/api/dialog/open-file"),
  dialogOpenFiles: (title: string, extensions: string[]) =>
    dialogApi({ title, extensions }, "/api/dialog/open-files"),
  dialogOpenDirectory: (title: string) =>
    dialogApi({ title, extensions: [] }, "/api/dialog/open-directory"),
  dialogSaveFile: (title: string, extensions: string[], default_name = "") =>
    dialogApi(
      { title, extensions, default_name },
      "/api/dialog/save-file",
    ),
  getJob: (job_id: string) => api<BackgroundJob>(`/api/jobs/${job_id}`),
  frameUrl: (n: number) => `/api/labeling/frame/${n}?t=${Date.now()}`,
  fetchFrameJpeg: async (
    frame_number: number,
    options?: { preview?: boolean; signal?: AbortSignal },
  ): Promise<Blob> => {
    const params = new URLSearchParams();
    if (options?.preview) params.set("preview", "1");
    params.set("t", String(Date.now()));
    const res = await fetch(
      `/api/labeling/frame/${frame_number}?${params}`,
      { signal: options?.signal },
    );
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(detail || `Failed to load frame ${frame_number}`);
    }
    const contentType = res.headers.get("content-type") ?? "";
    if (!contentType.includes("image/jpeg")) {
      throw new Error(`Unexpected frame response (${contentType || "no content-type"})`);
    }
    return res.blob();
  },
};
