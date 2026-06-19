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

export interface AppSession {
  session_id: string;
  generation: number;
  workflow_state: WorkflowState;
  session_saved_path: string | null;
  videos: string[] | null;
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
}

export interface LabelingState {
  session_id: string;
  frame_number: number;
  frame_count: number;
  active_point: string;
  tracked_points: string[];
  labeled_frames: number;
  show_machine_labels: boolean;
  auto_next_point: boolean;
  grid_width: number;
  grid_height: number;
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export const client = {
  getSession: () => api<AppSession>("/api/session"),
  newSession: () => api<AppSession>("/api/session/new", { method: "POST" }),
  clearSession: () => api<AppSession>("/api/session/clear", { method: "POST" }),
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
  setTrainOnMachine: (enabled: boolean) =>
    api<AppSession>("/api/labels/train-on-machine", {
      method: "POST",
      body: JSON.stringify({ enabled }),
    }),
  openLabeler: () => api<AppSession>("/api/labeling/open", { method: "POST" }),
  closeLabeler: (save: boolean, savePath?: string) =>
    api<AppSession>("/api/labeling/close", {
      method: "POST",
      body: JSON.stringify({ save, save_path: savePath ?? null }),
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
  loadDlc: (path: string) =>
    api<AppSession>("/api/dlc/load", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
  createDlc: (parent_directory: string, project_name: string) =>
    api<AppSession>("/api/dlc/create", {
      method: "POST",
      body: JSON.stringify({ parent_directory, project_name }),
    }),
  train: () => api<{ job_id: string }>("/api/dlc/train", { method: "POST" }),
  analyze: (video_paths: string[], use_training_videos: boolean) =>
    api<{ job_id: string }>("/api/dlc/analyze", {
      method: "POST",
      body: JSON.stringify({ video_paths, use_training_videos }),
    }),
  frameUrl: (n: number) => `/api/labeling/frame/${n}?t=${Date.now()}`,
};
