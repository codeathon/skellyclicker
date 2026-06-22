import { useCallback, useEffect, useRef, useState } from "react";
import { AppSession, client } from "./api/client";
import { pathDialog } from "./api/pathDialog";
import { JobProgressBar, JobProgressState } from "./components/JobProgressBar";
import { LabelingCanvas } from "./components/LabelingCanvas";
import { LoadedAssets } from "./components/LoadedAssets";

function promptText(label: string, defaultValue = ""): string | null {
  const v = window.prompt(label, defaultValue);
  return v?.trim() || null;
}

function parseBodyparts(raw: string): string[] {
  return raw
    .split(/[,;\n]+/)
    .map((p) => p.trim())
    .filter(Boolean);
}

/** Open Labeler requires videos plus labels CSV or a DLC project with bodyparts. */
function canOpenLabeler(session: AppSession): boolean {
  if (!session.videos?.length) return false;
  if (session.human_labels_path || session.machine_labels_path) return true;
  return !!(session.dlc_project_path && session.tracked_point_names.length);
}

/** Shown in header — saved filename or unsaved placeholder. */
function sessionLabel(session: AppSession): string {
  if (session.session_saved_path) {
    const parts = session.session_saved_path.split(/[/\\]/);
    return parts[parts.length - 1] || session.session_saved_path;
  }
  return "Unsaved session";
}

export default function App() {
  const [session, setSession] = useState<AppSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState<JobProgressState | null>(null);
  const watchedJobRef = useRef<string | null>(null);
  const hideJobTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refresh = useCallback(async () => {
    setSession(await client.getSession());
  }, []);

  const clearJobProgressLater = useCallback((delayMs = 2500) => {
    if (hideJobTimerRef.current) clearTimeout(hideJobTimerRef.current);
    hideJobTimerRef.current = setTimeout(() => {
      setJobProgress(null);
      watchedJobRef.current = null;
    }, delayMs);
  }, []);

  const watchJob = useCallback(
    (jobId: string, jobName: string) => {
      if (watchedJobRef.current === jobId) return;
      watchedJobRef.current = jobId;

      const ws = new WebSocket(
        `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/jobs/${jobId}`,
      );

      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "progress") {
          setJobProgress({
            jobId,
            jobName,
            percent: msg.percent ?? null,
            message: msg.message ?? "",
            status: "running",
          });
        }
        if (msg.type === "done") {
          const failed = msg.status === "failed";
          setJobProgress({
            jobId,
            jobName,
            percent: failed ? msg.percent ?? null : 1,
            message: msg.message ?? (failed ? "Job failed" : "Complete"),
            status: failed ? "failed" : "completed",
          });
          refresh().then(() => clearJobProgressLater(failed ? 5000 : 2500));
          ws.close();
        }
      };

      ws.onerror = () => {
        setJobProgress((prev) =>
          prev
            ? { ...prev, message: "Lost connection to job stream", status: "failed" }
            : prev,
        );
      };
    },
    [refresh, clearJobProgressLater],
  );

  const startJob = useCallback(
    async (jobId: string, jobName: string) => {
      if (hideJobTimerRef.current) clearTimeout(hideJobTimerRef.current);
      setJobProgress({
        jobId,
        jobName,
        percent: null,
        message: "Starting…",
        status: "running",
      });
      watchJob(jobId, jobName);
      try {
        const job = await client.getJob(jobId);
        setJobProgress({
          jobId,
          jobName: job.name,
          percent: job.progress_percent,
          message: job.message || "Running…",
          status:
            job.status === "failed"
              ? "failed"
              : job.status === "completed"
                ? "completed"
                : "running",
        });
      } catch {
        /* WS will drive updates */
      }
      await refresh();
    },
    [watchJob, refresh],
  );

  // Fresh in-memory session every time the app is opened (no Start New button).
  useEffect(() => {
    client
      .newSession()
      .then(setSession)
      .catch((e) => setError(String(e)));
  }, []);

  // Reattach to a running job after page reload.
  useEffect(() => {
    if (!session?.active_job_id) return;
    const name =
      session.workflow_state === "training" ? "Train Network" : "Analyze Videos";
    startJob(session.active_job_id, name).catch((e) => setError(String(e)));
  }, [session?.active_job_id, session?.workflow_state, startJob]);

  const run = async (fn: () => Promise<AppSession>) => {
    try {
      setError(null);
      setSession(await fn());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  if (!session) return <div className="app">Loading…</div>;

  const labeling = session.workflow_state === "labeling";

  return (
    <div className="app">
      <header>
        <div className="header-left">
          <span className="session-label">{sessionLabel(session)}</span>
          <p className="status">{session.status_message}</p>
        </div>
        <h1>SkellyClicker</h1>
      </header>

      <div className="layout">
        <aside>
          <LoadedAssets session={session} />
        </aside>

        <main>
          {error && <div className="error">{error}</div>}
          {jobProgress && <JobProgressBar progress={jobProgress} />}

          {labeling ? (
            <LabelingCanvas
              humanLabelsPath={session.human_labels_path}
              onClose={(updated) => setSession(updated)}
            />
          ) : (
            <section className="panel actions">
              <div className="action-group">
                <h3>Videos</h3>
                {session.videos?.length ? (
                  <ul className="video-list">
                    {session.videos.map((p) => (
                      <li key={p}>{p.split(/[/\\]/).pop() ?? p}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="hint inline-hint">
                    Multi-camera: add each view (same folder for training).
                  </p>
                )}
                <button
                  onClick={async () => {
                    const paths = await pathDialog.openVideos(
                      session.videos?.length
                        ? "Replace all videos with"
                        : "Select videos",
                    );
                    if (paths) run(() => client.setVideos(paths));
                  }}
                >
                  {session.videos?.length ? "Replace Videos" : "Select Videos"}
                </button>
                {session.videos?.length ? (
                  <button
                    onClick={async () => {
                      const paths = await pathDialog.openVideos(
                        "Add videos to the current list",
                      );
                      if (paths) run(() => client.addVideos(paths));
                    }}
                  >
                    Add Videos
                  </button>
                ) : null}
              </div>

              <div className="action-group">
                <h3>DeepLabCut</h3>
                <button
                  onClick={async () => {
                    const parent = await pathDialog.openDirectory(
                      "Parent directory for new DLC project",
                    );
                    if (!parent) return;
                    const name = promptText("Project name");
                    if (!name) return;
                    let bodyparts = session.tracked_point_names;
                    if (!bodyparts.length) {
                      const raw = promptText(
                        "Bodyparts (comma-separated, e.g. nose,left_eye,right_eye)",
                      );
                      if (!raw) return;
                      bodyparts = parseBodyparts(raw);
                      if (!bodyparts.length) return;
                    }
                    run(() => client.createDlc(parent, name, bodyparts));
                  }}
                >
                  Create DLC Project
                </button>
                <button
                  onClick={async () => {
                    const p = await pathDialog.openDlcProject(
                      "DLC project folder",
                    );
                    if (p) run(() => client.loadDlc(p));
                  }}
                >
                  Load DLC Project
                </button>
              </div>

              <div className="action-group">
                <h3>Labels</h3>
                <p className="hint inline-hint">
                  Import labels to resume, or open labeler to create/edit using
                  bodyparts from your DLC project.
                </p>
                <button
                  onClick={async () => {
                    const p = await pathDialog.openCsv("Human labels CSV");
                    if (p) run(() => client.setHumanLabels(p));
                  }}
                >
                  Import Human Labels
                </button>
                <button
                  onClick={async () => {
                    const p = await pathDialog.openCsv("Machine labels CSV");
                    if (p) run(() => client.setMachineLabels(p));
                  }}
                >
                  Import Machine Labels
                </button>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={session.train_on_machine_labels}
                    disabled={!session.machine_labels_path}
                    onChange={(e) =>
                      run(() => client.setTrainOnMachine(e.target.checked))
                    }
                  />
                  Train on machine labels
                </label>
                <button
                  disabled={!canOpenLabeler(session)}
                  onClick={() => run(client.openLabeler)}
                >
                  Open Labeler
                </button>
                {!canOpenLabeler(session) && (
                  <p className="hint inline-hint">
                    Import Human or Machine labels, or load/create a DLC project
                    to define bodyparts.
                  </p>
                )}
              </div>

              <div className="action-group">
                <h3>Train &amp; Analyze</h3>
                <button
                  disabled={!!session.active_job_id || !session.dlc_project_path}
                  onClick={async () => {
                    try {
                      const { job_id } = await client.train();
                      await startJob(job_id, "Train Network");
                    } catch (e) {
                      setError(e instanceof Error ? e.message : String(e));
                    }
                  }}
                >
                  Train Network
                </button>
                <button
                  disabled={!!session.active_job_id || !session.dlc_project_path}
                  onClick={async () => {
                    const paths = session.videos ?? [];
                    if (!paths.length) {
                      setError("Select videos before analyzing.");
                      return;
                    }
                    try {
                      const { job_id } = await client.analyze(paths, true);
                      await startJob(job_id, "Analyze Videos");
                    } catch (e) {
                      setError(e instanceof Error ? e.message : String(e));
                    }
                  }}
                >
                  Analyze Videos
                </button>
              </div>

              <div className="session-actions">
                <button
                  onClick={async () => {
                    const defaultName = session.session_saved_path
                      ? session.session_saved_path.split(/[/\\]/).pop() ??
                        "session.json"
                      : "session.json";
                    const p = await pathDialog.saveSessionJson(defaultName);
                    if (p) run(() => client.saveSession(p));
                  }}
                >
                  Save Session
                </button>
                <button
                  onClick={async () => {
                    const p = await pathDialog.openSessionJson();
                    if (p) run(() => client.loadSession(p));
                  }}
                >
                  Load Session
                </button>
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
