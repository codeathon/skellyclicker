import { useCallback, useEffect, useRef, useState } from "react";
import { AppSession, client } from "./api/client";
import { JobProgressBar, JobProgressState } from "./components/JobProgressBar";
import { LabelingCanvas } from "./components/LabelingCanvas";
import { LoadedAssets } from "./components/LoadedAssets";

function promptPath(label: string, defaultValue = ""): string | null {
  const v = window.prompt(label, defaultValue);
  return v?.trim() || null;
}

function promptPaths(label: string): string[] | null {
  const v = window.prompt(
    label + "\n(Enter comma-separated absolute paths)",
  );
  if (!v?.trim()) return null;
  return v.split(",").map((p) => p.trim()).filter(Boolean);
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
            <LabelingCanvas onClose={(updated) => setSession(updated)} />
          ) : (
            <section className="panel actions">
              <div className="action-group">
                <h3>Videos &amp; Labeling</h3>
                <button
                  onClick={() => {
                    const paths = promptPaths("Video file paths");
                    if (paths) run(() => client.setVideos(paths));
                  }}
                >
                  Select Videos
                </button>
                <button
                  disabled={!session.videos?.length}
                  onClick={() => run(client.openLabeler)}
                >
                  Open Labeler
                </button>
              </div>

              <div className="action-group">
                <h3>Labels</h3>
                <button
                  onClick={() => {
                    const p = promptPath("Human labels CSV");
                    if (p) run(() => client.setHumanLabels(p));
                  }}
                >
                  Import Human Labels
                </button>
                <button
                  onClick={() => {
                    const p = promptPath("Machine labels CSV");
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
              </div>

              <div className="action-group">
                <h3>DeepLabCut</h3>
                <button
                  onClick={() => {
                    const parent = promptPath("Parent directory for new project");
                    const name = promptPath("Project name");
                    if (parent && name)
                      run(() => client.createDlc(parent, name));
                  }}
                >
                  Create DLC Project
                </button>
                <button
                  onClick={() => {
                    const p = promptPath("DLC project directory");
                    if (p) run(() => client.loadDlc(p));
                  }}
                >
                  Load DLC Project
                </button>
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
                    const useTraining = window.confirm(
                      "Analyze training videos? OK = yes, Cancel = use custom list",
                    );
                    let paths = session.videos ?? [];
                    if (!useTraining) {
                      paths = promptPaths("Videos to analyze") ?? [];
                    }
                    try {
                      const { job_id } = await client.analyze(
                        paths,
                        useTraining,
                      );
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
                  onClick={() => {
                    const p = promptPath(
                      "Save session to",
                      session.session_saved_path ?? "",
                    );
                    if (p) run(() => client.saveSession(p));
                  }}
                >
                  Save Session
                </button>
                <button
                  onClick={() => {
                    const p = promptPath("Session JSON path");
                    if (p) run(() => client.loadSession(p));
                  }}
                >
                  Load Session
                </button>
              </div>

              {session.workflow_state === "review" && (
                <div className="action-group highlight">
                  <h3>Review Predictions</h3>
                  <p>
                    Machine labels loaded. Open the labeler to compare and
                    correct.
                  </p>
                  <button onClick={() => run(client.openLabeler)}>
                    Review in Labeler
                  </button>
                </div>
              )}
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
