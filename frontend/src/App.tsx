import { useCallback, useEffect, useRef, useState } from "react";
import { AppSession, client } from "./api/client";
import { pathDialog } from "./api/pathDialog";
import { JobProgressBar, JobProgressState } from "./components/JobProgressBar";
import { LabelingCanvas } from "./components/LabelingCanvas";
import { LoadedAssets } from "./components/LoadedAssets";
import { NextStepBanner } from "./components/NextStepBanner";
import { WorkflowStepper } from "./components/WorkflowStepper";
import {
  canAnalyze,
  canOpenLabeler,
  canTrain,
  analyzeBlockReason,
  deriveWorkflowGuide,
  trainBlockReason,
  StepId,
} from "./workflow/workflowSteps";

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
  const guide = deriveWorkflowGuide(session);
  const focusStep = guide.currentStepId;

  const stepGroupClass = (steps: StepId[]) =>
    steps.includes(focusStep as StepId) ? "action-group highlight" : "action-group";

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

        <main className={labeling ? "main--labeling" : undefined}>
          {error && <div className="error">{error}</div>}
          {jobProgress && <JobProgressBar progress={jobProgress} />}
          {!labeling && (
            <>
              <WorkflowStepper guide={guide} />
              <NextStepBanner nextStep={guide.nextStep} />
            </>
          )}

          {labeling ? (
            <LabelingCanvas
              humanLabelsPath={session.human_labels_path}
              onClose={(updated) => setSession(updated)}
            />
          ) : (
            <section className="panel actions">
              <div className={stepGroupClass(["videos"])}>
                <h3>Videos</h3>
                {session.videos?.length ? (
                  <ul className="video-list">
                    {session.videos.map((p) => {
                      const name = p.split(/[/\\]/).pop() ?? p;
                      return (
                        <li key={p} className="video-list-item">
                          <span className="video-list-name" title={p}>
                            {name}
                          </span>
                          <button
                            type="button"
                            className="video-remove-btn"
                            aria-label={`Remove ${name}`}
                            disabled={!!session.active_job_id}
                            onClick={() => run(() => client.removeVideo(p))}
                          />
                        </li>
                      );
                    })}
                  </ul>
                ) : (
                  <p className="hint inline-hint">
                    Multi-camera: add each view (same folder for training).
                  </p>
                )}
                <button
                  disabled={!!session.active_job_id}
                  onClick={async () => {
                    const paths = await pathDialog.openVideos("Add videos");
                    if (paths) run(() => client.addVideos(paths));
                  }}
                >
                  Add Videos
                </button>
              </div>

              <div className={stepGroupClass(["dlc"])}>
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

              <div className={stepGroupClass(["label", "review"])}>
                <h3>Labels</h3>
                <p className="hint inline-hint">
                  Open labeler to create or edit frames using bodyparts from your
                  DLC project. After analyze, press m in the labeler to overlay
                  machine predictions.
                </p>
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
                    Add videos plus labels or a DLC project with bodyparts.
                  </p>
                )}
              </div>

              <details
                className="resume-section"
                open={guide.showResumeSection}
              >
                <summary>Resume / import</summary>
                <div className="resume-section-body">
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
                </div>
              </details>

              <div className={stepGroupClass(["train", "analyze"])}>
                <h3>Train &amp; Analyze</h3>
                <button
                  disabled={!canTrain(session)}
                  title={trainBlockReason(session) ?? undefined}
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
                {!canTrain(session) && trainBlockReason(session) && (
                  <p className="hint inline-hint">{trainBlockReason(session)}</p>
                )}
                <button
                  disabled={!canAnalyze(session)}
                  title={analyzeBlockReason(session) ?? undefined}
                  onClick={async () => {
                    const paths = session.videos ?? [];
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
                {!canAnalyze(session) && analyzeBlockReason(session) && (
                  <p className="hint inline-hint">{analyzeBlockReason(session)}</p>
                )}
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
