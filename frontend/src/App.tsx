import { useCallback, useEffect, useState } from "react";
import { AppSession, client } from "./api/client";
import { LabelingCanvas } from "./components/LabelingCanvas";
import { LoadedAssets } from "./components/LoadedAssets";
import { WorkflowStepper } from "./components/WorkflowStepper";

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

export default function App() {
  const [session, setSession] = useState<AppSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobLog, setJobLog] = useState<string[]>([]);

  const refresh = useCallback(async () => {
    setSession(await client.getSession());
  }, []);

  useEffect(() => {
    refresh().catch((e) => setError(String(e)));
  }, [refresh]);

  const run = async (fn: () => Promise<AppSession>) => {
    try {
      setError(null);
      setSession(await fn());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const watchJob = (jobId: string) => {
    const ws = new WebSocket(
      `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/jobs/${jobId}`,
    );
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "log") setJobLog((prev) => [...prev, msg.message]);
      if (msg.type === "done") {
        refresh();
        ws.close();
      }
    };
    ws.onopen = () => ws.send("ping");
  };

  if (!session) return <div className="app">Loading…</div>;

  const labeling = session.workflow_state === "labeling";

  return (
    <div className="app">
      <header>
        <h1>SkellyClicker</h1>
        <p className="status">{session.status_message}</p>
      </header>

      <div className="layout">
        <aside>
          <WorkflowStepper session={session} />
          <LoadedAssets session={session} />
        </aside>

        <main>
          {error && <div className="error">{error}</div>}
          {jobLog.length > 0 && (
            <pre className="job-log">{jobLog.join("\n")}</pre>
          )}

          {labeling ? (
            <LabelingCanvas onClose={() => refresh()} />
          ) : (
            <section className="panel actions">
              <h2>Workflow</h2>

              <div className="action-group">
                <h3>Session</h3>
                <button onClick={() => run(client.newSession)}>Start New Session</button>
                <button onClick={() => {
                  const p = promptPath("Session JSON path");
                  if (p) run(() => client.loadSession(p));
                }}>Load Session</button>
                <button onClick={() => {
                  const p = promptPath("Save session to", session.session_saved_path ?? "");
                  if (p) run(() => client.saveSession(p));
                }}>Save Session</button>
                <button onClick={() => run(client.clearSession)}>Clear Session</button>
              </div>

              <div className="action-group">
                <h3>Videos &amp; Labeling</h3>
                <button onClick={() => {
                  const paths = promptPaths("Video file paths");
                  if (paths) run(() => client.setVideos(paths));
                }}>Select Videos</button>
                <button
                  disabled={!session.videos?.length}
                  onClick={() => run(client.openLabeler)}
                >
                  Open Labeler
                </button>
                <button onClick={() => {
                  const p = promptPath("Human labels CSV");
                  if (p) run(() => client.setHumanLabels(p));
                }}>Import Human Labels</button>
                <button onClick={() => {
                  const p = promptPath("Machine labels CSV");
                  if (p) run(() => client.setMachineLabels(p));
                }}>Import Machine Labels</button>
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
                <button onClick={() => {
                  const parent = promptPath("Parent directory for new project");
                  const name = promptPath("Project name");
                  if (parent && name) run(() => client.createDlc(parent, name));
                }}>Create DLC Project</button>
                <button onClick={() => {
                  const p = promptPath("DLC project directory");
                  if (p) run(() => client.loadDlc(p));
                }}>Load DLC Project</button>
                <button
                  disabled={!!session.active_job_id || !session.dlc_project_path}
                  onClick={async () => {
                    try {
                      const { job_id } = await client.train();
                      setJobLog([]);
                      watchJob(job_id);
                      await refresh();
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
                      const { job_id } = await client.analyze(paths, useTraining);
                      setJobLog([]);
                      watchJob(job_id);
                      await refresh();
                    } catch (e) {
                      setError(e instanceof Error ? e.message : String(e));
                    }
                  }}
                >
                  Analyze Videos
                </button>
              </div>

              {session.workflow_state === "review" && (
                <div className="action-group highlight">
                  <h3>Review Predictions</h3>
                  <p>Machine labels loaded. Open the labeler to compare and correct.</p>
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
