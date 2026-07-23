import { AppSession, client } from "../api/client";

type Props = {
  session: AppSession;
  onUpdate: (fn: () => Promise<AppSession>) => void;
};

/** Legacy Tk training spinboxes + analyze checkboxes for the web UI. */
export function DlcSettings({ session, onUpdate }: Props) {
  const jobRunning = Boolean(session.active_job_id);

  const commitTrainingField = (
    field: "epochs" | "save_epochs" | "batch_size",
    raw: string,
    current: number,
  ) => {
    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed) || parsed < 1 || parsed > 1000) {
      return;
    }
    if (parsed === current) return;
    onUpdate(() =>
      client.setTrainingSettings({
        epochs: field === "epochs" ? parsed : undefined,
        save_epochs: field === "save_epochs" ? parsed : undefined,
        batch_size: field === "batch_size" ? parsed : undefined,
      }),
    );
  };

  return (
    <div className="dlc-settings">
      <fieldset className="dlc-settings__group" disabled={jobRunning}>
        <legend>Training</legend>
        <div className="dlc-settings__row">
          <label>
            Epochs
            <input
              type="number"
              min={1}
              max={1000}
              defaultValue={session.training_epochs}
              key={`epochs-${session.training_epochs}`}
              onBlur={(e) =>
                commitTrainingField("epochs", e.target.value, session.training_epochs)
              }
            />
          </label>
          <label>
            Save epochs
            <input
              type="number"
              min={1}
              max={1000}
              defaultValue={session.training_save_epochs}
              key={`save-${session.training_save_epochs}`}
              onBlur={(e) =>
                commitTrainingField(
                  "save_epochs",
                  e.target.value,
                  session.training_save_epochs,
                )
              }
            />
          </label>
          <label>
            Batch size
            <input
              type="number"
              min={1}
              max={1000}
              defaultValue={session.training_batch_size}
              key={`batch-${session.training_batch_size}`}
              onBlur={(e) =>
                commitTrainingField(
                  "batch_size",
                  e.target.value,
                  session.training_batch_size,
                )
              }
            />
          </label>
        </div>
        {jobRunning && (
          <p className="hint inline-hint">Settings locked while a job is running.</p>
        )}
      </fieldset>

      <fieldset className="dlc-settings__group" disabled={jobRunning}>
        <legend>Analyze</legend>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={session.filter_predictions}
            onChange={(e) =>
              onUpdate(() =>
                client.setAnalyzeOptions({ filter_predictions: e.target.checked }),
              )
            }
          />
          Filter predictions
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={session.annotate_videos}
            onChange={(e) =>
              onUpdate(() =>
                client.setAnalyzeOptions({ annotate_videos: e.target.checked }),
              )
            }
          />
          Annotate videos
        </label>
        <p className="hint inline-hint">
          Annotated videos are slower to produce; use when you need visual QC.
        </p>
        <label>
          Parallel videos (0 = auto per GPU)
          <input
            type="number"
            min={0}
            max={8}
            defaultValue={session.analyze_parallel_workers}
            key={`parallel-${session.analyze_parallel_workers}`}
            onBlur={(e) => {
              const parsed = Number.parseInt(e.target.value, 10);
              if (!Number.isFinite(parsed) || parsed < 0 || parsed > 8) return;
              if (parsed === session.analyze_parallel_workers) return;
              onUpdate(() =>
                client.setAnalyzeOptions({ parallel_workers: parsed }),
              );
            }}
          />
        </label>
        <p className="hint inline-hint">
          With 2+ GPUs, analyzes one video per GPU (up to ~2x faster). On a single
          GPU, leave at auto — running multiple at once shares one card and rarely
          helps.
        </p>
      </fieldset>
    </div>
  );
}
