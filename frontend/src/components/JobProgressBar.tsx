export type JobStatus = "pending" | "running" | "completed" | "failed";

export interface BackgroundJob {
  job_id: string;
  name: string;
  status: JobStatus;
  message: string;
  log_lines: string[];
  progress_percent: number | null;
}

export interface JobProgressState {
  jobId: string;
  jobName: string;
  percent: number | null;
  message: string;
  status: "running" | "completed" | "failed";
}

interface Props {
  progress: JobProgressState;
}

export function JobProgressBar({ progress }: Props) {
  const pct =
    progress.percent != null
      ? Math.round(Math.min(100, Math.max(0, progress.percent * 100)))
      : null;
  const indeterminate = progress.status === "running" && pct == null;

  return (
    <div
      className={`job-progress job-progress--${progress.status}`}
      role="status"
      aria-live="polite"
    >
      <div className="job-progress-header">
        <strong>{progress.jobName}</strong>
        {pct != null && <span className="job-progress-pct">{pct}%</span>}
      </div>
      <div className="job-progress-track">
        {indeterminate ? (
          <div className="job-progress-fill indeterminate" />
        ) : (
          <div
            className="job-progress-fill"
            style={{ width: `${pct ?? (progress.status === "completed" ? 100 : 0)}%` }}
          />
        )}
      </div>
      <p className="job-progress-message">{progress.message}</p>
    </div>
  );
}
