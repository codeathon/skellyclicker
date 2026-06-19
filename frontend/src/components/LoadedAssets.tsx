import { AppSession } from "../api/client";

interface Props {
  session: AppSession;
}

function Row({ label, value }: { label: string; value: string | null | undefined }) {
  return (
    <div className="asset-row">
      <span className="asset-label">{label}</span>
      <span className="asset-value">{value || "—"}</span>
    </div>
  );
}

export function LoadedAssets({ session }: Props) {
  const videoSummary = session.videos
    ? `${session.videos.length} file(s)`
    : null;

  return (
    <section className="panel assets">
      <h2>Loaded Assets</h2>
      <Row label="Videos" value={videoSummary} />
      <Row label="Human labels" value={session.human_labels_path} />
      <Row label="Machine labels" value={session.machine_labels_path} />
      <Row label="DLC project" value={session.dlc_project_path} />
      <Row
        label="Iteration"
        value={
          session.dlc_iteration != null ? String(session.dlc_iteration) : null
        }
      />
      <Row
        label="Bodyparts"
        value={
          session.tracked_point_names.length
            ? session.tracked_point_names.join(", ")
            : null
        }
      />
      <Row
        label="Labeled frames"
        value={
          session.frame_count
            ? `${session.labeled_frame_count} / ${session.frame_count}`
            : String(session.labeled_frame_count)
        }
      />
    </section>
  );
}
