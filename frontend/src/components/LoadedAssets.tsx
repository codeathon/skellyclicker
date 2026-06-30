import { AppSession } from "../api/client";

interface Props {
  session: AppSession;
}

function pathExists(session: AppSession, path: string | null | undefined): boolean | null {
  if (!path) return null;
  const check = session.asset_path_checks?.find((c) => c.path === path);
  return check ? check.exists : null;
}

function PathStatus({ exists }: { exists: boolean | null }) {
  if (exists === null) return null;
  return (
    <span
      className={`asset-status ${exists ? "asset-status--ok" : "asset-status--missing"}`}
      title={exists ? "Found on disk" : "Not found on disk"}
      aria-label={exists ? "Found on disk" : "Not found on disk"}
    >
      {exists ? "✓" : "✗"}
    </span>
  );
}

function Row({
  label,
  value,
  exists,
}: {
  label: string;
  value: string | null | undefined;
  exists?: boolean | null;
}) {
  return (
    <div className="asset-row">
      <div className="asset-row-header">
        {exists !== undefined && <PathStatus exists={exists ?? null} />}
        {label ? <span className="asset-label">{label}</span> : null}
      </div>
      <span className="asset-value">{value || "—"}</span>
    </div>
  );
}

export function LoadedAssets({ session }: Props) {
  const checks = session.asset_path_checks ?? [];
  const missingCount = checks.filter((c) => !c.exists).length;
  const hasChecks = checks.length > 0;

  return (
    <section className="panel assets">
      <h2>Loaded Assets</h2>
      {hasChecks && missingCount > 0 && (
        <p className="asset-path-warning">
          {missingCount} path{missingCount === 1 ? "" : "s"} from this session were not found on
          disk.
        </p>
      )}
      <Row
        label="Videos"
        value={
          session.videos?.length ? `${session.videos.length} file(s)` : null
        }
      />
      {session.videos?.map((p) => (
        <Row key={p} label="" value={p} exists={pathExists(session, p)} />
      ))}
      <Row
        label="Human labels"
        value={session.human_labels_path}
        exists={pathExists(session, session.human_labels_path)}
      />
      <Row
        label="Machine labels"
        value={session.machine_labels_path}
        exists={pathExists(session, session.machine_labels_path)}
      />
      <Row
        label="DLC project"
        value={session.dlc_project_path}
        exists={pathExists(session, session.dlc_project_path)}
      />
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
