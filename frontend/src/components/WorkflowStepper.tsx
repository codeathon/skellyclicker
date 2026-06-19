import { AppSession, WorkflowState } from "../api/client";

const STEPS: { state: WorkflowState; label: string }[] = [
  { state: "idle", label: "Session" },
  { state: "needs_videos", label: "Videos" },
  { state: "ready_to_label", label: "Label" },
  { state: "needs_project", label: "DLC Project" },
  { state: "ready_to_train", label: "Train" },
  { state: "ready_to_analyze", label: "Analyze" },
  { state: "review", label: "Review" },
];

const ORDER: WorkflowState[] = STEPS.map((s) => s.state);

function stepIndex(state: WorkflowState): number {
  if (state === "labeling") return ORDER.indexOf("ready_to_label");
  if (state === "training") return ORDER.indexOf("ready_to_train");
  if (state === "analyzing") return ORDER.indexOf("ready_to_analyze");
  const idx = ORDER.indexOf(state);
  return idx >= 0 ? idx : 0;
}

interface Props {
  session: AppSession;
}

export function WorkflowStepper({ session }: Props) {
  const current = stepIndex(session.workflow_state);

  return (
    <nav className="stepper">
      {STEPS.map((step, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <div
            key={step.state}
            className={`step ${done ? "done" : ""} ${active ? "active" : ""}`}
          >
            <span className="step-num">{done ? "✓" : i + 1}</span>
            <span>{step.label}</span>
          </div>
        );
      })}
    </nav>
  );
}
