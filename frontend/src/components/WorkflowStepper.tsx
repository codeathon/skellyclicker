import { WorkflowGuide } from "../workflow/workflowSteps";

interface Props {
	guide: WorkflowGuide;
}

export function WorkflowStepper({ guide }: Props) {
	return (
		<div className="workflow-stepper-wrap">
			<div className="workflow-phase">
				<span className="workflow-phase-label">{guide.phaseLabel}</span>
			</div>
			<ol className="workflow-stepper" aria-label="Workflow progress">
				{guide.steps.map((step) => (
					<li
						key={step.id}
						className={`workflow-step workflow-step--${step.status}`}
						aria-current={step.status === "current" ? "step" : undefined}
					>
						<span className="workflow-step-marker" />
						<span className="workflow-step-label">{step.label}</span>
					</li>
				))}
			</ol>
		</div>
	);
}
