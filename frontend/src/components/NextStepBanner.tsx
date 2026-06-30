import { NextStep } from "../workflow/workflowSteps";

interface Props {
	nextStep: NextStep | null;
}

export function NextStepBanner({ nextStep }: Props) {
	if (!nextStep) return null;
	return (
		<div className="next-step-banner" role="status">
			<strong>Next: {nextStep.title}</strong>
			<p>{nextStep.detail}</p>
		</div>
	);
}
