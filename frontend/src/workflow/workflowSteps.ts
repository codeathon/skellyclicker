import { AppSession } from "../api/client";

export type StepId =
	| "videos"
	| "dlc"
	| "label"
	| "train"
	| "analyze"
	| "review";

export type StepStatus = "done" | "current" | "upcoming";

export type PhaseLabel = "New model" | "Resume" | "Review predictions";

export interface WorkflowStep {
	id: StepId;
	label: string;
	status: StepStatus;
}

export interface NextStep {
	stepId: StepId;
	title: string;
	detail: string;
}

export interface WorkflowGuide {
	phaseLabel: PhaseLabel;
	steps: WorkflowStep[];
	nextStep: NextStep | null;
	currentStepId: StepId | null;
	showResumeSection: boolean;
}

function hasVideos(session: AppSession): boolean {
	return !!session.videos?.length;
}

function hasDlc(session: AppSession): boolean {
	return !!session.dlc_project_path;
}

function hasLabelsForTrain(session: AppSession): boolean {
	if (session.train_on_machine_labels) {
		return !!session.machine_labels_path;
	}
	return !!session.human_labels_path;
}

function hasTrained(session: AppSession): boolean {
	return session.dlc_iteration != null;
}

function hasMachineLabels(session: AppSession): boolean {
	return !!session.machine_labels_path;
}

export function isResumeContext(session: AppSession): boolean {
	if (session.session_saved_path) return true;
	return hasDlc(session) && hasTrained(session);
}

export function derivePhaseLabel(session: AppSession): PhaseLabel {
	if (session.workflow_state === "review" || (hasMachineLabels(session) && !session.human_labels_path)) {
		return "Review predictions";
	}
	if (isResumeContext(session)) return "Resume";
	return "New model";
}

function stepDone(id: StepId, session: AppSession): boolean {
	switch (id) {
		case "videos":
			return hasVideos(session);
		case "dlc":
			return hasDlc(session);
		case "label":
			return hasLabelsForTrain(session);
		case "train":
			return hasTrained(session);
		case "analyze":
			return hasMachineLabels(session);
		case "review":
			return session.workflow_state === "review" && hasMachineLabels(session);
		default:
			return false;
	}
}

function firstIncompleteStep(session: AppSession): StepId {
	const order: StepId[] = ["videos", "dlc", "label", "train", "analyze", "review"];
	for (const id of order) {
		if (!stepDone(id, session)) return id;
	}
	return "review";
}

const NEXT_COPY: Record<StepId, { title: string; detail: string }> = {
	videos: {
		title: "Add videos",
		detail: "Use Add Videos to load your training recordings (same folder for multi-camera).",
	},
	dlc: {
		title: "Create a DLC project",
		detail: "Create a new project or load an existing one below.",
	},
	label: {
		title: "Label frames",
		detail: "Open Labeler, click bodyparts on key frames, then Save & Close.",
	},
	train: {
		title: "Train the network",
		detail: "Run Train Network once human labels are saved.",
	},
	analyze: {
		title: "Analyze videos",
		detail:
			"Partial Analysis re-runs inference on human-labeled frames only (fast after re-train). Full Analysis processes every frame.",
	},
	review: {
		title: "Review predictions",
		detail: "Open Labeler (press m for machine overlay), fix mistakes, Save & Close, then re-train.",
	},
};

export function deriveWorkflowGuide(session: AppSession): WorkflowGuide {
	const phaseLabel = derivePhaseLabel(session);
	const currentId = firstIncompleteStep(session);
	const order: StepId[] = ["videos", "dlc", "label", "train", "analyze", "review"];
	const labels: Record<StepId, string> = {
		videos: "Videos",
		dlc: "DLC project",
		label: "Label",
		train: "Train",
		analyze: "Analyze",
		review: "Review",
	};

	const steps: WorkflowStep[] = order.map((id) => {
		let status: StepStatus = "upcoming";
		if (stepDone(id, session)) status = "done";
		else if (id === currentId) status = "current";
		return { id, label: labels[id], status };
	});

	const nextCopy = NEXT_COPY[currentId];
	const showResumeSection =
		phaseLabel !== "New model" || stepDone("dlc", session) || !!session.session_saved_path;

	return {
		phaseLabel,
		steps,
		nextStep: nextCopy
			? { stepId: currentId, title: nextCopy.title, detail: nextCopy.detail }
			: null,
		currentStepId: currentId,
		showResumeSection,
	};
}

export function canOpenLabeler(session: AppSession): boolean {
	if (!hasVideos(session)) return false;
	if (session.human_labels_path || session.machine_labels_path) return true;
	return !!(session.dlc_project_path && session.tracked_point_names.length);
}

export function trainBlockReason(session: AppSession): string | null {
	if (session.active_job_id) return "Wait for the current job to finish";
	if (!session.dlc_project_path) return "Load or create a DLC project first";
	if (!hasVideos(session)) return "Add videos first";
	if (!hasLabelsForTrain(session)) {
		return session.train_on_machine_labels
			? "Import or generate machine labels before training"
			: "Label videos or import human labels before training";
	}
	return null;
}

export function canTrain(session: AppSession): boolean {
	return trainBlockReason(session) === null;
}

export function analyzeBlockReason(session: AppSession): string | null {
	if (session.active_job_id) return "Wait for the current job to finish";
	if (!session.dlc_project_path) return "Load or create a DLC project first";
	if (!hasVideos(session)) return "Add videos first";
	if (!hasTrained(session)) return "Train the network before analyzing";
	return null;
}

export function partialAnalyzeBlockReason(session: AppSession): string | null {
	const base = analyzeBlockReason(session);
	if (base) return base;
	if (!session.human_labels_path) return "Save human labels before partial analysis";
	return null;
}

export function canAnalyze(session: AppSession): boolean {
	return analyzeBlockReason(session) === null;
}

export function canPartialAnalyze(session: AppSession): boolean {
	return partialAnalyzeBlockReason(session) === null;
}
