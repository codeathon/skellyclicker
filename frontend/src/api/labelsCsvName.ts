/** Match server default in skellyclicker.core.human_labels_io.human_labels_csv_filename */
export function humanLabelsCsvDefaultName(
	videoPaths: string[] | null | undefined,
	now = new Date(),
): string {
	const pad = (n: number) => String(n).padStart(2, "0");
	const timestamp = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}_${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`;

	const stems = (videoPaths ?? [])
		.map((path) => {
			const file = path.split(/[/\\]/).pop() ?? "";
			const dot = file.lastIndexOf(".");
			const stem = dot > 0 ? file.slice(0, dot) : file;
			return stem.replace(/[^\w\-.]+/g, "_") || "";
		})
		.filter(Boolean);

	const videoPart =
		stems.length === 0 ? "video" : stems.length === 1 ? stems[0] : stems.join("_");

	return `${timestamp}_${videoPart}_skellyclicker_labels.csv`;
}

/** Basename of a labels CSV path for compact UI display. */
export function labelsFileBasename(path: string | null | undefined): string | null {
	if (!path) return null;
	const parts = path.split(/[/\\]/);
	return parts[parts.length - 1] || path;
}

/** Basename of the human labels CSV in use, or the default name if saving later. */
export function humanLabelsDisplayName(
	humanLabelsPath: string | null | undefined,
	videoPaths: string[] | null | undefined,
): string {
	if (humanLabelsPath) {
		return labelsFileBasename(humanLabelsPath) ?? humanLabelsPath;
	}
	return humanLabelsCsvDefaultName(videoPaths);
}

/** Default filename for the save dialog (basename only). */
export function humanLabelsSaveDefaultName(
	humanLabelsPath: string | null | undefined,
	videoPaths: string[] | null | undefined,
): string {
	return humanLabelsDisplayName(humanLabelsPath, videoPaths);
}
