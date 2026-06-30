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

/** Basename of the human labels CSV in use, or the default name if saving later. */
export function humanLabelsDisplayName(
	humanLabelsPath: string | null | undefined,
	videoPaths: string[] | null | undefined,
): string {
	if (humanLabelsPath) {
		const parts = humanLabelsPath.split(/[/\\]/);
		return parts[parts.length - 1] || humanLabelsPath;
	}
	return humanLabelsCsvDefaultName(videoPaths);
}
