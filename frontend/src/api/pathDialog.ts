/**
 * Native file dialogs via the API (tkinter on the server machine).
 * Falls back to window.prompt when dialogs are unavailable (e.g. headless).
 */

import { client } from "./client";

function promptPath(label: string, defaultValue = ""): string | null {
	const v = window.prompt(label, defaultValue);
	return v?.trim() || null;
}

function promptPaths(label: string): string[] | null {
	const v = window.prompt(
		label + "\nEnter one path per line, or comma-separated.",
	);
	if (!v?.trim()) return null;
	return v
		.split(/[\n,]+/)
		.map((p) => p.trim())
		.filter(Boolean);
}

async function withFallback<T>(
	picker: () => Promise<T | null>,
	fallback: () => T | null,
): Promise<T | null> {
	try {
		return await picker();
    } catch (e) {
		if (e instanceof Error && e.message === "DIALOG_UNAVAILABLE") {
			return fallback();
		}
		throw e;
	}
}

export const pathDialog = {
	openVideos: (title = "Select videos") =>
		withFallback(
			async () => {
				const { paths } = await client.dialogOpenFiles(title, [
					"mp4",
					"avi",
					"mov",
				]);
				return paths.length ? paths : null;
			},
			() => promptPaths(title),
		),

	openCsv: (title: string) =>
		withFallback(
			async () => {
				const { paths } = await client.dialogOpenFile(title, ["csv"]);
				return paths[0] ?? null;
			},
			() => promptPath(title),
		),

	openDirectory: (title: string) =>
		withFallback(
			async () => {
				const { paths } = await client.dialogOpenDirectory(title);
				return paths[0] ?? null;
			},
			() => promptPath(title),
		),

	saveCsv: (title = "Save human labels CSV") =>
		withFallback(
			async () => {
				const { paths } = await client.dialogSaveFile(
					title,
					["csv"],
					"skellyclicker_labels.csv",
				);
				return paths[0] ?? null;
			},
			() => promptPath(title),
		),

	saveSessionJson: (defaultName = "session.json") =>
		withFallback(
			async () => {
				const { paths } = await client.dialogSaveFile(
					"Save session",
					["json"],
					defaultName,
				);
				return paths[0] ?? null;
			},
			() => promptPath("Save session JSON path", defaultName),
		),

	openSessionJson: () =>
		withFallback(
			async () => {
				const { paths } = await client.dialogOpenFile("Load session", ["json"]);
				return paths[0] ?? null;
			},
			() => promptPath("Session JSON path"),
		),
};
