/**
 * File pickers: native Ubuntu dialog (zenity/tk) on the server, then browser
 * file picker + upload when server dialogs are unavailable.
 */

import { client } from "./client";

function pickBrowserFiles(
	accept: string,
	multiple: boolean,
): Promise<File[] | null> {
	return new Promise((resolve) => {
		const input = document.createElement("input");
		input.type = "file";
		input.accept = accept;
		input.multiple = multiple;
		input.style.display = "none";
		document.body.appendChild(input);
		input.addEventListener("change", () => {
			const files = input.files ? Array.from(input.files) : [];
			input.remove();
			resolve(files.length ? files : null);
		});
		input.addEventListener("cancel", () => {
			input.remove();
			resolve(null);
		});
		input.click();
	});
}

async function uploadBrowserFiles(files: File[]): Promise<string[]> {
	const form = new FormData();
	for (const file of files) form.append("files", file);
	const res = await fetch("/api/upload/files", { method: "POST", body: form });
	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: res.statusText }));
		throw new Error(
			typeof err.detail === "string" ? err.detail : res.statusText,
		);
	}
	const data = (await res.json()) as { paths: string[] };
	return data.paths;
}

async function withBrowserUploadFallback(
	picker: () => Promise<string[] | null>,
	accept: string,
	multiple: boolean,
): Promise<string[] | null> {
	try {
		return await picker();
	} catch (e) {
		if (e instanceof Error && e.message === "DIALOG_UNAVAILABLE") {
			const files = await pickBrowserFiles(accept, multiple);
			if (!files?.length) return null;
			return uploadBrowserFiles(files);
		}
		throw e;
	}
}

function serverDialogUnavailableMessage(action: string): string {
	return (
		`Could not open a file browser for ${action} on the Ubuntu machine.\n\n` +
		"Install zenity and restart from the desktop session:\n" +
		"  sudo apt install zenity\n\n" +
		"Server-side folder pickers are required for this action."
	);
}

async function requireServerDialog<T>(
	picker: () => Promise<T | null>,
	action: string,
): Promise<T | null> {
	try {
		return await picker();
	} catch (e) {
		if (e instanceof Error && e.message === "DIALOG_UNAVAILABLE") {
			window.alert(serverDialogUnavailableMessage(action));
			return null;
		}
		throw e;
	}
}

async function pickServerPaths(
	call: () => Promise<{ paths: string[] }>,
): Promise<string[] | null> {
	const { paths } = await call();
	return paths.length ? paths : null;
}

export const pathDialog = {
	openVideos: (title = "Select videos") =>
		withBrowserUploadFallback(
			() =>
				pickServerPaths(() =>
					client.dialogOpenFiles(title, ["mp4", "avi", "mov"]),
				),
			".mp4,.avi,.mov,video/*",
			true,
		),

	openCsv: (title: string) =>
		withBrowserUploadFallback(
			() => pickServerPaths(() => client.dialogOpenFile(title, ["csv"])),
			".csv,text/csv",
			false,
		).then((paths) => paths?.[0] ?? null),

	openDirectory: (title: string) =>
		requireServerDialog(
			() => pickServerPaths(() => client.dialogOpenDirectory(title)),
			"folder selection",
		).then((paths) => paths?.[0] ?? null),

	openDlcProject: (title = "DLC project folder") =>
		requireServerDialog(async () => {
			const config = await pickServerPaths(() =>
				client.dialogOpenFile("Select config.yaml", ["yaml", "yml"]),
			);
			if (config?.[0]) return config[0];
			const dir = await pickServerPaths(() =>
				client.dialogOpenDirectory(title),
			);
			return dir?.[0] ?? null;
		}, "DLC project selection"),

	saveCsv: (title = "Save human labels CSV") =>
		requireServerDialog(
			() =>
				pickServerPaths(() =>
					client.dialogSaveFile(title, ["csv"], "skellyclicker_labels.csv"),
				),
			"saving labels",
		).then((paths) => paths?.[0] ?? null),

	/** Labeler save — null means use the server default path under the video folder. */
	saveCsvForLabeler: async () => {
		try {
			const { paths } = await client.dialogSaveFile(
				"Save human labels CSV",
				["csv"],
				"skellyclicker_labels.csv",
			);
			return paths[0] ?? null;
		} catch (e) {
			if (e instanceof Error && e.message === "DIALOG_UNAVAILABLE") {
				return null;
			}
			throw e;
		}
	},

	saveSessionJson: (defaultName = "session.json") =>
		requireServerDialog(
			() =>
				pickServerPaths(() =>
					client.dialogSaveFile("Save session", ["json"], defaultName),
				),
			"saving the session",
		).then((paths) => paths?.[0] ?? null),

	openSessionJson: () =>
		withBrowserUploadFallback(
			() =>
				pickServerPaths(() => client.dialogOpenFile("Load session", ["json"])),
			".json,application/json",
			false,
		).then((paths) => paths?.[0] ?? null),
};
