import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { AppSession, client, LabelingState, NavFrameItem } from "../api/client";
import { humanLabelsDisplayName, labelsFileBasename } from "../api/labelsCsvName";

interface Props {
	humanLabelsPath: string | null;
	machineLabelsPath: string | null;
	videoPaths: string[] | null;
	onClose: (session: AppSession) => void;
	onSessionUpdate: (session: AppSession) => void;
}

/** Firefox reports aborted fetches as NetworkError, not AbortError. */
function isIgnorableFetchError(err: unknown): boolean {
	if (err instanceof DOMException) {
		return err.name === "AbortError";
	}
	if (err instanceof TypeError) {
		const msg = err.message.toLowerCase();
		return msg.includes("networkerror") || msg.includes("aborted");
	}
	return false;
}

/** Matches skellyclicker.core.video_handler.image_annotator.WEB_FULL_HELP_TEXT */
const WEB_FULL_HELP_TEXT = `Click the video to place the active bodypart.
Use 'a' / 'd' or arrow keys for previous / next frame.
Finish all bodyparts on a frame before leaving it (empty frames may be skipped).
Drag the frame slider to scrub previews.
Use the vertical contrast slider (left of the frame) when paused — up = more contrast.
Press 'm' to toggle machine / live prediction overlay.
Press 'n' to toggle bodypart names on the video.
Press 'h' to hide this help.
Press Esc to close (prompts to save).
Use Save to write labels; Close to exit.
Press Space to play or pause frames.
Press 'u' or Ctrl+Z to undo the last label (or clear active label on frame).`;

const PLAY_INTERVAL_MS = 66;
/** After scrub release, repaint a few times so late live-infer crosses appear. */
const LIVE_OVERLAY_RETRY_MS = [120, 280, 500];
/** Matches labeling_engine CONTRAST_* — display-only OpenCV alpha. */
const CONTRAST_MIN = 0.25;
const CONTRAST_MAX = 3;
const CONTRAST_DEFAULT = 1;
const CONTRAST_STEP = 0.05;

function formatPointList(points: string[]): string {
	return `[${points.join(", ")}]`;
}

function pointColorCss(
	colors: Record<string, [number, number, number]>,
	name: string,
): string {
	const rgb = colors[name];
	return rgb ? `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})` : "rgb(255, 0, 255)";
}

/** Colored crosshair cursor matching the active bodypart in the legend. */
function crosshairCursorCss(color: string): string {
	const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'><line x1='12' y1='3' x2='12' y2='21' stroke='${color}' stroke-width='2'/><line x1='3' y1='12' x2='21' y2='12' stroke='${color}' stroke-width='2'/></svg>`;
	return `url("data:image/svg+xml,${encodeURIComponent(svg)}") 12 12, crosshair`;
}

async function isJpegBlob(blob: Blob): Promise<boolean> {
	const header = new Uint8Array(await blob.slice(0, 2).arrayBuffer());
	return header[0] === 0xff && header[1] === 0xd8;
}

function resolveNavFrames(state: LabelingState): NavFrameItem[] {
	// Left panel is human-labeled frames only (live scrub covers predictions).
	if (state.nav_frame_list?.length) {
		return state.nav_frame_list.map((item) => ({
			frame: item.frame,
			kind: "human" as const,
		}));
	}
	return (state.labeled_frame_list ?? []).map((frame) => ({
		frame,
		kind: "human" as const,
	}));
}

function navFrameBtnClass(frame: number, activeFrame: number): string {
	const classes = ["labeling-frame-btn", "labeling-frame-btn--human"];
	if (frame === activeFrame) {
		classes.push("labeling-frame-btn--active");
	}
	return classes.join(" ");
}

export function LabelingCanvas({
	humanLabelsPath,
	machineLabelsPath,
	videoPaths,
	onClose,
	onSessionUpdate,
}: Props) {
	const [state, setState] = useState<LabelingState | null>(null);
	const [sliderFrame, setSliderFrame] = useState(0);
	const [error, setError] = useState<string | null>(null);
	const [saveNotice, setSaveNotice] = useState<string | null>(null);
	const [isClosing, setIsClosing] = useState(false);
	const [isSaving, setIsSaving] = useState(false);
	const [switchingVideo, setSwitchingVideo] = useState(false);
	const labelsPathRef = useRef(humanLabelsPath);
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const containerRef = useRef<HTMLDivElement>(null);
	const stageRef = useRef<HTMLDivElement>(null);
	const frameRef = useRef(0);
	const scrubRafRef = useRef<number | null>(null);
	const liveOverlayTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
	const pendingPreviewFrameRef = useRef<number | null>(null);
	const previewBusyRef = useRef(false);
	const scrubbingRef = useRef(false);
	const previewGenRef = useRef(0);
	const stateRef = useRef<LabelingState | null>(null);
	// Prevent Esc / double-click from starting a second close while the save dialog is open.
	const closingRef = useRef(false);
	const [scrubbing, setScrubbing] = useState(false);
	const playingRef = useRef(false);
	const playTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
	const frameCountRef = useRef(0);
	const gridSizeRef = useRef({ w: 0, h: 0 });
	// CSS display size locked from the last committed frame so scrub previews
	// upscale in place instead of shrinking the on-screen box.
	const lockedDisplaySizeRef = useRef<{ w: number; h: number } | null>(null);
	const [playing, setPlaying] = useState(false);
	const [contrast, setContrast] = useState(CONTRAST_DEFAULT);
	const contrastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	// Vertical rail length = 1/4 of the frame stage height.
	const [contrastSliderPx, setContrastSliderPx] = useState(120);

	useEffect(() => {
		labelsPathRef.current = humanLabelsPath;
	}, [humanLabelsPath]);

	useEffect(() => {
		stateRef.current = state;
		if (typeof state?.contrast === "number") {
			setContrast(state.contrast);
		}
	}, [state]);

	const fitCanvasToStage = useCallback(() => {
		const stage = stageRef.current;
		const canvas = canvasRef.current;
		if (!stage || !canvas) return;

		const displayW = gridSizeRef.current.w || canvas.width;
		const displayH = gridSizeRef.current.h || canvas.height;
		if (displayW <= 0 || displayH <= 0) return;

		const maxW = stage.clientWidth;
		const maxH = stage.clientHeight;
		if (maxW <= 0 || maxH <= 0) return;

		const scale = Math.min(maxW / displayW, maxH / displayH);
		const w = Math.floor(displayW * scale);
		const h = Math.floor(displayH * scale);
		// Explicit pixels — disable CSS max-* so scrub hints shrinking the stage
		// cannot clamp the canvas smaller than the locked committed size.
		canvas.style.maxWidth = "none";
		canvas.style.maxHeight = "none";
		canvas.style.width = `${w}px`;
		canvas.style.height = `${h}px`;
		if (!scrubbingRef.current && !playingRef.current) {
			lockedDisplaySizeRef.current = { w, h };
		}
	}, []);

	const applyCanvasDisplaySize = useCallback(
		(preview: boolean) => {
			const canvas = canvasRef.current;
			if (!canvas) return;
			if (preview && lockedDisplaySizeRef.current) {
				const { w, h } = lockedDisplaySizeRef.current;
				canvas.style.maxWidth = "none";
				canvas.style.maxHeight = "none";
				canvas.style.width = `${w}px`;
				canvas.style.height = `${h}px`;
				return;
			}
			fitCanvasToStage();
		},
		[fitCanvasToStage],
	);

	useEffect(() => {
		if (state?.grid_width && state?.grid_height) {
			gridSizeRef.current = { w: state.grid_width, h: state.grid_height };
			fitCanvasToStage();
		}
	}, [state?.grid_width, state?.grid_height, fitCanvasToStage]);

	const paintFrameBlob = useCallback(
		async (blob: Blob, gen: number, preview: boolean) => {
			if (gen !== previewGenRef.current) return;
			try {
				const bitmap = await createImageBitmap(blob);
				if (gen !== previewGenRef.current) {
					bitmap.close();
					return;
				}
				const canvas = canvasRef.current;
				if (!canvas) {
					bitmap.close();
					return;
				}
				// Prefer session native grid; fall back to bitmap (scrub and commit
				// now share the same server-side grid size).
				let gridW = gridSizeRef.current.w;
				let gridH = gridSizeRef.current.h;
				if (gridW <= 0 || gridH <= 0) {
					gridW = bitmap.width;
					gridH = bitmap.height;
					gridSizeRef.current = { w: gridW, h: gridH };
				}
				const ctx = canvas.getContext("2d");
				if (ctx) {
					// Setting width/height resets the canvas; re-apply CSS size after.
					canvas.width = gridW;
					canvas.height = gridH;
					ctx.imageSmoothingEnabled = true;
					ctx.imageSmoothingQuality = "high";
					ctx.drawImage(bitmap, 0, 0, gridW, gridH);
				}
				bitmap.close();
				applyCanvasDisplaySize(preview);
				setError(null);
			} catch (err) {
				if (gen !== previewGenRef.current) return;
				setError(err instanceof Error ? err.message : String(err));
			}
		},
		[applyCanvasDisplaySize],
	);

	const fetchAndPaintFrame = useCallback(
		async (frameNumber: number, preview: boolean, gen: number) => {
			const blob = await client.fetchFrameJpeg(frameNumber, preview ? { preview: true } : undefined);
			if (gen !== previewGenRef.current) return;
			if (!(await isJpegBlob(blob))) {
				throw new Error(`Failed to load frame ${frameNumber}`);
			}
			await paintFrameBlob(blob, gen, preview);
		},
		[paintFrameBlob],
	);

	const clearLiveOverlayRetries = useCallback(() => {
		for (const id of liveOverlayTimersRef.current) clearTimeout(id);
		liveOverlayTimersRef.current = [];
	}, []);

	const scheduleLiveOverlayRetries = useCallback(
		(frameNumber: number, gen: number) => {
			clearLiveOverlayRetries();
			if (!stateRef.current?.live_inference_ready) return;
			for (const delay of LIVE_OVERLAY_RETRY_MS) {
				const id = setTimeout(() => {
					if (gen !== previewGenRef.current) return;
					if (scrubbingRef.current || playingRef.current) return;
					if (frameRef.current !== frameNumber) return;
					void fetchAndPaintFrame(frameNumber, false, gen).catch(() => {
						/* ignore — next retry or user action will refresh */
					});
				}, delay);
				liveOverlayTimersRef.current.push(id);
			}
		},
		[clearLiveOverlayRetries, fetchAndPaintFrame],
	);

	const loadFrame = useCallback(
		async (frameNumber: number) => {
			clearLiveOverlayRetries();
			const gen = ++previewGenRef.current;
			pendingPreviewFrameRef.current = null;
			const committed = frameRef.current;
			try {
				const s = await client.setFrame(frameNumber);
				if (gen !== previewGenRef.current) return;
				frameRef.current = s.frame_number;
				setSliderFrame(s.frame_number);
				setState(s);
				setError(null);
				await fetchAndPaintFrame(s.frame_number, false, gen);
				if (gen === previewGenRef.current) {
					scheduleLiveOverlayRetries(s.frame_number, gen);
				}
			} catch (err) {
				// Incomplete-frame gate (or other setFrame errors): stay put and notify.
				if (gen !== previewGenRef.current) return;
				frameRef.current = committed;
				setSliderFrame(committed);
				await fetchAndPaintFrame(committed, false, gen).catch(() => {
					/* keep prior canvas if repaint fails */
				});
				if (!isIgnorableFetchError(err)) {
					// Set after repaint — paintFrameBlob clears error on success.
					setError(err instanceof Error ? err.message : String(err));
				}
				throw err;
			}
		},
		[fetchAndPaintFrame, clearLiveOverlayRetries, scheduleLiveOverlayRetries],
	);

	const applyContrast = useCallback(
		async (value: number) => {
			if (closingRef.current || playingRef.current) return;
			const clamped = Math.min(
				CONTRAST_MAX,
				Math.max(CONTRAST_MIN, value),
			);
			try {
				const s = await client.setContrast(clamped);
				setState(s);
				if (typeof s.contrast === "number") setContrast(s.contrast);
				else setContrast(clamped);
				const gen = ++previewGenRef.current;
				await fetchAndPaintFrame(frameRef.current, false, gen);
			} catch (err) {
				if (!isIgnorableFetchError(err)) {
					setError(err instanceof Error ? err.message : String(err));
				}
			}
		},
		[fetchAndPaintFrame],
	);

	const onContrastInput = useCallback(
		(value: number) => {
			setContrast(value);
			if (contrastTimerRef.current != null) clearTimeout(contrastTimerRef.current);
			contrastTimerRef.current = setTimeout(() => {
				contrastTimerRef.current = null;
				void applyContrast(value);
			}, 60);
		},
		[applyContrast],
	);

	const stopPlaying = useCallback(
		(commit = true) => {
			if (playTimerRef.current != null) {
				clearInterval(playTimerRef.current);
				playTimerRef.current = null;
			}
			if (!playingRef.current) return;
			playingRef.current = false;
			setPlaying(false);
			if (commit) {
				void loadFrame(frameRef.current).catch((err) => {
					if (isIgnorableFetchError(err)) return;
					setError(String(err));
				});
			}
		},
		[loadFrame],
	);

	const togglePlaying = useCallback(() => {
		if (!state || closingRef.current || isClosing) return;
		if (playingRef.current) {
			stopPlaying(true);
			return;
		}
		if (scrubbingRef.current) return;
		if (frameRef.current >= state.frame_count - 1) {
			void loadFrame(0).then(() => {
				if (closingRef.current) return;
				playingRef.current = true;
				setPlaying(true);
				playTimerRef.current = setInterval(() => {
					const next = frameRef.current + 1;
					if (next >= frameCountRef.current) {
						stopPlaying(true);
						return;
					}
					frameRef.current = next;
					setSliderFrame(next);
					const gen = ++previewGenRef.current;
					void fetchAndPaintFrame(next, true, gen).catch((err) => {
						if (isIgnorableFetchError(err)) return;
						stopPlaying(false);
						setError(String(err));
					});
				}, PLAY_INTERVAL_MS);
			});
			return;
		}
		playingRef.current = true;
		setPlaying(true);
		playTimerRef.current = setInterval(() => {
			const next = frameRef.current + 1;
			if (next >= frameCountRef.current) {
				stopPlaying(true);
				return;
			}
			frameRef.current = next;
			setSliderFrame(next);
			const gen = ++previewGenRef.current;
			void fetchAndPaintFrame(next, true, gen).catch((err) => {
				if (isIgnorableFetchError(err)) return;
				stopPlaying(false);
				setError(String(err));
			});
		}, PLAY_INTERVAL_MS);
	}, [state, isClosing, stopPlaying, loadFrame, fetchAndPaintFrame]);

	const drainPreviewQueue = useCallback(async () => {
		if (previewBusyRef.current) return;
		previewBusyRef.current = true;
		try {
			while (pendingPreviewFrameRef.current != null && scrubbingRef.current) {
				const frameNumber = pendingPreviewFrameRef.current;
				pendingPreviewFrameRef.current = null;
				const gen = ++previewGenRef.current;
				try {
					await fetchAndPaintFrame(frameNumber, true, gen);
					if (scrubbingRef.current && gen === previewGenRef.current) {
						frameRef.current = frameNumber;
					}
				} catch (err) {
					if (isIgnorableFetchError(err)) continue;
					if (gen !== previewGenRef.current) continue;
					setError(err instanceof Error ? err.message : String(err));
					return;
				}
			}
		} finally {
			previewBusyRef.current = false;
			if (pendingPreviewFrameRef.current != null && scrubbingRef.current) {
				void drainPreviewQueue();
			}
		}
	}, [fetchAndPaintFrame]);

	const schedulePreviewFrame = useCallback(
		(frameNumber: number) => {
			pendingPreviewFrameRef.current = frameNumber;
			if (scrubRafRef.current != null) return;
			scrubRafRef.current = requestAnimationFrame(() => {
				scrubRafRef.current = null;
				void drainPreviewQueue();
			});
		},
		[drainPreviewQueue],
	);

	const commitScrub = useCallback(
		(frameNumber: number) => {
			scrubbingRef.current = false;
			setScrubbing(false);
			pendingPreviewFrameRef.current = null;
			void loadFrame(frameNumber).catch((err) => {
				if (isIgnorableFetchError(err)) return;
				setError(String(err));
			});
		},
		[loadFrame],
	);

	const refresh = useCallback(async () => {
		const gen = ++previewGenRef.current;
		const s = await client.labelingState();
		if (gen !== previewGenRef.current) return;
		frameRef.current = s.frame_number;
		setSliderFrame(s.frame_number);
		setState(s);
		await fetchAndPaintFrame(s.frame_number, false, gen);
	}, [fetchAndPaintFrame]);

	useEffect(() => {
		refresh().catch((e) => {
			if (!isIgnorableFetchError(e)) setError(String(e));
		});
		return () => {
			previewGenRef.current += 1;
			pendingPreviewFrameRef.current = null;
			clearLiveOverlayRetries();
			if (scrubRafRef.current != null) cancelAnimationFrame(scrubRafRef.current);
			if (playTimerRef.current != null) clearInterval(playTimerRef.current);
			if (contrastTimerRef.current != null) clearTimeout(contrastTimerRef.current);
			playingRef.current = false;
		};
	}, [refresh, clearLiveOverlayRetries]);

	useEffect(() => {
		frameCountRef.current = state?.frame_count ?? 0;
	}, [state?.frame_count]);

	useEffect(() => {
		containerRef.current?.focus();
	}, [state?.session_id]);

	useEffect(() => {
		const stage = stageRef.current;
		if (!stage) return;
		const update = () => {
			const h = stage.clientHeight;
			if (h > 0) setContrastSliderPx(Math.max(64, Math.round(h * 0.25)));
		};
		update();
		const observer = new ResizeObserver(update);
		observer.observe(stage);
		return () => observer.disconnect();
	}, [state?.session_id]);

	useEffect(() => {
		const stage = stageRef.current;
		if (!stage) return;
		const observer = new ResizeObserver(() => fitCanvasToStage());
		observer.observe(stage);
		return () => observer.disconnect();
	}, [fitCanvasToStage, state?.session_id]);

	const closeLabeler = useCallback(
		async (save: boolean) => {
			if (closingRef.current) return;
			closingRef.current = true;
			setIsClosing(true);
			setError(null);
			stopPlaying(false);
			try {
				// Human labels always go to the DLC project labeled-data folder.
				const session = await client.closeLabeler(save);
				onClose(session);
			} catch (e) {
				closingRef.current = false;
				setIsClosing(false);
				setError(e instanceof Error ? e.message : String(e));
			}
		},
		[onClose, stopPlaying],
	);

	const saveLabels = useCallback(async () => {
		if (closingRef.current || isSaving) return;
		setIsSaving(true);
		setError(null);
		setSaveNotice(null);
		stopPlaying(false);
		try {
			const session = await client.saveLabeler();
			labelsPathRef.current = session.human_labels_path;
			onSessionUpdate(session);
			setSaveNotice(session.status_message ?? "Labels saved.");
		} catch (e) {
			setError(e instanceof Error ? e.message : String(e));
		} finally {
			setIsSaving(false);
		}
	}, [onSessionUpdate, stopPlaying, isSaving]);

	const undoLastLabel = useCallback(async () => {
		if (!state || closingRef.current) return;
		stopPlaying(false);
		const gen = ++previewGenRef.current;
		try {
			const s = await client.undoLabel();
			if (gen !== previewGenRef.current) return;
			frameRef.current = s.frame_number;
			setSliderFrame(s.frame_number);
			setState(s);
			setError(null);
			await fetchAndPaintFrame(s.frame_number, false, gen);
		} catch (err) {
			if (isIgnorableFetchError(err)) return;
			setError(err instanceof Error ? err.message : String(err));
		}
	}, [state, fetchAndPaintFrame, stopPlaying]);

	const refreshAfterOverlayToggle = useCallback(
		async (toggleFn: () => Promise<LabelingState>) => {
			if (!state || closingRef.current) return;
			stopPlaying(false);
			const frame = frameRef.current;
			const gen = ++previewGenRef.current;
			try {
				// Preview playback advances frameRef without committing; sync before repaint.
				await client.setFrame(frame);
				const s = await toggleFn();
				if (gen !== previewGenRef.current) return;
				frameRef.current = frame;
				setSliderFrame(frame);
				setState(s);
				await fetchAndPaintFrame(frame, false, gen);
			} catch (err) {
				if (isIgnorableFetchError(err)) return;
				setError(err instanceof Error ? err.message : String(err));
			}
		},
		[state, stopPlaying, fetchAndPaintFrame],
	);

	useEffect(() => {
		const onKey = (e: KeyboardEvent) => {
			if (!state || closingRef.current) return;
			const key = e.key.toLowerCase();

			if (key === "escape") {
				e.preventDefault();
				const save = window.confirm("Save labels before closing?");
				void closeLabeler(save);
				return;
			}
			if (key === " ") {
				e.preventDefault();
				togglePlaying();
				return;
			}
			if (
				key === "u" ||
				key === "backspace" ||
				((e.ctrlKey || e.metaKey) && key === "z" && !e.shiftKey)
			) {
				e.preventDefault();
				void undoLastLabel();
				return;
			}
			if (key === "a" || key === "arrowleft") {
				e.preventDefault();
				stopPlaying(false);
				const n = Math.max(0, frameRef.current - 1);
				loadFrame(n).catch((err) => {
					if (isIgnorableFetchError(err)) return;
					setError(String(err));
				});
				return;
			}
			if (key === "d" || key === "arrowright") {
				e.preventDefault();
				stopPlaying(false);
				const n = Math.min(state.frame_count - 1, frameRef.current + 1);
				loadFrame(n).catch((err) => {
					if (isIgnorableFetchError(err)) return;
					setError(String(err));
				});
				return;
			}
			if (key === "m") {
				e.preventDefault();
				void refreshAfterOverlayToggle(() => client.toggleMachineOverlay());
				return;
			}
			if (key === "n") {
				e.preventDefault();
				void refreshAfterOverlayToggle(() => client.toggleLabelNames());
				return;
			}
			if (key === "h") {
				e.preventDefault();
				client
					.toggleHelp()
					.then((s) => {
						frameRef.current = s.frame_number;
						setState(s);
					})
					.catch((err) => {
						if (isIgnorableFetchError(err)) return;
						setError(String(err));
					});
			}
		};
		window.addEventListener("keydown", onKey);
		return () => window.removeEventListener("keydown", onKey);
	}, [state, closeLabeler, loadFrame, fetchAndPaintFrame, stopPlaying, togglePlaying, undoLastLabel, refreshAfterOverlayToggle]);

	const onActivePoint = useCallback(
		(pointName: string) => {
			client
				.setActivePoint(pointName)
				.then((s) => {
					frameRef.current = s.frame_number;
					setState(s);
				})
				.catch((err) => {
					if (isIgnorableFetchError(err)) return;
					setError(String(err));
				});
		},
		[],
	);

	// Corpus mode: persist current video labels, reopen labeler on the selected video.
	const onActiveVideo = useCallback(
		async (path: string) => {
			setSwitchingVideo(true);
			setError(null);
			stopPlaying(false);
			try {
				const session = await client.setActiveLabelingVideo(path);
				onSessionUpdate(session);
				await refresh();
			} catch (err) {
				if (!isIgnorableFetchError(err)) setError(String(err));
			} finally {
				setSwitchingVideo(false);
			}
		},
		[onSessionUpdate, refresh, stopPlaying],
	);

	const onClick = async (e: React.MouseEvent<HTMLCanvasElement>) => {
		const canvas = canvasRef.current;
		if (!canvas || !state || closingRef.current) return;
		stopPlaying(false);
		const rect = canvas.getBoundingClientRect();
		const scaleX = canvas.width / rect.width;
		const scaleY = canvas.height / rect.height;
		const x = Math.round((e.clientX - rect.left) * scaleX);
		const y = Math.round((e.clientY - rect.top) * scaleY);
		const gen = ++previewGenRef.current;
		try {
			const s = await client.click(x, y);
			if (gen !== previewGenRef.current) return;
			frameRef.current = s.frame_number;
			setState(s);
			await fetchAndPaintFrame(s.frame_number, false, gen);
		} catch (err) {
			if (isIgnorableFetchError(err)) return;
			setError(err instanceof Error ? err.message : String(err));
		}
	};

	const onScrubStart = () => {
		stopPlaying(false);
		// Freeze on-screen canvas size before preview JPEGs arrive.
		const canvas = canvasRef.current;
		if (canvas) {
			const w = canvas.clientWidth;
			const h = canvas.clientHeight;
			if (w > 0 && h > 0) {
				lockedDisplaySizeRef.current = { w, h };
			}
		}
		scrubbingRef.current = true;
		setScrubbing(true);
		setError(null);
	};

	const onSliderInput = (frameNumber: number) => {
		setSliderFrame(frameNumber);
		if (scrubbingRef.current) {
			schedulePreviewFrame(frameNumber);
		}
	};

	const onSliderCommit = (frameNumber: number) => {
		stopPlaying(false);
		commitScrub(frameNumber);
	};

	const jumpToLabeledFrame = useCallback(
		(frameNumber: number) => {
			if (isClosing || closingRef.current) return;
			stopPlaying(false);
			loadFrame(frameNumber).catch((e) => {
				if (isIgnorableFetchError(e)) return;
				setError(String(e));
			});
		},
		[isClosing, loadFrame, stopPlaying],
	);

	if (!state) return <p>Loading labeler…</p>;

	const activeCursor = crosshairCursorCss(
		pointColorCss(state.point_colors, state.active_point),
	);
	const humanLabelsName = humanLabelsDisplayName(
		labelsPathRef.current ?? humanLabelsPath,
		videoPaths,
	);
	const machineLabelsName = labelsFileBasename(machineLabelsPath);
	const navFrames = resolveNavFrames(state);
	const showVideoSelector =
		state.labeling_mode === "corpus" &&
		(state.session_videos?.length ?? 0) > 1;

	return (
		<div
			className="labeling"
			ref={containerRef}
			tabIndex={0}
			onMouseDown={() => containerRef.current?.focus()}
		>
			<div className="labeling-toolbar">
				<span className="hint labeling-toolbar-hint">
					a/d or ←/→ frames · Space play/pause · u / Ctrl+Z undo · n label names · scrub slider · m machine overlay · h help · Esc close
				</span>
			</div>
			{error && <div className="error">{error}</div>}
			{saveNotice && <p className="hint save-notice">{saveNotice}</p>}
			{isClosing && <p className="hint">Saving and closing…</p>}
			<div className="labeling-body">
				<aside className="labeling-frame-list" aria-label="Frame navigation">
					<h3 className="labeling-hud-title">
						Labeled frames: {state.labeled_frames}
					</h3>
					{navFrames.length > 0 ? (
						<ul className="labeling-frame-queue">
							{navFrames.map((item) => (
								<li key={item.frame}>
									<button
										type="button"
										className={navFrameBtnClass(
											item.frame,
											state.frame_number,
										)}
										disabled={isClosing}
										onClick={() => jumpToLabeledFrame(item.frame)}
									>
										Frame {item.frame}
									</button>
								</li>
							))}
						</ul>
					) : (
						<p className="labeling-frame-list-empty hint">No labeled frames yet</p>
					)}
				</aside>
				{/* Keep rail mounted while scrubbing/playing so the frame stage does not shift. */}
				<aside
					className="labeling-contrast-rail"
					aria-label="Frame contrast"
					style={
						{
							["--contrast-slider-h" as string]: `${contrastSliderPx}px`,
						} as CSSProperties
					}
				>
					<span
						className="frame-contrast-value"
						title="Double-click to reset to 100%"
						onDoubleClick={() => {
							if (isClosing || playing) return;
							setContrast(CONTRAST_DEFAULT);
							if (contrastTimerRef.current != null) {
								clearTimeout(contrastTimerRef.current);
								contrastTimerRef.current = null;
							}
							void applyContrast(CONTRAST_DEFAULT);
						}}
					>
						{Math.round(contrast * 100)}%
					</span>
					<input
						id="contrast-slider"
						className="frame-contrast-slider"
						type="range"
						min={CONTRAST_MIN}
						max={CONTRAST_MAX}
						step={CONTRAST_STEP}
						value={contrast}
						disabled={isClosing || playing}
						aria-valuetext={`${Math.round(contrast * 100)} percent`}
						aria-orientation="vertical"
						title="Drag up for more contrast (display only)"
						onInput={(e) => onContrastInput(Number(e.currentTarget.value))}
						onChange={(e) => onContrastInput(Number(e.currentTarget.value))}
					/>
					<label htmlFor="contrast-slider" className="frame-contrast-label">
						Contrast
					</label>
				</aside>
				<div className="labeling-center">
					<div className="labeling-nav labeling-nav--actions">
						<p className="hint labeling-save-hint">
							Save writes <strong>human labels</strong> only. Machine labels are
							read-only overlay (press m).
						</p>
						<div className="labeler-action-btn-row">
							<button
								type="button"
								className="labeler-action-btn"
								disabled={isClosing || isSaving}
								onClick={() => void saveLabels()}
							>
								{isSaving ? "Saving…" : "Save"}
							</button>
							<button
								type="button"
								className="labeler-action-btn"
								disabled={isClosing}
								onClick={() => {
									if (window.confirm("Close without saving labels?")) {
										void closeLabeler(false);
									}
								}}
							>
								Close
							</button>
						</div>
					</div>
					<div className="labeling-stage" ref={stageRef}>
						<canvas
							ref={canvasRef}
							className="label-canvas"
							style={{ cursor: activeCursor }}
							onClick={onClick}
						/>
					</div>
					<div className="labeling-close-actions labeling-close-actions--nav">
						<div className="labeler-action-btn-row">
							<button
								type="button"
								disabled={state.frame_number <= 0 || isClosing}
								onClick={() => {
									stopPlaying(false);
									loadFrame(state.frame_number - 1).catch((e) => {
										if (isIgnorableFetchError(e)) return;
										setError(String(e));
									});
								}}
							>
								← Prev
							</button>
							<button
								type="button"
								disabled={state.frame_number >= state.frame_count - 1 || isClosing}
								onClick={() => {
									stopPlaying(false);
									loadFrame(state.frame_number + 1).catch((e) => {
										if (isIgnorableFetchError(e)) return;
										setError(String(e));
									});
								}}
							>
								Next →
							</button>
						</div>
					</div>
				</div>
				<aside className="labeling-hud" aria-label="Labeler info">
					{showVideoSelector && (
						<div className="labeling-hud-section">
							<h3 className="labeling-hud-title">Video</h3>
							<select
								className="labeling-video-select"
								aria-label="Active video"
								disabled={isClosing || isSaving || switchingVideo}
								value={state.active_video_path ?? ""}
								onChange={(e) => {
									const next = e.target.value;
									if (!next || next === state.active_video_path) return;
									void onActiveVideo(next);
								}}
							>
								{state.session_videos!.map((v) => (
									<option key={v.path} value={v.path}>
										{v.name}
									</option>
								))}
							</select>
							{switchingVideo && (
								<p className="hint labeling-hud-line">Switching video…</p>
							)}
						</div>
					)}
					<div className="labeling-hud-section">
						<h3 className="labeling-hud-title">Label files</h3>
						<p className="labeling-hud-line">
							<span className="labeling-hud-file-kind">Human (diamonds — edit &amp; save)</span>
							<strong className="labeling-hud-file-name" title={labelsPathRef.current ?? humanLabelsPath ?? undefined}>
								{humanLabelsName}
							</strong>
						</p>
						{machineLabelsName ? (
							<p className="labeling-hud-line">
								<span className="labeling-hud-file-kind">Machine (crosses — read-only, m)</span>
								<strong className="labeling-hud-file-name" title={machineLabelsPath ?? undefined}>
									{machineLabelsName}
								</strong>
							</p>
						) : (
							<p className="labeling-hud-line labeling-hud-line--muted">
								No machine labels loaded
							</p>
						)}
					</div>
					<div className="labeling-hud-section">
						<h3 className="labeling-hud-title">Frame</h3>
						<p className="labeling-hud-line">
							{state.frame_number} / {state.frame_count}
						</p>
						<p className="labeling-hud-line">
							Active: <strong>{state.active_point}</strong>
						</p>
					</div>
					<div className="labeling-hud-section">
						<h3 className="labeling-hud-title">Labels on frame</h3>
						{state.placed_points.length > 0 && (
							<p className="labeling-hud-line">
								On frame: {formatPointList(state.placed_points)}
							</p>
						)}
						<p className="labeling-hud-line">
							Available: {formatPointList(state.available_points)}
						</p>
					</div>
					<div className="labeling-hud-section labeling-hud-section--legend">
						<h3 className="labeling-hud-title">Legend</h3>
						<div className="label-legend-keys">
							<span className="label-legend-key">
								<span className="label-legend-marker label-legend-marker--human label-legend-marker--sample" />
								Your labels (diamond)
							</span>
							{state.has_machine_labels && (
								<span className="label-legend-key">
									<span className="label-legend-marker label-legend-marker--machine label-legend-marker--sample" />
									Model overlay (cross, m)
								</span>
							)}
						</div>
						<div className="label-legend">
							{state.tracked_points.map((name) => {
								const color = pointColorCss(state.point_colors, name);
								const markerStyle = { "--marker-color": color } as CSSProperties;
								return (
									<div className="label-legend-row" key={name}>
										<span
											className="label-legend-marker label-legend-marker--human"
											style={markerStyle}
										/>
										<button
											type="button"
											className={`label-legend-name label-legend-name-btn${
												state.active_point === name
													? " label-legend-name-btn--active"
													: ""
											}`}
											style={{ color }}
											disabled={isClosing}
											onClick={() => onActivePoint(name)}
										>
											{name}
										</button>
									</div>
								);
							})}
						</div>
					</div>
					<div className="labeling-hud-section labeling-hud-section--help">
						{state.show_help ? (
							<pre className="labeling-help-full">{WEB_FULL_HELP_TEXT}</pre>
						) : (
							<p className="labeling-help-short">Press H for help · Esc to close</p>
						)}
					</div>
				</aside>
			</div>
			<div className={`frame-scrubber${scrubbing ? " frame-scrubber--scrubbing" : ""}${playing ? " frame-scrubber--playing" : ""}`}>
				<div className="frame-scrubber-row">
					<button
						type="button"
						className={`frame-play-btn${playing ? " frame-play-btn--pause" : ""}`}
						disabled={isClosing || state.frame_count <= 1}
						onClick={() => togglePlaying()}
						aria-label={playing ? "Pause" : "Play"}
						title={playing ? "Pause (Space)" : "Play (Space)"}
					>
						<span className="frame-play-btn-icon" aria-hidden="true" />
					</button>
					<input
						id="frame-slider"
						className="frame-scrubber-slider"
						type="range"
						min={0}
						max={Math.max(0, state.frame_count - 1)}
						value={sliderFrame}
						disabled={isClosing}
						onPointerDown={onScrubStart}
						onInput={(e) => onSliderInput(Number(e.currentTarget.value))}
						onChange={(e) => onSliderInput(Number(e.currentTarget.value))}
						onPointerUp={(e) => onSliderCommit(Number(e.currentTarget.value))}
						onPointerCancel={(e) => onSliderCommit(Number(e.currentTarget.value))}
					/>
					<span className="frame-scrubber-time">
						{sliderFrame} / {state.frame_count}
					</span>
				</div>
				{playing && (
					<p className="hint scrub-hint">Playing preview frames — pause to edit labels</p>
				)}
				{scrubbing && !playing && (state.has_machine_labels || state.live_inference_ready) && (
					<p className="hint scrub-hint">
						{state.live_inference_ready
							? "Live machine preview (not saved) — m toggles · release to label"
							: "Machine predictions while scrubbing — m toggles · release to label"}
					</p>
				)}
				{scrubbing && !playing && !state.has_machine_labels && !state.live_inference_ready && (
					<p className="hint scrub-hint">Release slider to load full frame and label</p>
				)}
				{!scrubbing && !playing && state.live_inference_ready && (
					<p className="hint scrub-hint">
						{state.show_machine_labels
							? "Live preview on (not saved) — press m to hide · click to place human labels"
							: "Live preview hidden — press m to show · click to place human labels"}
					</p>
				)}
			</div>
		</div>
	);
}
