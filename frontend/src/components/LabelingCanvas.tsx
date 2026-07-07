import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { AppSession, client, LabelingState } from "../api/client";
import { pathDialog } from "../api/pathDialog";
import { humanLabelsCsvDefaultName, humanLabelsSaveDefaultPath, labelsFileBasename } from "../api/labelsCsvName";

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
Drag the frame slider to scrub previews.
Press 'm' to toggle machine label overlay.
Press 'n' to toggle bodypart names on the video.
Press 'h' to hide this help.
Press Esc to close (prompts to save).
Use Save to write labels; Close to exit.
Press Space to play or pause frames.
Press 'u' or Ctrl+Z to undo the last label (or clear active label on frame).`;

const PLAY_INTERVAL_MS = 66;

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
	const labelsPathRef = useRef(humanLabelsPath);
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const containerRef = useRef<HTMLDivElement>(null);
	const stageRef = useRef<HTMLDivElement>(null);
	const frameRef = useRef(0);
	const scrubRafRef = useRef<number | null>(null);
	const pendingPreviewFrameRef = useRef<number | null>(null);
	const previewBusyRef = useRef(false);
	const scrubbingRef = useRef(false);
	const previewGenRef = useRef(0);
	// Prevent Esc / double-click from starting a second close while the save dialog is open.
	const closingRef = useRef(false);
	const [scrubbing, setScrubbing] = useState(false);
	const playingRef = useRef(false);
	const playTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
	const frameCountRef = useRef(0);
	const gridSizeRef = useRef({ w: 0, h: 0 });
	const [playing, setPlaying] = useState(false);

	useEffect(() => {
		labelsPathRef.current = humanLabelsPath;
	}, [humanLabelsPath]);

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
		canvas.style.width = `${Math.floor(displayW * scale)}px`;
		canvas.style.height = `${Math.floor(displayH * scale)}px`;
	}, []);

	useEffect(() => {
		if (state?.grid_width && state?.grid_height) {
			gridSizeRef.current = { w: state.grid_width, h: state.grid_height };
			fitCanvasToStage();
		}
	}, [state?.grid_width, state?.grid_height, fitCanvasToStage]);

	const paintFrameBlob = useCallback(
		async (blob: Blob, gen: number) => {
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
				const gridW = gridSizeRef.current.w;
				const gridH = gridSizeRef.current.h;
				const ctx = canvas.getContext("2d");
				// Keep canvas internal size at native grid so scrub preview does not resize the view.
				if (gridW > 0 && gridH > 0 && ctx) {
					canvas.width = gridW;
					canvas.height = gridH;
					ctx.drawImage(bitmap, 0, 0, gridW, gridH);
				} else if (ctx) {
					canvas.width = bitmap.width;
					canvas.height = bitmap.height;
					ctx.drawImage(bitmap, 0, 0);
				}
				bitmap.close();
				fitCanvasToStage();
				setError(null);
			} catch (err) {
				if (gen !== previewGenRef.current) return;
				setError(err instanceof Error ? err.message : String(err));
			}
		},
		[fitCanvasToStage],
	);

	const fetchAndPaintFrame = useCallback(
		async (frameNumber: number, preview: boolean, gen: number) => {
			const blob = await client.fetchFrameJpeg(frameNumber, preview ? { preview: true } : undefined);
			if (gen !== previewGenRef.current) return;
			if (!(await isJpegBlob(blob))) {
				throw new Error(`Failed to load frame ${frameNumber}`);
			}
			await paintFrameBlob(blob, gen);
		},
		[paintFrameBlob],
	);

	const loadFrame = useCallback(
		async (frameNumber: number) => {
			const gen = ++previewGenRef.current;
			pendingPreviewFrameRef.current = null;
			const s = await client.setFrame(frameNumber);
			if (gen !== previewGenRef.current) return;
			frameRef.current = s.frame_number;
			setSliderFrame(s.frame_number);
			setState(s);
			await fetchAndPaintFrame(s.frame_number, false, gen);
		},
		[fetchAndPaintFrame],
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
			previewGenRef.current += 1;
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
			if (scrubRafRef.current != null) cancelAnimationFrame(scrubRafRef.current);
			if (playTimerRef.current != null) clearInterval(playTimerRef.current);
			playingRef.current = false;
		};
	}, [refresh]);

	useEffect(() => {
		frameCountRef.current = state?.frame_count ?? 0;
	}, [state?.frame_count]);

	useEffect(() => {
		containerRef.current?.focus();
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
				let savePath: string | undefined;
				if (save) {
					const existing = labelsPathRef.current ?? humanLabelsPath;
					if (existing) {
						savePath = existing;
					} else {
						const picked = await pathDialog.saveCsvForLabeler(
							humanLabelsSaveDefaultPath(existing, videoPaths),
						);
						savePath = picked ?? undefined;
					}
				}
				const session = await client.closeLabeler(save, savePath);
				onClose(session);
			} catch (e) {
				closingRef.current = false;
				setIsClosing(false);
				setError(e instanceof Error ? e.message : String(e));
			}
		},
		[humanLabelsPath, videoPaths, onClose, stopPlaying],
	);

	const saveLabels = useCallback(async () => {
		if (closingRef.current || isSaving) return;
		setIsSaving(true);
		setError(null);
		setSaveNotice(null);
		stopPlaying(false);
		try {
			let savePath: string | undefined;
			if (labelsPathRef.current) {
				savePath = labelsPathRef.current;
			} else {
				const picked = await pathDialog.saveCsvForLabeler(
					humanLabelsSaveDefaultPath(labelsPathRef.current, videoPaths),
				);
				if (!picked) return;
				savePath = picked;
			}
			const session = await client.saveLabeler(savePath);
			labelsPathRef.current = session.human_labels_path;
			onSessionUpdate(session);
			setSaveNotice(session.status_message ?? "Labels saved.");
		} catch (e) {
			setError(e instanceof Error ? e.message : String(e));
		} finally {
			setIsSaving(false);
		}
	}, [videoPaths, onSessionUpdate, stopPlaying, isSaving]);

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
	const humanLabelsName =
		labelsFileBasename(labelsPathRef.current ?? humanLabelsPath) ??
		humanLabelsCsvDefaultName(videoPaths);
	const machineLabelsName = labelsFileBasename(machineLabelsPath);

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
				{state.labeled_frame_list?.length > 0 && (
					<aside className="labeling-frame-list" aria-label="Human-labeled frames">
						<h3 className="labeling-hud-title">Labeled frames</h3>
						<ul className="labeling-frame-queue">
							{(state.labeled_frame_list ?? []).map((frame) => (
								<li key={frame}>
									<button
										type="button"
										className={
											frame === state.frame_number
												? "labeling-frame-btn labeling-frame-btn--active"
												: "labeling-frame-btn"
										}
										disabled={isClosing}
										onClick={() => jumpToLabeledFrame(frame)}
									>
										Frame {frame + 1}
									</button>
								</li>
							))}
						</ul>
					</aside>
				)}
				<div className="labeling-center">
					<p className="labeling-labeled-count">
						Labeled frames: {state.labeled_frames}
					</p>
					<div className="labeling-nav">
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
					<div className="labeling-stage" ref={stageRef}>
						<canvas
							ref={canvasRef}
							className="label-canvas"
							style={{ cursor: activeCursor }}
							onClick={onClick}
						/>
					</div>
					<div className="labeling-close-actions">
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
				</div>
				<aside className="labeling-hud" aria-label="Labeler info">
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
							{state.frame_number + 1} / {state.frame_count}
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
						{sliderFrame + 1} / {state.frame_count}
					</span>
				</div>
				{playing && (
					<p className="hint scrub-hint">Playing preview frames — pause to edit labels</p>
				)}
				{scrubbing && !playing && state.has_machine_labels && (
					<p className="hint scrub-hint">Machine predictions shown while scrubbing</p>
				)}
				{scrubbing && !playing && !state.has_machine_labels && (
					<p className="hint scrub-hint">Release slider to load full frame</p>
				)}
			</div>
		</div>
	);
}
