import { useCallback, useEffect, useRef, useState } from "react";
import { AppSession, client, LabelingState } from "../api/client";
import { pathDialog } from "../api/pathDialog";
import { humanLabelsCsvDefaultName } from "../api/labelsCsvName";

interface Props {
	humanLabelsPath: string | null;
	videoPaths: string[] | null;
	onClose: (session: AppSession) => void;
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

async function isJpegBlob(blob: Blob): Promise<boolean> {
	const header = new Uint8Array(await blob.slice(0, 2).arrayBuffer());
	return header[0] === 0xff && header[1] === 0xd8;
}

export function LabelingCanvas({ humanLabelsPath, videoPaths, onClose }: Props) {
	const [state, setState] = useState<LabelingState | null>(null);
	const [sliderFrame, setSliderFrame] = useState(0);
	const [error, setError] = useState<string | null>(null);
	const [isClosing, setIsClosing] = useState(false);
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

	const fitCanvasToStage = useCallback(() => {
		const stage = stageRef.current;
		const canvas = canvasRef.current;
		if (!stage || !canvas || canvas.width <= 0 || canvas.height <= 0) return;

		const maxW = stage.clientWidth;
		const maxH = stage.clientHeight;
		if (maxW <= 0 || maxH <= 0) return;

		const scale = Math.min(maxW / canvas.width, maxH / canvas.height);
		canvas.style.width = `${Math.floor(canvas.width * scale)}px`;
		canvas.style.height = `${Math.floor(canvas.height * scale)}px`;
	}, []);

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
				canvas.width = bitmap.width;
				canvas.height = bitmap.height;
				const ctx = canvas.getContext("2d");
				if (ctx) ctx.drawImage(bitmap, 0, 0);
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
		};
	}, [refresh]);

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
			try {
				let savePath: string | undefined;
				if (save) {
					if (humanLabelsPath) {
						savePath = humanLabelsPath;
					} else {
						const picked = await pathDialog.saveCsvForLabeler(
							humanLabelsCsvDefaultName(videoPaths),
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
		[humanLabelsPath, videoPaths, onClose],
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
			if (key === "a" || key === "arrowleft") {
				e.preventDefault();
				const n = Math.max(0, frameRef.current - 1);
				loadFrame(n).catch((err) => {
					if (isIgnorableFetchError(err)) return;
					setError(String(err));
				});
				return;
			}
			if (key === "d" || key === "arrowright") {
				e.preventDefault();
				const n = Math.min(state.frame_count - 1, frameRef.current + 1);
				loadFrame(n).catch((err) => {
					if (isIgnorableFetchError(err)) return;
					setError(String(err));
				});
				return;
			}
			if (key === "m") {
				e.preventDefault();
				const gen = ++previewGenRef.current;
				client
					.toggleMachineOverlay()
					.then(async (s) => {
						if (gen !== previewGenRef.current) return;
						frameRef.current = s.frame_number;
						setState(s);
						await fetchAndPaintFrame(s.frame_number, false, gen);
					})
					.catch((err) => {
						if (isIgnorableFetchError(err)) return;
						setError(String(err));
					});
				return;
			}
			if (key === "h") {
				e.preventDefault();
				const gen = ++previewGenRef.current;
				client
					.toggleHelp()
					.then(async (s) => {
						if (gen !== previewGenRef.current) return;
						frameRef.current = s.frame_number;
						setState(s);
						await fetchAndPaintFrame(s.frame_number, false, gen);
					})
					.catch((err) => {
						if (isIgnorableFetchError(err)) return;
						setError(String(err));
					});
			}
		};
		window.addEventListener("keydown", onKey);
		return () => window.removeEventListener("keydown", onKey);
	}, [state, closeLabeler, loadFrame, fetchAndPaintFrame]);

	const onClick = async (e: React.MouseEvent<HTMLCanvasElement>) => {
		const canvas = canvasRef.current;
		if (!canvas || !state || closingRef.current) return;
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
		commitScrub(frameNumber);
	};

	if (!state) return <p>Loading labeler…</p>;

	return (
		<div
			className="labeling"
			ref={containerRef}
			tabIndex={0}
			onMouseDown={() => containerRef.current?.focus()}
		>
			<div className="labeling-toolbar">
				<span>
					Frame {state.frame_number + 1} / {state.frame_count}
				</span>
				<span>
					Active: <strong>{state.active_point}</strong>
				</span>
				<span>Labeled: {state.labeled_frames}</span>
				<div className="labeling-nav">
					<button
						type="button"
						disabled={state.frame_number <= 0 || isClosing}
						onClick={() =>
							loadFrame(state.frame_number - 1).catch((e) => {
								if (isIgnorableFetchError(e)) return;
								setError(String(e));
							})
						}
					>
						← Prev
					</button>
					<button
						type="button"
						disabled={state.frame_number >= state.frame_count - 1 || isClosing}
						onClick={() =>
							loadFrame(state.frame_number + 1).catch((e) => {
								if (isIgnorableFetchError(e)) return;
								setError(String(e));
							})
						}
					>
						Next →
					</button>
				</div>
				<button
					type="button"
					className="save-close-labeler"
					disabled={isClosing}
					onClick={() => void closeLabeler(true)}
				>
					Save &amp; Close
				</button>
				<button
					type="button"
					className="close-labeler"
					disabled={isClosing}
					onClick={() => {
						if (window.confirm("Close without saving labels?")) {
							void closeLabeler(false);
						}
					}}
				>
					Close without Saving
				</button>
				<span className="hint">
					a/d or ←/→ frames · scrub slider · m machine overlay · h help · Esc close
				</span>
			</div>
			{error && <div className="error">{error}</div>}
			{isClosing && <p className="hint">Saving and closing…</p>}
			<div className="labeling-stage" ref={stageRef}>
				<canvas ref={canvasRef} className="label-canvas" onClick={onClick} />
			</div>
			<div className={`frame-scrubber${scrubbing ? " frame-scrubber--scrubbing" : ""}`}>
				<label htmlFor="frame-slider">
					Frame {sliderFrame + 1} / {state.frame_count}
				</label>
				{scrubbing && state.has_machine_labels && (
					<p className="hint scrub-hint">Machine predictions shown while scrubbing</p>
				)}
				{scrubbing && !state.has_machine_labels && (
					<p className="hint scrub-hint">Release slider to load full frame</p>
				)}
				<input
					id="frame-slider"
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
			</div>
		</div>
	);
}
