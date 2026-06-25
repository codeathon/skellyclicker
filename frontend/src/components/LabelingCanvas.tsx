import { useCallback, useEffect, useRef, useState } from "react";
import { AppSession, client, LabelingState } from "../api/client";
import { pathDialog } from "../api/pathDialog";

interface Props {
	humanLabelsPath: string | null;
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

export function LabelingCanvas({ humanLabelsPath, onClose }: Props) {
	const [state, setState] = useState<LabelingState | null>(null);
	const [imgSrc, setImgSrc] = useState("");
	const [sliderFrame, setSliderFrame] = useState(0);
	const [error, setError] = useState<string | null>(null);
	const [isClosing, setIsClosing] = useState(false);
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const containerRef = useRef<HTMLDivElement>(null);
	const stageRef = useRef<HTMLDivElement>(null);
	const frameRef = useRef(0);
	const previewBlobRef = useRef<string | null>(null);
	const scrubRafRef = useRef<number | null>(null);
	const pendingPreviewFrameRef = useRef<number | null>(null);
	const previewBusyRef = useRef(false);
	const scrubbingRef = useRef(false);
	const previewGenRef = useRef(0);
	// Prevent Esc / double-click from starting a second close while the save dialog is open.
	const closingRef = useRef(false);
	const [scrubbing, setScrubbing] = useState(false);

	const revokePreviewBlob = useCallback(() => {
		if (previewBlobRef.current) {
			URL.revokeObjectURL(previewBlobRef.current);
			previewBlobRef.current = null;
		}
	}, []);

	const showFrameBlob = useCallback((blob: Blob, gen: number) => {
		if (gen !== previewGenRef.current) return;
		revokePreviewBlob();
		const url = URL.createObjectURL(blob);
		previewBlobRef.current = url;
		setImgSrc(url);
	}, [revokePreviewBlob]);

	const loadFrame = useCallback(async (frameNumber: number) => {
		previewGenRef.current += 1;
		pendingPreviewFrameRef.current = null;
		const s = await client.setFrame(frameNumber);
		frameRef.current = s.frame_number;
		setSliderFrame(s.frame_number);
		setState(s);
		const blob = await client.fetchFrameJpeg(s.frame_number);
		if (!(await isJpegBlob(blob))) {
			throw new Error(`Failed to load frame ${s.frame_number}`);
		}
		revokePreviewBlob();
		const url = URL.createObjectURL(blob);
		previewBlobRef.current = url;
		setImgSrc(url);
	}, [revokePreviewBlob]);

	const drainPreviewQueue = useCallback(async () => {
		if (previewBusyRef.current) return;
		previewBusyRef.current = true;
		try {
			while (pendingPreviewFrameRef.current != null && scrubbingRef.current) {
				const frameNumber = pendingPreviewFrameRef.current;
				pendingPreviewFrameRef.current = null;
				const gen = ++previewGenRef.current;
				try {
					const blob = await client.fetchFrameJpeg(frameNumber, { preview: true });
					if (!scrubbingRef.current || gen !== previewGenRef.current) continue;
					if (!(await isJpegBlob(blob))) continue;
					frameRef.current = frameNumber;
					showFrameBlob(blob, gen);
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
	}, [showFrameBlob]);

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
		const s = await client.labelingState();
		frameRef.current = s.frame_number;
		setState(s);
		const blob = await client.fetchFrameJpeg(s.frame_number);
		if (!(await isJpegBlob(blob))) {
			throw new Error(`Failed to load frame ${s.frame_number}`);
		}
		revokePreviewBlob();
		const url = URL.createObjectURL(blob);
		previewBlobRef.current = url;
		setImgSrc(url);
	}, [revokePreviewBlob]);

	useEffect(() => {
		refresh().catch((e) => {
			if (!isIgnorableFetchError(e)) setError(String(e));
		});
		return () => {
			previewGenRef.current += 1;
			pendingPreviewFrameRef.current = null;
			if (scrubRafRef.current != null) cancelAnimationFrame(scrubRafRef.current);
			revokePreviewBlob();
		};
	}, [refresh, revokePreviewBlob]);

	useEffect(() => {
		containerRef.current?.focus();
	}, [state?.session_id]);

	useEffect(() => {
		const img = document.getElementById("label-img") as HTMLImageElement | null;
		if (!img || !imgSrc) return;
		img.src = imgSrc;
	}, [imgSrc]);

	const closeLabeler = useCallback(
		async (save: boolean) => {
			if (closingRef.current) return;
			closingRef.current = true;
			setIsClosing(true);
			setError(null);
			try {
				let savePath: string | undefined;
				if (save) {
					// Re-save to the same CSV when updating labels — skips a slow file dialog round-trip.
					if (humanLabelsPath) {
						savePath = humanLabelsPath;
					} else {
						const picked = await pathDialog.saveCsvForLabeler();
						// Cancelled save dialog → default path under the video folder on the server.
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
		[humanLabelsPath, onClose],
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
				client
					.toggleMachineOverlay()
					.then(async (s) => {
						frameRef.current = s.frame_number;
						setState(s);
						const blob = await client.fetchFrameJpeg(s.frame_number);
						if (!(await isJpegBlob(blob))) {
							throw new Error(`Failed to load frame ${s.frame_number}`);
						}
						revokePreviewBlob();
						const url = URL.createObjectURL(blob);
						previewBlobRef.current = url;
						setImgSrc(url);
					})
					.catch((err) => {
						if (isIgnorableFetchError(err)) return;
						setError(String(err));
					});
			}
		};
		window.addEventListener("keydown", onKey);
		return () => window.removeEventListener("keydown", onKey);
	}, [state, closeLabeler, loadFrame]);

	const fitCanvasToStage = useCallback(() => {
		const stage = stageRef.current;
		const canvas = canvasRef.current;
		const img = document.getElementById("label-img") as HTMLImageElement | null;
		if (!stage || !canvas || !img?.naturalWidth || !img.naturalHeight) return;

		const maxW = stage.clientWidth;
		const maxH = stage.clientHeight;
		if (maxW <= 0 || maxH <= 0) return;

		const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight);
		canvas.style.width = `${Math.floor(img.naturalWidth * scale)}px`;
		canvas.style.height = `${Math.floor(img.naturalHeight * scale)}px`;
	}, []);

	useEffect(() => {
		const stage = stageRef.current;
		if (!stage) return;
		const observer = new ResizeObserver(() => fitCanvasToStage());
		observer.observe(stage);
		return () => observer.disconnect();
	}, [fitCanvasToStage, state?.session_id]);

	const onImageLoad = () => {
		const canvas = canvasRef.current;
		const img = document.getElementById("label-img") as HTMLImageElement;
		if (!canvas || !img) return;
		canvas.width = img.naturalWidth;
		canvas.height = img.naturalHeight;
		const ctx = canvas.getContext("2d");
		if (ctx) ctx.drawImage(img, 0, 0);
		fitCanvasToStage();
	};

	const onImageError = () => {
		// Ignore stale blob URLs while scrubbing or when a newer preview superseded this one.
		if (scrubbingRef.current) return;
		setError("Failed to load frame image");
	};

	const onClick = async (e: React.MouseEvent<HTMLCanvasElement>) => {
		const canvas = canvasRef.current;
		if (!canvas || !state || closingRef.current) return;
		const rect = canvas.getBoundingClientRect();
		const scaleX = canvas.width / rect.width;
		const scaleY = canvas.height / rect.height;
		const x = Math.round((e.clientX - rect.left) * scaleX);
		const y = Math.round((e.clientY - rect.top) * scaleY);
		try {
			const s = await client.click(x, y);
			frameRef.current = s.frame_number;
			setState(s);
			const blob = await client.fetchFrameJpeg(s.frame_number);
			if (!(await isJpegBlob(blob))) {
				throw new Error(`Failed to load frame ${s.frame_number}`);
			}
			revokePreviewBlob();
			const url = URL.createObjectURL(blob);
			previewBlobRef.current = url;
			setImgSrc(url);
		} catch (err) {
			if (isIgnorableFetchError(err)) return;
			setError(err instanceof Error ? err.message : String(err));
		}
	};

	const onScrubStart = () => {
		scrubbingRef.current = true;
		setScrubbing(true);
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
				<span className="hint">a/d or ←/→ frames · drag slider to scrub · m overlay · Esc close</span>
			</div>
			{error && <div className="error">{error}</div>}
			{isClosing && <p className="hint">Saving and closing…</p>}
			<img
				id="label-img"
				src={imgSrc}
				alt=""
				hidden
				onLoad={onImageLoad}
				onError={onImageError}
			/>
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
