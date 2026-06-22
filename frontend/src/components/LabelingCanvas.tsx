import { useCallback, useEffect, useRef, useState } from "react";
import { AppSession, client, LabelingState } from "../api/client";
import { pathDialog } from "../api/pathDialog";

interface Props {
	onClose: (session: AppSession) => void;
}

export function LabelingCanvas({ onClose }: Props) {
	const [state, setState] = useState<LabelingState | null>(null);
	const [imgSrc, setImgSrc] = useState("");
	const [sliderFrame, setSliderFrame] = useState(0);
	const [error, setError] = useState<string | null>(null);
	const [isClosing, setIsClosing] = useState(false);
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const containerRef = useRef<HTMLDivElement>(null);
	const frameRef = useRef(0);
	const sliderDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	// Prevent Esc / double-click from starting a second close while the save dialog is open.
	const closingRef = useRef(false);

	const loadFrame = useCallback(async (frameNumber: number) => {
		const s = await client.setFrame(frameNumber);
		frameRef.current = s.frame_number;
		setSliderFrame(s.frame_number);
		setState(s);
		setImgSrc(client.frameUrl(s.frame_number));
	}, []);

	const refresh = useCallback(async () => {
		const s = await client.labelingState();
		frameRef.current = s.frame_number;
		setState(s);
		setImgSrc(client.frameUrl(s.frame_number));
	}, []);

	useEffect(() => {
		refresh().catch((e) => setError(String(e)));
	}, [refresh]);

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
					const picked = await pathDialog.saveCsvForLabeler();
					// Cancelled save dialog → default path under the video folder on the server.
					savePath = picked ?? undefined;
				}
				const session = await client.closeLabeler(save, savePath);
				onClose(session);
			} catch (e) {
				closingRef.current = false;
				setIsClosing(false);
				setError(e instanceof Error ? e.message : String(e));
			}
		},
		[onClose],
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
				loadFrame(n).catch((err) => setError(String(err)));
				return;
			}
			if (key === "d" || key === "arrowright") {
				e.preventDefault();
				const n = Math.min(state.frame_count - 1, frameRef.current + 1);
				loadFrame(n).catch((err) => setError(String(err)));
				return;
			}
			if (key === "m") {
				e.preventDefault();
				client
					.toggleMachineOverlay()
					.then((s) => {
						frameRef.current = s.frame_number;
						setState(s);
						setImgSrc(client.frameUrl(s.frame_number));
					})
					.catch((err) => setError(String(err)));
			}
		};
		window.addEventListener("keydown", onKey);
		return () => window.removeEventListener("keydown", onKey);
	}, [state, closeLabeler, loadFrame]);

	const onImageLoad = () => {
		const canvas = canvasRef.current;
		const img = document.getElementById("label-img") as HTMLImageElement;
		if (!canvas || !img) return;
		canvas.width = img.naturalWidth;
		canvas.height = img.naturalHeight;
		const ctx = canvas.getContext("2d");
		if (ctx) ctx.drawImage(img, 0, 0);
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
			setImgSrc(client.frameUrl(s.frame_number));
		} catch (err) {
			setError(err instanceof Error ? err.message : String(err));
		}
	};

	const onSliderInput = (frameNumber: number) => {
		setSliderFrame(frameNumber);
		if (sliderDebounceRef.current) clearTimeout(sliderDebounceRef.current);
		sliderDebounceRef.current = setTimeout(() => {
			loadFrame(frameNumber).catch((err) => setError(String(err)));
		}, 120);
	};

	const onSliderCommit = (frameNumber: number) => {
		if (sliderDebounceRef.current) clearTimeout(sliderDebounceRef.current);
		loadFrame(frameNumber).catch((err) => setError(String(err)));
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
						onClick={() => loadFrame(state.frame_number - 1).catch((e) => setError(String(e)))}
					>
						← Prev
					</button>
					<button
						type="button"
						disabled={state.frame_number >= state.frame_count - 1 || isClosing}
						onClick={() => loadFrame(state.frame_number + 1).catch((e) => setError(String(e)))}
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
				<span className="hint">a/d or ←/→ frames · m overlay · Esc close</span>
			</div>
			{error && <div className="error">{error}</div>}
			{isClosing && <p className="hint">Saving and closing…</p>}
			<img id="label-img" src={imgSrc} alt="" hidden onLoad={onImageLoad} />
			<canvas ref={canvasRef} className="label-canvas" onClick={onClick} />
			<div className="frame-scrubber">
				<label htmlFor="frame-slider">
					Frame {state.frame_number + 1} / {state.frame_count}
				</label>
				<input
					id="frame-slider"
					type="range"
					min={0}
					max={Math.max(0, state.frame_count - 1)}
					value={sliderFrame}
					disabled={isClosing}
					onInput={(e) => onSliderInput(Number(e.currentTarget.value))}
					onChange={(e) => onSliderInput(Number(e.currentTarget.value))}
					onMouseUp={(e) => onSliderCommit(Number(e.currentTarget.value))}
					onTouchEnd={(e) => onSliderCommit(Number(e.currentTarget.value))}
				/>
			</div>
		</div>
	);
}
