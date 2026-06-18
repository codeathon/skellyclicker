import { useCallback, useEffect, useRef, useState } from "react";
import { client, LabelingState } from "../api/client";

interface Props {
  onClose: (saved: boolean) => void;
}

export function LabelingCanvas({ onClose }: Props) {
  const [state, setState] = useState<LabelingState | null>(null);
  const [imgSrc, setImgSrc] = useState("");
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const refresh = useCallback(async (frame?: number) => {
    const s = await client.labelingState();
    const f = frame ?? s.frame_number;
    setState(s);
    setImgSrc(client.frameUrl(f));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    const img = document.getElementById("label-img") as HTMLImageElement | null;
    if (!img || !imgSrc) return;
    img.src = imgSrc;
  }, [imgSrc]);

  useEffect(() => {
    const onKey = async (e: KeyboardEvent) => {
      if (!state) return;
      if (e.key === "Escape") {
        const save = window.confirm("Save labels before closing?");
        await client.closeLabeler(save);
        onClose(save);
        return;
      }
      if (e.key === "a") {
        const n = Math.max(0, state.frame_number - 1);
        const s = await client.setFrame(n);
        setState(s);
        setImgSrc(client.frameUrl(n));
      }
      if (e.key === "d") {
        const n = Math.min(state.frame_count - 1, state.frame_number + 1);
        const s = await client.setFrame(n);
        setState(s);
        setImgSrc(client.frameUrl(n));
      }
      if (e.key === "m") {
        const s = await client.toggleMachineOverlay();
        setState(s);
        setImgSrc(client.frameUrl(state.frame_number));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [state, onClose]);

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
    if (!canvas || !state) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);
    const s = await client.click(x, y);
    setState(s);
    setImgSrc(client.frameUrl(state.frame_number));
  };

  if (!state) return <p>Loading labeler…</p>;

  return (
    <div className="labeling">
      <div className="labeling-toolbar">
        <span>
          Frame {state.frame_number + 1} / {state.frame_count}
        </span>
        <span>Active: <strong>{state.active_point}</strong></span>
        <span>Labeled: {state.labeled_frames}</span>
        <span className="hint">a/d frames · m overlay · Esc close</span>
      </div>
      <img id="label-img" src={imgSrc} alt="" hidden onLoad={onImageLoad} />
      <canvas ref={canvasRef} className="label-canvas" onClick={onClick} />
    </div>
  );
}
