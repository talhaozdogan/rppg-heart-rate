"""Live webcam rPPG demo (OpenCV).

Shows the webcam feed with a forehead/cheek ROI overlay, a scrolling pulse
waveform, the current bpm and a signal-quality indicator. Displays
"Calibrating..." until the first 10 s window is filled. A recording mode saves
raw video + per-frame timestamps so sessions can be re-analysed offline (Stage 3).

Run:
    python -m app.live_demo                       # webcam
    python -m app.live_demo --record              # webcam + save session
    python -m app.live_demo --source clip.avi     # replay a video file
    python -m app.live_demo --source data/subject35/vid.avi --headless  # no window

Keys:  q / Esc = quit    r = toggle recording
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.face_roi import REGIONS, FaceROIExtractor
from src.stream import RollingHRProcessor

WAVE_H = 160                       # height of the waveform panel (px)
GREEN, YELLOW, RED, WHITE = (0, 220, 0), (0, 215, 255), (0, 0, 255), (240, 240, 240)


def quality_label(state: str, snr: float) -> tuple[str, tuple[int, int, int]]:
    if state == "calibrating":
        return "calibrating", YELLOW
    if state == "low_signal":
        return "no face / low signal", RED
    if snr >= 2.0:
        return "good", GREEN
    if snr >= 0.7:
        return "fair", YELLOW
    return "poor", RED


def draw_roi(frame: np.ndarray, landmarks) -> None:
    """Translucent fill + outline of the forehead and cheek ROIs."""
    if landmarks is None:
        return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for region in REGIONS:
        pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)]
                        for i in region], dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(overlay, hull, GREEN)
        cv2.polylines(frame, [hull], True, GREEN, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)


def draw_waveform(pulse: np.ndarray, w: int) -> np.ndarray:
    panel = np.zeros((WAVE_H, w, 3), dtype=np.uint8)
    cv2.putText(panel, "pulse", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    if pulse.size >= 2 and np.ptp(pulse) > 0:
        y = pulse - pulse.mean()
        y = y / (np.max(np.abs(y)) + 1e-9)
        mid, amp = WAVE_H // 2, WAVE_H // 2 - 12
        xs = np.linspace(0, w - 1, pulse.size).astype(int)
        ys = (mid - y * amp).astype(int)
        cv2.polylines(panel, [np.column_stack([xs, ys])], False, GREEN, 2, cv2.LINE_AA)
    return panel


def render(frame: np.ndarray, landmarks, res: dict, recording: bool,
           rec_elapsed: float) -> np.ndarray:
    frame = frame.copy()
    draw_roi(frame, landmarks)
    h, w = frame.shape[:2]
    label, color = quality_label(res["state"], res["snr"])

    if res["state"] == "calibrating":
        msg = f"Calibrating... {res['progress'] * 100:.0f}%"
        cv2.putText(frame, msg, (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, YELLOW, 2, cv2.LINE_AA)
        bpm_txt = "-- BPM"
    else:
        bpm_txt = f"{res['hr']:.0f} BPM" if np.isfinite(res["hr"]) else "-- BPM"

    cv2.putText(frame, bpm_txt, (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
    cv2.putText(frame, f"quality: {label}", (14, 76), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)
    if res["state"] == "ok":
        cv2.putText(frame, f"SNR {res['snr']:.2f}", (14, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, WHITE, 1, cv2.LINE_AA)

    if recording:
        cv2.circle(frame, (w - 120, 30), 8, RED, -1)
        cv2.putText(frame, f"REC {rec_elapsed:4.0f}s", (w - 100, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)

    cv2.putText(frame, "q quit   r record", (14, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, WHITE, 1, cv2.LINE_AA)
    return np.vstack([frame, draw_waveform(res["pulse"], w)])


class SessionRecorder:
    """Saves raw video (lossless FFV1 if available) + per-frame timestamps CSV."""

    def __init__(self, out_dir: Path, w: int, h: int, fps: float) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.dir = out_dir
        self.t0: float | None = None
        for fourcc in ("FFV1", "MJPG"):
            self.writer = cv2.VideoWriter(
                str(out_dir / "vid.avi"), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if self.writer.isOpened():
                self.codec = fourcc
                break
        self._ts = open(out_dir / "timestamps.csv", "w", newline="")
        self._csv = csv.writer(self._ts)
        self._csv.writerow(["frame", "t_sec"])
        self._i = 0

    def write(self, frame: np.ndarray, t: float) -> None:
        if self.t0 is None:
            self.t0 = t
        self.writer.write(frame)
        self._csv.writerow([self._i, f"{t - self.t0:.6f}"])
        self._i += 1

    def close(self) -> str:
        self.writer.release()
        self._ts.close()
        return str(self.dir)


def print_environment_tips() -> None:
    print(
        "\n=== rPPG live demo — tips for a clean signal ===\n"
        "Lighting: steady, bright DAYLIGHT works best. Avoid flickering/cheap LED\n"
        "  bulbs — their PWM flicker aliases into the heart-rate band and corrupts it.\n"
        "Stay relatively still; hold the face centred and well-lit.\n"
        "\nWebcam auto-adjustments fight the pulse signal. Disable them if you can:\n"
        "  macOS: built-in/Continuity cameras expose no manual controls. Use an\n"
        "    EXTERNAL UVC webcam plus a tool like 'Webcam Settings' (App Store) to\n"
        "    LOCK exposure and white balance. We also try to disable them via OpenCV\n"
        "    below, but AVFoundation often ignores it.\n"
        "===============================================\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Live rPPG webcam demo.")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", default=None, help="video file to replay instead of webcam")
    ap.add_argument("--model", default="models/face_landmarker.task")
    ap.add_argument("--fs", type=float, default=30.0)
    ap.add_argument("--record", action="store_true", help="start recording immediately")
    ap.add_argument("--out", default="recordings")
    ap.add_argument("--headless", action="store_true", help="no window (smoke test)")
    ap.add_argument("--headless-frames", type=int, default=450)
    ap.add_argument("--preview", default=None, help="save one composite frame here (headless)")
    args = ap.parse_args()

    print_environment_tips()

    cap = cv2.VideoCapture(args.source if args.source else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open video source.")
    if not args.source:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # best-effort; may no-op on macOS
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    extractor = FaceROIExtractor(args.model)
    proc = RollingHRProcessor(fs=args.fs)
    recorder: SessionRecorder | None = None
    t0 = time.perf_counter()
    last_ms, frame_idx = -1.0, 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # timestamps: container time for files, wall clock for webcam
            if args.source:
                t_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or frame_idx / args.fs * 1000.0
            else:
                t_ms = (time.perf_counter() - t0) * 1000.0
            t_ms = max(t_ms, last_ms + 1.0)
            last_ms = t_ms

            fr, landmarks = extractor.extract_detailed(frame, int(round(t_ms)))
            rgb = (fr.r, fr.g, fr.b) if fr.face_found else None
            proc.add(t_ms / 1000.0, rgb)
            res = proc.update()

            if (args.record and recorder is None):
                recorder = SessionRecorder(
                    Path(args.out) / datetime.now().strftime("session_%Y%m%d_%H%M%S"),
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1],
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0],
                    args.fs)
            if recorder is not None:
                recorder.write(frame, t_ms / 1000.0)

            rec_elapsed = (t_ms / 1000.0 - recorder.t0) if recorder and recorder.t0 else 0.0
            canvas = render(frame, landmarks, res, recorder is not None, rec_elapsed)
            frame_idx += 1

            if args.headless:
                if frame_idx >= args.headless_frames:
                    if args.preview:
                        cv2.imwrite(args.preview, canvas)
                    print(f"[headless] processed {frame_idx} frames; "
                          f"state={res['state']} hr={res['hr']:.1f} snr={res['snr']:.2f}")
                    break
                continue

            cv2.imshow("rPPG live demo", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                if recorder is None:
                    recorder = SessionRecorder(
                        Path(args.out) / datetime.now().strftime("session_%Y%m%d_%H%M%S"),
                        frame.shape[1], frame.shape[0], args.fs)
                    print(f"Recording -> {recorder.dir} (codec {recorder.codec})")
                else:
                    print(f"Saved session to {recorder.close()}")
                    recorder = None
    finally:
        if recorder is not None:
            print(f"Saved session to {recorder.close()}")
        cap.release()
        extractor.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
