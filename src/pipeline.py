"""End-to-end pipeline: video (or frame stream) -> HR time series.

Chain: FaceROIExtractor -> resample_uniform -> pos -> bandpass
       -> sliding-window estimate_hr -> (time, hr_bpm, snr) series.

Shared by eval_ubfc, eval_own, and (online variant) the live app, so both
sides of every comparison are processed identically.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.face_roi import FaceROIExtractor
from src.pos import pos
from src.signal_utils import bandpass, estimate_hr, resample_uniform


@dataclass
class PipelineResult:
    t: np.ndarray            # uniform time grid (s)
    rgb: np.ndarray          # (N, 3) resampled RGB
    pulse: np.ndarray        # band-passed POS pulse
    hr_t: np.ndarray         # window end-times (s)
    hr: np.ndarray           # HR per window (bpm)
    snr: np.ndarray          # quality per window
    fs: float
    n_missing: int           # frames with no face
    n_frames: int
    raw_t: np.ndarray = field(repr=False, default=None)  # original timestamps (s)


def hr_series_from_signal(signal: np.ndarray, fs: float, t: np.ndarray,
                          win_sec: float = 10.0, hop_sec: float = 1.0):
    """Sliding-window spectral HR over a 1-D signal (used for both camera and GT)."""
    win, hop = int(round(win_sec * fs)), int(round(hop_sec * fs))
    hr_t, hr, snr = [], [], []
    for start in range(0, len(signal) - win + 1, hop):
        seg = signal[start:start + win]
        h, s = estimate_hr(seg, fs)
        hr_t.append(t[start + win - 1])
        hr.append(h)
        snr.append(s)
    return np.asarray(hr_t), np.asarray(hr), np.asarray(snr)


def process_rgb(raw_t: np.ndarray, raw_rgb: np.ndarray, fs: float = 30.0,
                win_sec: float = 10.0, hop_sec: float = 1.0) -> PipelineResult:
    """Run resample -> POS -> bandpass -> sliding-window HR on raw RGB traces."""
    t, rgb = resample_uniform(raw_t, raw_rgb, fs)
    pulse = bandpass(pos(rgb, fs), fs)
    hr_t, hr, snr = hr_series_from_signal(pulse, fs, t, win_sec, hop_sec)
    n_missing = int(np.sum(~np.isfinite(raw_rgb).all(axis=1)))
    return PipelineResult(
        t=t, rgb=rgb, pulse=pulse, hr_t=hr_t, hr=hr, snr=snr, fs=fs,
        n_missing=n_missing, n_frames=len(raw_t), raw_t=np.asarray(raw_t),
    )


def process_video(path: str, fs: float = 30.0,
                  model_path: str = "models/face_landmarker.task",
                  win_sec: float = 10.0, hop_sec: float = 1.0,
                  max_frames: int | None = None) -> PipelineResult:
    """Process a video file into an HR time series.

    Timestamps come from the container (CAP_PROP_POS_MSEC), falling back to
    frame_index / fps, forced strictly increasing for the VIDEO-mode landmarker.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    fps_meta = cap.get(cv2.CAP_PROP_FPS) or fs

    extractor = FaceROIExtractor(model_path)
    traces: list = []
    raw_t: list[float] = []
    frame_idx = 0
    last_ms = -1.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if not t_ms or t_ms <= 0:
                t_ms = frame_idx / fps_meta * 1000.0
            t_ms = max(t_ms, last_ms + 1.0)   # enforce strictly increasing
            last_ms = t_ms
            fr = extractor.extract(frame, int(round(t_ms)))
            traces.append([fr.r, fr.g, fr.b])
            raw_t.append(t_ms / 1000.0)
            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break
    finally:
        cap.release()
        extractor.close()

    return process_rgb(np.asarray(raw_t), np.asarray(traces), fs, win_sec, hop_sec)


def process_recording(session_dir: str, fs: float = 30.0,
                      model_path: str = "models/face_landmarker.task",
                      win_sec: float = 10.0, hop_sec: float = 1.0) -> PipelineResult:
    """Re-analyse a recorded session (vid.avi + timestamps.csv) offline, using the
    saved per-frame timestamps so webcam timing matches what was captured."""
    session = Path(session_dir)
    times: list[float] = []
    with open(session / "timestamps.csv", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header: frame,t_sec
        for row in reader:
            if row:
                times.append(float(row[1]))
    times_arr = np.asarray(times, dtype=float)

    cap = cv2.VideoCapture(str(session / "vid.avi"))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open {session / 'vid.avi'}")
    extractor = FaceROIExtractor(model_path)
    traces: list = []
    raw_t: list[float] = []
    last_ms, i = -1.0, 0
    try:
        while i < len(times_arr):
            ok, frame = cap.read()
            if not ok:
                break
            t_ms = max(times_arr[i] * 1000.0, last_ms + 1.0)
            last_ms = t_ms
            fr = extractor.extract(frame, int(round(t_ms)))
            traces.append([fr.r, fr.g, fr.b])
            raw_t.append(times_arr[i])
            i += 1
    finally:
        cap.release()
        extractor.close()

    return process_rgb(np.asarray(raw_t), np.asarray(traces), fs, win_sec, hop_sec)
