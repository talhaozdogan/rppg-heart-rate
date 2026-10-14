"""Online, rolling-buffer HR estimation for the live app.

Buffers the recent per-frame RGB and runs the same resample -> POS -> band-pass ->
spectral chain over the latest window on each update(), so the live readout matches
the offline pipeline. update() returns state 'calibrating' (not enough data yet),
'low_signal' (too many missing faces), or 'ok'.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from src.pos import pos
from src.signal_utils import bandpass, estimate_hr, resample_uniform


class RollingHRProcessor:
    def __init__(self, fs: float = 30.0, win_sec: float = 10.0,
                 display_sec: float = 6.0, lo: float = 0.7, hi: float = 4.0,
                 smooth: int = 5, min_face_frac: float = 0.5) -> None:
        self.fs = fs
        self.win_sec = win_sec
        self.display_sec = display_sec
        self.lo, self.hi = lo, hi
        self.min_face_frac = min_face_frac
        self.keep_sec = win_sec + 2.0
        self._ts: deque[float] = deque()
        self._rgb: deque[tuple[float, float, float]] = deque()
        self._hr_hist: deque[float] = deque(maxlen=smooth)

    def add(self, t: float, rgb: tuple[float, float, float] | None) -> None:
        """Append one frame's timestamp (s) and mean RGB (None if no face)."""
        self._ts.append(t)
        self._rgb.append(rgb if rgb is not None else (np.nan, np.nan, np.nan))
        while len(self._ts) >= 2 and (self._ts[-1] - self._ts[0]) > self.keep_sec:
            self._ts.popleft()
            self._rgb.popleft()

    def span(self) -> float:
        return (self._ts[-1] - self._ts[0]) if len(self._ts) >= 2 else 0.0

    def update(self) -> dict:
        span = self.span()
        if span < self.win_sec:
            return {"state": "calibrating", "progress": min(1.0, span / self.win_sec),
                    "hr": float("nan"), "snr": 0.0, "pulse": np.zeros(0)}

        t = np.asarray(self._ts, dtype=float)
        rgb = np.asarray(self._rgb, dtype=float)
        face_frac = float(np.mean(np.isfinite(rgb).all(axis=1)))
        if face_frac < self.min_face_frac:
            return {"state": "low_signal", "progress": 1.0,
                    "hr": float("nan"), "snr": 0.0, "pulse": np.zeros(0)}

        _, rgbu = resample_uniform(t, rgb, self.fs)
        pulse = bandpass(pos(rgbu, self.fs), self.fs)

        win = int(round(self.win_sec * self.fs))
        seg = pulse[-win:]
        if not np.all(np.isfinite(seg)):
            return {"state": "low_signal", "progress": 1.0,
                    "hr": float("nan"), "snr": 0.0, "pulse": np.zeros(0)}

        hr, snr = estimate_hr(seg, self.fs, self.lo, self.hi)
        self._hr_hist.append(hr)
        hr_disp = float(np.nanmedian(self._hr_hist))  # light median smoothing
        disp = pulse[-int(round(self.display_sec * self.fs)):]
        return {"state": "ok", "progress": 1.0, "hr": hr_disp, "snr": snr, "pulse": disp}
