"""Resampling, band-pass filtering, and spectral HR estimation."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float = 30.0):
    """Interpolate x(t) onto a uniform grid at fs Hz.

    t is per-frame timestamps (seconds, non-decreasing); x is (N,) or (N, C) with
    NaN marking dropped frames, which are interpolated across. Needed because
    webcam/dataset frame intervals vary, and filtering/FFT assume a steady rate.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("t must be 1-D with >= 2 samples")

    tu = np.arange(t[0], t[-1], 1.0 / fs)

    def _interp_col(col: np.ndarray) -> np.ndarray:
        good = np.isfinite(t) & np.isfinite(col)
        if good.sum() < 2:
            return np.full(tu.shape, np.nan)
        return np.interp(tu, t[good], col[good])

    if x.ndim == 1:
        xu = _interp_col(x)
    else:
        xu = np.column_stack([_interp_col(x[:, c]) for c in range(x.shape[1])])
    return tu, xu


def bandpass(x: np.ndarray, fs: float, lo: float = 0.7, hi: float = 4.0,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass (default 0.7-4.0 Hz = 42-240 bpm)."""
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def estimate_hr(x: np.ndarray, fs: float, lo: float = 0.7, hi: float = 4.0,
                nfft: int | None = None) -> tuple[float, float]:
    """Return (HR in bpm, SNR) for one window.

    HR is the largest spectral peak in [lo, hi]. SNR is the power at the
    fundamental and first harmonic (+/-0.1 Hz) over the rest of the band.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if not np.any(x):
        return float("nan"), 0.0
    win = x * np.hanning(len(x))

    if nfft is None:
        nfft = max(2048, 1 << int(np.ceil(np.log2(len(x) * 4))))  # zero-pad for resolution
    spec = np.abs(np.fft.rfft(win, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    band = (freqs >= lo) & (freqs <= hi)
    bf, bp = freqs[band], spec[band]
    if bp.size == 0 or not np.any(bp):
        return float("nan"), 0.0

    f0 = bf[int(np.argmax(bp))]
    sig_mask = (np.abs(bf - f0) <= 0.1) | (np.abs(bf - 2.0 * f0) <= 0.1)
    snr = float(bp[sig_mask].sum() / (bp[~sig_mask].sum() + 1e-12))
    return float(f0 * 60.0), snr
