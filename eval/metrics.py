"""Shared HR-comparison metrics and plots (used by eval_ubfc and eval_own)."""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # headless; we save PNGs, no GUI
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Metrics:
    mae: float
    rmse: float
    r: float          # Pearson correlation
    n: int

    def row(self, name: str) -> str:
        return f"| {name} | {self.n} | {self.mae:.2f} | {self.rmse:.2f} | {self.r:.3f} |"


def compute_metrics(est: np.ndarray, ref: np.ndarray) -> Metrics:
    """MAE, RMSE, Pearson r over paired HR samples (NaNs dropped)."""
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    ok = np.isfinite(est) & np.isfinite(ref)
    est, ref = est[ok], ref[ok]
    if est.size < 2:
        return Metrics(np.nan, np.nan, np.nan, int(est.size))
    err = est - ref
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r = float(np.corrcoef(est, ref)[0, 1]) if np.std(est) and np.std(ref) else np.nan
    return Metrics(mae, rmse, r, int(est.size))


def align_series(t_a: np.ndarray, a: np.ndarray, t_b: np.ndarray, b: np.ndarray,
                 tol: float = 0.75):
    """Pair samples of (t_a, a) and (t_b, b) by nearest timestamp within `tol` s."""
    t_a, a, t_b, b = map(np.asarray, (t_a, a, t_b, b))
    pa, pb = [], []
    for ta, va in zip(t_a, a):
        j = int(np.argmin(np.abs(t_b - ta)))
        if abs(t_b[j] - ta) <= tol:
            pa.append(va)
            pb.append(b[j])
    return np.asarray(pa), np.asarray(pb)


def scatter_plot(est: np.ndarray, ref: np.ndarray, path: str, title: str) -> None:
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    ok = np.isfinite(est) & np.isfinite(ref)
    est, ref = est[ok], ref[ok]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ref, est, s=12, alpha=0.5)
    lo = min(est.min(), ref.min()) - 5
    hi = max(est.max(), ref.max()) + 5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Ground-truth HR (bpm)")
    ax.set_ylabel("Estimated HR (bpm)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def bland_altman_plot(est: np.ndarray, ref: np.ndarray, path: str, title: str) -> None:
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    ok = np.isfinite(est) & np.isfinite(ref)
    est, ref = est[ok], ref[ok]
    mean = (est + ref) / 2
    diff = est - ref
    md, sd = np.mean(diff), np.std(diff)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mean, diff, s=12, alpha=0.5)
    ax.axhline(md, color="k", lw=1, label=f"mean {md:+.2f}")
    ax.axhline(md + 1.96 * sd, color="r", ls="--", lw=1, label=f"+1.96 SD {md + 1.96 * sd:+.2f}")
    ax.axhline(md - 1.96 * sd, color="r", ls="--", lw=1, label=f"-1.96 SD {md - 1.96 * sd:+.2f}")
    ax.set_xlabel("Mean of estimated & GT HR (bpm)")
    ax.set_ylabel("Estimated - GT (bpm)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
