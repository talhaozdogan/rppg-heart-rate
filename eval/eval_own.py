"""Stage 3 evaluation on my own recordings vs. a manual-pulse-count reference.

Re-analyses every recorded session offline (using its saved per-frame
timestamps), matches each manual pulse-count reading to the app's HR at the same
moment, and reports MAE per condition plus one estimated-vs-reference timeline
plot for a representative session.

Reference log: a CSV (see eval/reference_template.csv) with columns
    session_id, condition, ref_hr_bpm, t_sec, notes
where session_id is a folder under --recordings, and t_sec is seconds into the
recording when the count was taken (blank -> compared to the session median).

Usage
-----
    python -m eval.eval_own --recordings recordings --ref recordings/reference.csv
    python -m eval.eval_own ... --plot-session session_20261012_140000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline import process_recording

RESULTS = Path(__file__).resolve().parents[1] / "results"


def match_estimate(res, t_sec: float) -> float:
    """App HR nearest to t_sec (session-relative); session median if t_sec is NaN."""
    hr = np.asarray(res.hr, dtype=float)
    if hr.size == 0 or np.all(~np.isfinite(hr)):
        return float("nan")
    if not np.isfinite(t_sec):
        return float(np.nanmedian(hr))
    # hr_t are window-end times in session seconds; res.t[0] is the session start.
    t_rel = np.asarray(res.hr_t) - res.t[0]
    return float(hr[int(np.argmin(np.abs(t_rel - t_sec)))])


def timeline_plot(res, readings: pd.DataFrame, session_id: str, path: str) -> None:
    t_rel = np.asarray(res.hr_t) - res.t[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_rel, res.hr, "-", color="C0", label="rPPG estimate")
    for _, row in readings.iterrows():
        ts = row["t_sec"] if np.isfinite(row["t_sec"]) else t_rel[-1] / 2
        ax.scatter(ts, row["ref_hr_bpm"], color="C3", zorder=5)
    ax.scatter([], [], color="C3", label="manual count")  # legend proxy
    ax.set_xlabel("time into session (s)")
    ax.set_ylabel("HR (bpm)")
    ax.set_title("rPPG estimate vs manual pulse count (representative resting session)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate rPPG on own recordings.")
    ap.add_argument("--recordings", default="recordings")
    ap.add_argument("--ref", default="recordings/reference.csv")
    ap.add_argument("--fs", type=float, default=30.0)
    ap.add_argument("--plot-session", default=None, help="session_id for the timeline plot")
    args = ap.parse_args()

    rec_root = Path(args.recordings)
    df = pd.read_csv(args.ref, comment="#", skipinitialspace=True)
    if "t_sec" not in df.columns:
        df["t_sec"] = np.nan
    df["t_sec"] = pd.to_numeric(df["t_sec"], errors="coerce")

    RESULTS.mkdir(exist_ok=True)
    cache: dict[str, object] = {}
    records = []  # (condition, ref, est, session_id)

    for session_id, grp in df.groupby("session_id"):
        session_dir = rec_root / str(session_id)
        if not (session_dir / "vid.avi").exists():
            print(f"  {session_id}: SKIP (no recording at {session_dir})")
            continue
        print(f"  processing {session_id} ...")
        res = process_recording(str(session_dir), fs=args.fs)
        cache[str(session_id)] = res
        for _, row in grp.iterrows():
            est = match_estimate(res, row["t_sec"])
            records.append((row["condition"], float(row["ref_hr_bpm"]), est, str(session_id)))
            print(f"    {row['condition']:<14} ref={row['ref_hr_bpm']:.0f}  est={est:.1f} bpm")

    if not records:
        raise SystemExit("No sessions matched the reference log. Record some first.")

    res_df = pd.DataFrame(records, columns=["condition", "ref", "est", "session_id"])
    res_df["abs_err"] = (res_df["est"] - res_df["ref"]).abs()

    # Per-condition + overall MAE table
    rows = ["| Condition | N | MAE (bpm) | mean ref | mean est |",
            "|---|---|---|---|---|"]
    for cond, g in res_df.groupby("condition"):
        rows.append(f"| {cond} | {len(g)} | {g['abs_err'].mean():.2f} | "
                    f"{g['ref'].mean():.1f} | {g['est'].mean():.1f} |")
    rows.append(f"| **overall** | {len(res_df)} | **{res_df['abs_err'].mean():.2f}** | "
                f"{res_df['ref'].mean():.1f} | {res_df['est'].mean():.1f} |")
    table = "\n".join(rows) + "\n"
    (RESULTS / "own_results.md").write_text(table)
    print("\n" + table)

    # Timeline plot for a representative session (arg, else most-sampled session)
    plot_id = args.plot_session or res_df["session_id"].value_counts().idxmax()
    if plot_id in cache:
        timeline_plot(cache[plot_id], df[df["session_id"] == plot_id],
                      plot_id, str(RESULTS / "own_timeline.png"))
        print(f"Timeline plot ({plot_id}) + table written to {RESULTS}/")


if __name__ == "__main__":
    main()
