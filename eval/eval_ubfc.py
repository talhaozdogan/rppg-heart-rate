"""Stage 1 validation on UBFC-rPPG.

Processes every subject with the shared pipeline, computes ground-truth HR from
the contact-PPG trace using the IDENTICAL resample -> bandpass -> sliding-window
spectral method, and reports MAE / RMSE / Pearson r per subject and overall,
plus scatter and Bland-Altman plots.

IMPORTANT: UBFC ships in two layouts (DATASET_1 'gtdump.xmp' vs DATASET_2 3-line
'ground_truth.txt'). We do NOT assume — `--inspect` prints the real format first.

Usage
-----
    python -m eval.eval_ubfc --data /path/to/UBFC --inspect        # look first
    python -m eval.eval_ubfc --data /path/to/UBFC [--limit N]      # full run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.pipeline import hr_series_from_signal, process_video
from src.signal_utils import bandpass, resample_uniform
from eval.metrics import (
    align_series,
    bland_altman_plot,
    compute_metrics,
    scatter_plot,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def find_subjects(root: Path) -> list[Path]:
    """Return subject folders that contain a video + a ground-truth file."""
    subjects = []
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            has_vid = any((p / v).exists() for v in ("vid.avi", "vid.mp4"))
            has_gt = any((p / g).exists() for g in ("ground_truth.txt", "gtdump.xmp"))
            if has_vid and has_gt:
                subjects.append(p)
    return subjects


def _video_path(subject: Path) -> Path:
    for v in ("vid.avi", "vid.mp4"):
        if (subject / v).exists():
            return subject / v
    raise FileNotFoundError(f"no video in {subject}")


def _gt_path(subject: Path) -> Path:
    for g in ("ground_truth.txt", "gtdump.xmp"):
        if (subject / g).exists():
            return subject / g
    raise FileNotFoundError(f"no ground truth in {subject}")


def inspect_ground_truth(subject: Path) -> None:
    """Print a raw preview + parsed shape so we confirm the format, not assume it."""
    gt = _gt_path(subject)
    raw = gt.read_text(errors="replace")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    print(f"\n[inspect] {gt}")
    print(f"  size={gt.stat().st_size} bytes, non-empty lines={len(lines)}")
    for i, ln in enumerate(lines[:4]):
        toks = ln.split()
        preview = " ".join(toks[:6])
        print(f"  line {i}: {len(toks)} tokens | {preview} ...")


def load_ground_truth(subject: Path):
    """Load (ppg, times_seconds) from the UBFC ground-truth file.

    DATASET_2 'ground_truth.txt': 3 whitespace lines -> [PPG, HR, timestamps(s)].
    Raises with a preview for any other layout so we can adapt deliberately.
    """
    gt = _gt_path(subject)
    lines = [ln for ln in gt.read_text().splitlines() if ln.strip()]
    if gt.name == "ground_truth.txt" and len(lines) >= 3:
        ppg = np.fromstring(lines[0], sep=" ")
        times = np.fromstring(lines[2], sep=" ")
        if ppg.size and times.size == ppg.size:
            return ppg, times
    raise ValueError(
        f"Unrecognised ground-truth layout in {gt} "
        f"({len(lines)} lines). Run with --inspect and adapt load_ground_truth()."
    )


def evaluate_subject(subject: Path, fs: float, win_sec: float, hop_sec: float):
    """Return (est_hr, gt_hr) paired arrays for one subject, or None on failure."""
    # Camera side (full pipeline).
    res = process_video(str(_video_path(subject)), fs=fs,
                        win_sec=win_sec, hop_sec=hop_sec)

    # Ground-truth side: SAME resample -> bandpass -> sliding spectral method.
    ppg, gt_times = load_ground_truth(subject)
    gt_t, gt_ppg = resample_uniform(gt_times - gt_times[0], ppg, fs)
    gt_pulse = bandpass(gt_ppg, fs)
    gt_hr_t, gt_hr, _ = hr_series_from_signal(gt_pulse, fs, gt_t, win_sec, hop_sec)

    est, ref = align_series(res.hr_t - res.hr_t[0], res.hr,
                            gt_hr_t - gt_hr_t[0], gt_hr)
    if est.size == 0:
        return None
    return est, ref, res.n_missing, res.n_frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate rPPG pipeline on UBFC-rPPG.")
    ap.add_argument("--data", required=True, help="UBFC dataset root")
    ap.add_argument("--fs", type=float, default=30.0)
    ap.add_argument("--win", type=float, default=10.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None, help="process first N subjects")
    ap.add_argument("--inspect", action="store_true", help="print GT format and exit")
    args = ap.parse_args()

    root = Path(args.data).expanduser()
    subjects = find_subjects(root)
    if not subjects:
        raise SystemExit(f"No subject folders with video+ground truth under {root}")
    print(f"Found {len(subjects)} subjects under {root}")

    if args.inspect:
        inspect_ground_truth(subjects[0])
        return

    if args.limit:
        subjects = subjects[: args.limit]

    RESULTS.mkdir(exist_ok=True)
    rows = ["| Subject | N | MAE | RMSE | r |", "|---|---|---|---|---|"]
    all_est, all_ref = [], []

    for subject in subjects:
        try:
            out = evaluate_subject(subject, args.fs, args.win, args.hop)
        except Exception as e:  # keep going; report honestly
            print(f"  {subject.name}: FAILED ({e})")
            rows.append(f"| {subject.name} | - | FAILED | | |")
            continue
        if out is None:
            print(f"  {subject.name}: no overlapping HR windows")
            continue
        est, ref, n_missing, n_frames = out
        m = compute_metrics(est, ref)
        all_est.append(est); all_ref.append(ref)
        print(f"  {subject.name}: MAE={m.mae:.2f} RMSE={m.rmse:.2f} r={m.r:.3f} "
              f"(missing {n_missing}/{n_frames} frames)")
        rows.append(m.row(subject.name))

    if not all_est:
        raise SystemExit("No subjects produced results.")

    est = np.concatenate(all_est)
    ref = np.concatenate(all_ref)
    overall = compute_metrics(est, ref)
    rows.append(overall.row("**OVERALL**"))

    scatter_plot(est, ref, str(RESULTS / "ubfc_scatter.png"),
                 "UBFC-rPPG: estimated vs ground truth")
    bland_altman_plot(est, ref, str(RESULTS / "ubfc_bland_altman.png"),
                      "UBFC-rPPG: Bland-Altman")
    table = "\n".join(rows) + "\n"
    (RESULTS / "ubfc_results.md").write_text(table)

    print("\n" + table)
    print(f"OVERALL  MAE={overall.mae:.2f}  RMSE={overall.rmse:.2f}  r={overall.r:.3f}")
    print(f"Plots + table written to {RESULTS}/")


if __name__ == "__main__":
    main()
