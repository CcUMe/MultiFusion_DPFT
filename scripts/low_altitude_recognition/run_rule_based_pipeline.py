from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the low-altitude radar-only traditional recognition pipeline end to end.")
    parser.add_argument("--cap-dir", type=Path, required=True, help="Capture directory with image_to_antframe_time_aligned.csv and mmwave_ra_npy")
    parser.add_argument("--candidate-dir", type=Path, help="Output candidate directory. Defaults to <cap-dir>/recognition_candidates")
    parser.add_argument("--candidate-source", choices=["fusion", "protocol", "rae"], default="fusion")
    parser.add_argument("--protocol-root", type=Path, help="Protocol parse output root for protocol candidate source")
    parser.add_argument("--max-scans", type=int, default=0)
    parser.add_argument("--min-area-px", type=int, default=40)
    parser.add_argument("--threshold-percentile", type=float, default=99.2)
    parser.add_argument("--min-peak-db", type=float)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--max-candidates-per-scan", type=int, default=12)
    parser.add_argument("--min-confidence", type=float, default=0.18)
    parser.add_argument("--topk-per-scan", type=int, default=12)
    return parser.parse_args()


def run_step(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    cap_dir = args.cap_dir.resolve()
    candidate_dir = args.candidate_dir.resolve() if args.candidate_dir else cap_dir / "recognition_candidates"
    feature_csv = candidate_dir / "features.csv"
    pred_csv = candidate_dir / "predictions.csv"
    radar_overlay_dir = candidate_dir / "radar_overlays"
    structured_dir = candidate_dir / "structured_results"

    build_cmd = [
        sys.executable,
        "scripts/low_altitude_recognition/build_candidates.py",
        "--cap-dir", str(cap_dir),
        "--out-dir", str(candidate_dir),
        "--candidate-source", str(args.candidate_source),
        "--max-scans", str(args.max_scans),
        "--min-area-px", str(args.min_area_px),
        "--threshold-percentile", str(args.threshold_percentile),
        "--smooth-sigma", str(args.smooth_sigma),
        "--max-candidates-per-scan", str(args.max_candidates_per_scan),
    ]
    if args.min_peak_db is not None:
        build_cmd.extend(["--min-peak-db", str(args.min_peak_db)])
    if args.protocol_root is not None:
        build_cmd.extend(["--protocol-root", str(args.protocol_root.resolve())])
    run_step(build_cmd)

    run_step([
        sys.executable,
        "scripts/low_altitude_recognition/extract_features.py",
        "--candidate-dir", str(candidate_dir),
        "--out-csv", str(feature_csv),
    ])

    run_step([
        sys.executable,
        "scripts/low_altitude_recognition/rule_based_recognition.py",
        "--feature-csv", str(feature_csv),
        "--out-csv", str(pred_csv),
    ])

    run_step([
        sys.executable,
        "scripts/low_altitude_recognition/visualize_radar_predictions.py",
        "--prediction-csv", str(pred_csv),
        "--out-dir", str(radar_overlay_dir),
        "--cap-dir", str(cap_dir),
        "--min-confidence", str(args.min_confidence),
        "--topk-per-scan", str(args.topk_per_scan),
    ])

    run_step([
        sys.executable,
        "scripts/low_altitude_recognition/export_structured_results.py",
        "--prediction-csv", str(pred_csv),
        "--cap-dir", str(cap_dir),
        "--overlay-dir", str(radar_overlay_dir),
        "--out-dir", str(structured_dir),
    ])

    print(f"candidate dir: {candidate_dir}")
    print(f"features: {feature_csv}")
    print(f"predictions: {pred_csv}")
    print(f"radar overlays: {radar_overlay_dir}")
    print(f"structured results: {structured_dir}")


if __name__ == "__main__":
    main()
