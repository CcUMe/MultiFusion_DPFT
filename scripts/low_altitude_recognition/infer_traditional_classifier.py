from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from common import load_csv_rows, safe_float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run traditional target recognition inference on low-altitude data.")
    parser.add_argument("--feature-csv", type=Path, required=True, help="Feature CSV path")
    parser.add_argument("--model", type=Path, required=True, help="Trained model path")
    parser.add_argument("--out-csv", type=Path, required=True, help="Prediction output CSV path")
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("no rows to write")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model.resolve())
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    classes = list(bundle["classes"])

    rows = load_csv_rows(args.feature_csv.resolve())
    if not rows:
        raise RuntimeError(f"no feature rows found in {args.feature_csv}")

    x = np.asarray([[safe_float(row.get(name), 0.0) for name in feature_names] for row in rows], dtype=np.float32)
    pred_idx = model.predict(x)
    pred_prob = model.predict_proba(x)

    out_rows: list[dict[str, str]] = []
    for row, idx, probs in zip(rows, pred_idx, pred_prob):
        out = dict(row)
        out["pred_label"] = classes[int(idx)]
        out["pred_confidence"] = f"{float(np.max(probs)):.6f}"
        for cls_name, cls_prob in zip(classes, probs):
            key = "prob_" + cls_name.lower().replace(' ', '_')
            out[key] = f"{float(cls_prob):.6f}"
        out_rows.append(out)

    write_rows(args.out_csv.resolve(), out_rows)
    print(f"feature csv: {args.feature_csv.resolve()}")
    print(f"model: {args.model.resolve()}")
    print(f"prediction out: {args.out_csv.resolve()}")
    print(f"predicted rows: {len(out_rows)}")


if __name__ == "__main__":
    main()
