from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from common import TARGET_CLASSES, load_csv_rows, numeric_feature_columns, safe_float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a traditional classifier for low-altitude target recognition.")
    parser.add_argument("--feature-csv", type=Path, required=True, help="Labeled feature CSV path")
    parser.add_argument("--model-out", type=Path, required=True, help="Output model path")
    parser.add_argument("--label-column", default="label", help="Ground-truth label column name")
    parser.add_argument("--n-estimators", type=int, default=300, help="RandomForest tree count")
    parser.add_argument("--max-depth", type=int, default=18, help="RandomForest max depth")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def rows_to_matrix(rows: list[dict[str, str]], feature_names: list[str], label_column: str) -> tuple[np.ndarray, list[str], list[str]]:
    x = np.asarray([[safe_float(row.get(name), 0.0) for name in feature_names] for row in rows], dtype=np.float32)
    y = [row[label_column] for row in rows]
    ids = [row.get("candidate_id", f"row_{i}") for i, row in enumerate(rows)]
    return x, y, ids


def main() -> None:
    args = parse_args()
    rows = load_csv_rows(args.feature_csv.resolve())
    rows = [r for r in rows if r.get(args.label_column)]
    if not rows:
        raise RuntimeError(f"no labeled rows found in {args.feature_csv}")

    feature_names = numeric_feature_columns(rows, args.label_column)
    if not feature_names:
        raise RuntimeError("no numeric feature columns found")

    x, y_raw, ids = rows_to_matrix(rows, feature_names, args.label_column)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )),
    ])

    if len(rows) >= max(20, len(le.classes_) * 4):
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_val)
        report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
        print(classification_report(y_val, y_pred, target_names=le.classes_, zero_division=0))
    else:
        clf.fit(x, y)
        report = {"note": "dataset too small for reliable validation split; trained on all labeled rows"}
        print("[WARN] dataset too small for reliable validation split; trained on all labeled rows")

    bundle = {
        "model": clf,
        "feature_names": feature_names,
        "classes": list(le.classes_),
        "label_encoder": le,
        "label_column": args.label_column,
        "target_classes": TARGET_CLASSES,
        "report": report,
    }
    args.model_out.resolve().parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out.resolve())
    with args.model_out.resolve().with_suffix(args.model_out.suffix + ".json").open("w", encoding="utf-8") as f:
        json.dump({
            "feature_csv": str(args.feature_csv.resolve()),
            "model_out": str(args.model_out.resolve()),
            "feature_count": len(feature_names),
            "train_rows": len(rows),
            "classes": list(le.classes_),
            "report": report,
        }, f, ensure_ascii=False, indent=2)

    print(f"feature csv: {args.feature_csv.resolve()}")
    print(f"model out: {args.model_out.resolve()}")
    print(f"label column: {args.label_column}")
    print(f"training rows: {len(rows)}")
    print(f"feature count: {len(feature_names)}")
    print(f"classes: {list(le.classes_)}")


if __name__ == "__main__":
    main()
