#!/usr/bin/env python3
"""Train a simple SVC classifier to predict categorical `recon` from selected features.

Usage: python train_classifier.py
Writes model to `ML_noise_model/svc_model.joblib` and metrics to `ML_noise_model/metrics_class.json`.
"""
from pathlib import Path
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)
import joblib


ROOT = Path(__file__).parent.parent
DATA_CSV = ROOT / "data_stats" / "data_stats.csv"
OUT_DIR = ROOT / "ML_noise_model"
OUT_DIR.mkdir(exist_ok=True)


def load_and_prepare(csv_path: Path):
    df = pd.read_csv(csv_path)

    # coerce recon to string categories and drop missing
    df = df[df["recon"].notna()].copy()
    # normalize recon values to strings and remove trailing .0 if present
    df["recon"] = df["recon"].astype(str).str.strip()
    df["recon"] = df["recon"].str.lower()

    # keep only the ordered recon levels we care about and set categorical dtype
    RECON_ORDER = ["100", "50", "20", "10", "4", "ref"]
    df = df[df["recon"].isin(RECON_ORDER)].copy()
    df["recon"] = pd.Categorical(df["recon"], categories=RECON_ORDER, ordered=True)

    # normalize tracer names a bit
    df["tracer"] = df["tracer"].replace(
        {
            "Fluorodeoxyglucose": "FDG",
            "GA68": "DOTA",
            "GA": "DOTA",
            "Solution": "Unknown",
        }
    )
    df["tracer"] = df["tracer"].fillna("Unknown")

    features = [
        "noise_metric",
        "scantime_pi",
        "activity",
        "tracer",
        "bmi",
        "weight",
        "scanner",
    ]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in CSV: {missing}")

    X = df[features].copy()
    y = df["recon"].copy()

    return X, y


def build_model():
    numeric_features = [
        "noise_metric",
        "scantime_pi",
        "bmi",
        "weight",
        "activity",
    ]
    categorical_features = ["tracer", "scanner"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = Pipeline(steps=[("pre", preprocessor), ("svc", SVC(probability=False))])
    return clf


if __name__ == "__main__":
    X, y = load_and_prepare(DATA_CSV)

    RECON_ORDER = ["100", "50", "20", "10", "4", "ref"]
    labels = RECON_ORDER

    # Determine number of splits: up to 10, limited by smallest class count
    min_class_count = y.value_counts().min()
    n_splits = min(10, int(min_class_count)) if min_class_count >= 2 else 2
    if n_splits < 2:
        n_splits = 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    agg_cm = np.zeros((len(labels), len(labels)), dtype=int)

    print(f"Running stratified {n_splits}-fold CV on {len(X)} samples")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = build_model()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        bal = balanced_accuracy_score(y_test, preds)

        cm = confusion_matrix(y_test, preds, labels=labels)
        agg_cm += cm

        fold_metrics.append(
            {
                "fold": fold_idx + 1,
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "balanced_accuracy": float(bal),
            }
        )
        print(f"Fold {fold_idx+1}: acc={acc:.4f}, f1_macro={f1:.4f}, bal_acc={bal:.4f}")

    metrics = {
        "cv_folds": n_splits,
        "accuracy_mean": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "f1_macro_mean": float(np.mean([m["f1_macro"] for m in fold_metrics])),
        "balanced_accuracy_mean": float(
            np.mean([m["balanced_accuracy"] for m in fold_metrics])
        ),
        "folds": fold_metrics,
        "confusion_matrix": agg_cm.tolist(),
        "labels": labels,
        "samples": len(X),
    }

    # Fit final model on all data and save
    final_model = build_model()
    final_model.fit(X, y)

    model_path = OUT_DIR / "svc_model.joblib"
    joblib.dump(final_model, model_path)
    (OUT_DIR / "metrics_class_cv.json").write_text(json.dumps(metrics, indent=2))

    print(f"Saved final classifier to {model_path}")
    print("CV Metrics:", metrics)
