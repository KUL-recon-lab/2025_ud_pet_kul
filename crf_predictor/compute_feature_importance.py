#!/usr/bin/env python3
"""Compute permutation feature importance for the saved SVC model.

Saves results to `ML_noise_model/feature_importances.json`.
"""
from pathlib import Path
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import joblib


ROOT = Path(__file__).parent.parent
DATA_CSV = ROOT / "data_stats" / "data_stats.csv"
MODEL_PATH = ROOT / "ML_noise_model" / "svc_model.joblib"
OUT_PATH = ROOT / "ML_noise_model" / "feature_importances.json"


def load_data():
    df = pd.read_csv(DATA_CSV)
    df = df[df["recon"].notna()].copy()
    df["recon"] = df["recon"].astype(str).str.strip()
    df["recon"] = df["recon"].str.replace(r"\.0+$", "", regex=True)
    df["recon"] = df["recon"].str.lower()

    RECON_ORDER = ["100", "50", "20", "10", "4"]
    df = df[df["recon"].isin(RECON_ORDER)].copy()

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
    X = df[features].copy()
    y = df["recon"].copy()
    return X, y


def main():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Train the classifier first."
        )

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)

    # permutation_importance works with estimator.predict; pass raw X_test and the pipeline will preprocess
    r = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1
    )

    feature_names = []
    # for the pipeline we passed raw features; if the pipeline expands features (one-hot), permutation_importance
    # returns importances aligned to the pipeline's input columns, so we can map directly to original features
    feature_names = X_test.columns.tolist()

    importances = {
        name: float(imp) for name, imp in zip(feature_names, r["importances_mean"])
    }

    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    OUT_PATH.write_text(json.dumps({"importances": sorted_importances}, indent=2))
    print("Saved feature importances to", OUT_PATH)
    print(sorted_importances)


if __name__ == "__main__":
    main()
