#!/usr/bin/env python3
"""Compare several classifiers using stratified CV and aggregate metrics.

Saves results to `ML_noise_model/compare_models_results.json`.
"""
from pathlib import Path
import json
import importlib

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.base import clone, is_regressor
import joblib


ROOT = Path(__file__).parent.parent
DATA_CSV = ROOT / "data_stats" / "data_stats.csv"
OUT_DIR = ROOT / "ML_noise_model"
OUT_DIR.mkdir(exist_ok=True)


RECON_ORDER = ["100", "50", "20", "10", "4", "ref"]


def load_data():
    df = pd.read_csv(DATA_CSV)
    df = df[df["recon"].notna()].copy()
    df["recon"] = df["recon"].astype(str).str.strip()
    df["recon"] = df["recon"].str.replace(r"\.0+$", "", regex=True)
    df["recon"] = df["recon"].str.lower()
    df = df[df["recon"].isin(RECON_ORDER)].copy()
    df["recon"] = pd.Categorical(df["recon"], categories=RECON_ORDER, ordered=True)

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


def make_pipeline_for_model(model):
    numeric_features = ["noise_metric", "scantime_pi", "bmi", "weight", "activity"]
    categorical_features = ["tracer"]
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
    pipe = Pipeline(steps=[("pre", preprocessor), ("est", model)])
    return pipe


def try_import_xgboost():
    try:
        xgb = importlib.import_module("xgboost")
        return True
    except Exception:
        return False


def compare(models_to_run, n_splits=10):
    X, y = load_data()
    labels = RECON_ORDER

    # encode categorical labels to integer codes for estimators that expect numeric classes
    # y is a pandas.Categorical with categories in RECON_ORDER, so codes align with that order
    y_codes = y.cat.codes
    label_map = {int(k): v for k, v in enumerate(RECON_ORDER)}

    min_class = y.value_counts().min()
    n_splits = min(n_splits, int(min_class)) if min_class >= 2 else 2
    if n_splits < 2:
        n_splits = 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}

    for name, model in models_to_run.items():
        print(f"Evaluating {name}")
        fold_metrics = []
        agg_cm = np.zeros((len(labels), len(labels)), dtype=int)

        pipe = make_pipeline_for_model(model)

        for train_idx, test_idx in skf.split(X, y_codes):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_codes.iloc[train_idx], y_codes.iloc[test_idx]

            m = clone(pipe)
            m.fit(X_train, y_train)
            preds = m.predict(X_test)

            # If the estimator is a regressor (ordinal baseline), round predictions to nearest class code
            if is_regressor(model):
                preds = np.rint(preds).astype(int)
                preds = np.clip(preds, 0, len(labels) - 1)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro")
            bal = balanced_accuracy_score(y_test, preds)

            cm = confusion_matrix(y_test, preds, labels=list(range(len(labels))))
            agg_cm += cm

            fold_metrics.append(
                {
                    "accuracy": float(acc),
                    "f1_macro": float(f1),
                    "balanced_accuracy": float(bal),
                }
            )

        results[name] = {
            "n_splits": n_splits,
            "accuracy_mean": float(np.mean([f["accuracy"] for f in fold_metrics])),
            "f1_macro_mean": float(np.mean([f["f1_macro"] for f in fold_metrics])),
            "balanced_accuracy_mean": float(
                np.mean([f["balanced_accuracy"] for f in fold_metrics])
            ),
            "folds": fold_metrics,
            "confusion_matrix": agg_cm.tolist(),
            "labels": labels,
            "label_map": label_map,
        }

    out_path = OUT_DIR / "compare_models_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved comparison results to {out_path}")
    return results


if __name__ == "__main__":
    models = {
        "LogisticRegression": LogisticRegression(
            multi_class="multinomial",
            solver="saga",
            max_iter=2000,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200),
    }

    if try_import_xgboost():
        import xgboost as xgb

        models["XGBoost"] = xgb.XGBClassifier(
            use_label_encoder=False, eval_metric="mlogloss"
        )
    else:
        print("xgboost not available; skipping XGBoost model")

    # Ordinal baseline: train regressor on encoded labels and round to nearest category index
    # We'll wrap a RandomForestRegressor and postprocess predictions
    models["Ordinal_RF_regressor"] = RandomForestRegressor(n_estimators=300, n_jobs=-1)

    compare(models)
