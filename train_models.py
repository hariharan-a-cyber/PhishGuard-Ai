"""
PhishGuard AI - Model Training
==============================

Trains the detection ensemble (Random Forest + Gradient Boosting) on top of
the shared feature pipeline in `phishguard/features.py`, so training and
inference are guaranteed to use identical features.

Data sources, in priority order:
  1. --csv PATH            : your own CSV (columns: subject, sender, content,
                             urls, label) - or a Kaggle CSV; columns are
                             auto-detected.
  2. --kaggle SLUG         : download a Kaggle dataset via kagglehub.
  3. built-in generator    : realistic synthetic corpus (default, fully offline).

Examples:
  python train_models.py                          # offline synthetic data
  python train_models.py --csv data/my_emails.csv
  python train_models.py --kaggle ethancratchley/email-phishing-dataset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phishguard.features import (FEATURE_NAMES, extract_features,  # noqa: E402
                                 features_to_vector)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DEFAULT_CSV = os.path.join(HERE, "data", "phishing_emails.csv")

# Candidate column names for auto-detecting arbitrary CSV / Kaggle schemas.
COL_CANDIDATES = {
    "subject": ["subject", "email_subject", "title"],
    "sender": ["sender", "from", "sender_email", "email_from"],
    "content": ["content", "body", "text", "message", "email_text", "email_body"],
    "urls": ["urls", "url", "link", "links"],
    "label": ["label", "class", "phishing", "is_phishing", "classification",
              "result", "type", "target"],
}


def banner(msg: str) -> None:
    print("\n" + "=" * 64)
    print(msg)
    print("=" * 64)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _match_column(df: pd.DataFrame, names) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in lower:
            return lower[n]
    return None


def load_dataframe(args) -> pd.DataFrame:
    if args.kaggle:
        banner(f"Downloading Kaggle dataset: {args.kaggle}")
        try:
            import kagglehub
            path = kagglehub.dataset_download(args.kaggle)
            csvs = [f for f in os.listdir(path) if f.endswith(".csv")]
            if not csvs:
                raise RuntimeError("no CSV in downloaded dataset")
            df = pd.read_csv(os.path.join(path, csvs[0]))
            print(f"Loaded {len(df)} rows from {csvs[0]}")
            return df
        except Exception as exc:  # noqa: BLE001
            print(f"Kaggle download failed ({exc}); using synthetic data.")

    csv_path = args.csv or DEFAULT_CSV
    if not os.path.exists(csv_path):
        banner("Generating synthetic dataset (offline)")
        from data.generate_dataset import generate
        generate(args.rows, DEFAULT_CSV)
        csv_path = DEFAULT_CSV

    banner(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def build_xy(df: pd.DataFrame):
    cols = {k: _match_column(df, v) for k, v in COL_CANDIDATES.items()}
    print(f"Column mapping: {cols}")
    label_col = cols["label"]
    if label_col is None:
        # Some Kaggle datasets ship pre-engineered numeric features + a label.
        # If we can't find raw text, try to use numeric columns directly.
        raise SystemExit("Could not find a label column. Provide a CSV with a "
                         "'label' column (1=phishing, 0=legitimate).")

    rows = []
    for _, r in df.iterrows():
        email = {
            "subject": str(r[cols["subject"]]) if cols["subject"] else "",
            "sender": str(r[cols["sender"]]) if cols["sender"] else "",
            "content": str(r[cols["content"]]) if cols["content"] else "",
            "urls": str(r[cols["urls"]]) if cols["urls"] else "",
        }
        rows.append(features_to_vector(extract_features(email)))

    X = np.array(rows, dtype=float)

    y = df[label_col]
    if y.dtype == object:
        y = (y.astype(str).str.lower()
             .map({"phishing": 1, "phish": 1, "spam": 1, "1": 1, "true": 1,
                   "legitimate": 0, "ham": 0, "safe": 0, "0": 0, "false": 0}))
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int).to_numpy()
    return X, y


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args) -> None:
    df = load_dataframe(args)
    X, y = build_xy(df)
    banner("Feature matrix")
    print(f"Samples: {len(X)} | Features: {X.shape[1]}")
    print(f"Legitimate (0): {int((y == 0).sum())} | "
          f"Phishing (1): {int((y == 1).sum())}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    banner("Training Random Forest")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)

    banner("Training Gradient Boosting")
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    gb.fit(X_tr_s, y_tr)

    # ----- evaluation (ensemble = mean of the two probabilities) -----
    def evaluate(name, proba):
        pred = (proba >= 0.5).astype(int)
        print(f"\n{name}")
        print(f"  Accuracy : {accuracy_score(y_te, pred):.4f}")
        print(f"  Precision: {precision_score(y_te, pred, zero_division=0):.4f}")
        print(f"  Recall   : {recall_score(y_te, pred, zero_division=0):.4f}")
        print(f"  F1-score : {f1_score(y_te, pred, zero_division=0):.4f}")
        try:
            print(f"  ROC-AUC  : {roc_auc_score(y_te, proba):.4f}")
        except ValueError:
            pass
        return {
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred, zero_division=0),
            "f1": f1_score(y_te, pred, zero_division=0),
        }

    banner("Evaluation (held-out test set)")
    rf_proba = rf.predict_proba(X_te_s)[:, 1]
    gb_proba = gb.predict_proba(X_te_s)[:, 1]
    ens_proba = (rf_proba + gb_proba) / 2.0
    m_rf = evaluate("Random Forest", rf_proba)
    m_gb = evaluate("Gradient Boosting", gb_proba)
    m_ens = evaluate("Ensemble (mean)", ens_proba)

    print("\nClassification report (ensemble):")
    print(classification_report(y_te, (ens_proba >= 0.5).astype(int),
                                target_names=["legitimate", "phishing"],
                                zero_division=0))

    cv = cross_val_score(rf, scaler.transform(X), y, cv=5, scoring="f1")
    print(f"5-fold CV F1 (RF): mean={cv.mean():.4f} std={cv.std():.4f}")

    # ----- feature importances -----
    importances = sorted(
        zip(FEATURE_NAMES, rf.feature_importances_),
        key=lambda t: t[1], reverse=True)
    print("\nTop 10 features by importance:")
    for name, imp in importances[:10]:
        print(f"  {name:<26} {imp:.4f}")

    # ----- persist -----
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(rf, os.path.join(MODELS_DIR, "rf_model.pkl"))
    joblib.dump(gb, os.path.join(MODELS_DIR, "gb_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    metadata = {
        "version": "2.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "feature_names": FEATURE_NAMES,
        "models": ["RandomForest", "GradientBoosting"],
        "metrics": {
            "random_forest": {k: round(v, 4) for k, v in m_rf.items()},
            "gradient_boosting": {k: round(v, 4) for k, v in m_gb.items()},
            "ensemble": {k: round(v, 4) for k, v in m_ens.items()},
            "cv_f1_mean": round(float(cv.mean()), 4),
        },
        "top_features": [{"name": n, "importance": round(float(i), 4)}
                         for n, i in importances[:10]],
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    banner("Training complete")
    print(f"Ensemble accuracy: {m_ens['accuracy']:.2%} | "
          f"F1: {m_ens['f1']:.2%}")
    print(f"Models saved to: {MODELS_DIR}/")
    print("Run `python app.py` to start the web app with ML detection.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train PhishGuard AI models")
    ap.add_argument("--csv", help="path to a labelled CSV")
    ap.add_argument("--kaggle", help="Kaggle dataset slug to download")
    ap.add_argument("--rows", type=int, default=6000,
                    help="rows to generate if using synthetic data")
    train(ap.parse_args())
