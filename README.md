# 🛡️ PhishGuard AI

**An AI/ML-based phishing-detection system** that analyses an email's text,
sender and URLs and returns a calibrated risk verdict in real time — served
through a clean security-console web dashboard and a JSON REST API.

PhishGuard AI was originally built as a **first-year hackathon project under my
leadership of a 4-member team**. This repository is a full rebuild and
modernisation of that project: the original core idea (an ensemble ML model
over engineered email features, backed by a Flask app with a reports/alerts
database) is preserved, while the broken pieces have been fixed, the feature
pipeline has been unified, the models have been properly trained, and the UI
has been rebuilt.

---

## ✨ Highlights

- **27-feature engineered pipeline** covering URLs (IP literals, shorteners,
  punycode, suspicious TLDs, look-alike/digit-substituted domains, subdomain
  depth, domain entropy), text (urgency, credential/money terms, ALL-CAPS
  ratio), sender reputation and subject-line signals.
- **Ensemble model** — Random Forest + Gradient Boosting, averaged
  probabilities — with an automatic **heuristic fallback** if no trained model
  is present, so the app is never dead on arrival.
- **Single source of truth for features.** Training and inference import the
  *same* `phishguard/features.py`, which eliminates the train/serve feature
  mismatch that silently broke the original project.
- **Trains fully offline.** A realistic synthetic-email generator means you can
  build a working model with no Kaggle account and no internet — but you can
  also point it at your own CSV or a Kaggle dataset.
- **Explainable verdicts.** Every result ships with human-readable reasons
  (e.g. *"Link points to a raw IP address"*, *"Sender domain uses a suspicious
  TLD"*) and a list of sender issues.
- **Web dashboard + REST API.** Analyse emails interactively, browse scan
  history, and triage alerts; or integrate via JSON.
- **Zero heavyweight dependencies.** The data layer is built on Python's
  standard-library `sqlite3` (no ORM), and models persist as plain joblib files.

---

## 📊 Model performance

Numbers below are from the bundled run (`--rows 8000`, 80/20 split, 1,600 held-out test emails):

| Metric (held-out synthetic test set) | Score |
| ------------------------------------ | ----- |
| Accuracy                             | 97.7% |
| Precision                            | 97.4% |
| Recall                               | 98.0% |
| F1                                   | 97.7% |
| Random-Forest 5-fold CV F1           | 0.981 ± 0.001 |

More importantly, the model is validated against a **separate, hand-written
"real-world" evaluation set** of 16 emails (`tests/eval_set.py`) that never
appears in training — calm and loud phishing, look-alike domains, IP links,
shorteners, plus genuinely tricky legitimate mail (receipts, SaaS sign-in
alerts, person-to-person work email):

> **Real-world eval: 16 / 16 correct.** Legitimate mail scores ≤ 0.27,
> phishing scores ≥ 0.85 — wide, confident margins rather than borderline calls.

This holdout is the reason the feature generator deliberately *decorrelates*
structural tells (bad TLDs, IPs, shorteners) from loud text, so the model
learns the structural signals instead of over-fitting to "URGENT!!!".

---

## 🏗️ Architecture

```
phishguard-ai/
├── app.py                     # Flask app: dashboard + REST API
├── phishguard/
│   ├── __init__.py            # package exports (PhishingDetector)
│   ├── features.py            # 27-feature pipeline — SINGLE SOURCE OF TRUTH
│   ├── detector.py            # ensemble inference + heuristic fallback + reasons
│   └── database.py            # sqlite3 layer: reports / alerts / blacklist / stats
├── data/
│   ├── generate_dataset.py    # offline synthetic-email generator
│   └── phishing_emails.csv    # generated corpus (committed for convenience)
├── models/
│   ├── rf_model.pkl           # Random Forest
│   ├── gb_model.pkl           # Gradient Boosting
│   ├── scaler.pkl             # StandardScaler fitted on training features
│   └── metadata.json          # metrics, feature names, importances, timestamp
├── templates/
│   └── dashboard.html         # security-console UI (threat-meter gauge)
├── tests/
│   ├── test_system.py         # unit + integration tests
│   └── eval_set.py            # hand-written real-world holdout
├── train_models.py            # training pipeline (synthetic / CSV / Kaggle)
├── requirements.txt
├── runtime.txt
├── LICENSE                    # MIT
└── README.md
```

**Data flow:** `email → features.py (27 features) → scaler → [RandomForest,
GradientBoosting] → mean probability → verdict + reasons → sqlite3 (report,
and an alert if phishing)`.

---

## 🚀 Setup & run

### 1. Prerequisites
- Python **3.11+** (tested on 3.12)

### 2. Install

```bash
# clone, then from the project root:
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. (Optional) Rebuild the dataset and model

The repo already ships a trained model, so you can skip straight to step 4.
To rebuild everything from scratch:

```bash
python data/generate_dataset.py --rows 8000     # writes data/phishing_emails.csv
python train_models.py --rows 8000              # writes models/*.pkl + metadata.json
```

### 4. Run the web app

```bash
python app.py
# then open http://localhost:5000
```

Set a custom port with `PORT=8080 python app.py`. For production you can use a
WSGI server: uncomment `gunicorn` in `requirements.txt`, then
`gunicorn app:app`.

---

## 🧠 Training options

`train_models.py` can train from three sources (it always falls back to
synthetic data if a source is missing or malformed):

```bash
# 1) Offline synthetic data (default, no internet needed)
python train_models.py --rows 8000

# 2) Your own labelled CSV — columns: subject, sender, content, urls, label
python train_models.py --csv data/my_emails.csv

# 3) A Kaggle dataset (requires `pip install kagglehub` + Kaggle auth);
#    columns are auto-detected, with a synthetic fallback.
python train_models.py --kaggle ethancratchley/email-phishing-dataset
```

After training, `models/metadata.json` records the metrics, the exact feature
order, and per-feature importances.

---

## 🔌 REST API

All endpoints return JSON. Base URL `http://localhost:5000`.

| Method | Endpoint             | Purpose                                   |
| ------ | -------------------- | ----------------------------------------- |
| `GET`  | `/`                  | Web dashboard                             |
| `GET`  | `/api/health`        | Liveness + whether the ML model is active |
| `POST` | `/api/analyze`       | Analyse one email                         |
| `POST` | `/api/feedback`      | Mark a verdict correct / incorrect        |
| `GET`  | `/api/reports`       | Recent scan history                       |
| `GET`  | `/api/alerts`        | Phishing alerts                           |
| `POST` | `/api/alerts/read`   | Mark alerts as read                       |
| `GET`  | `/api/statistics`    | Aggregate stats for the dashboard         |

**Analyse an email:**

```bash
curl -s http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
        "sender": "micros0ft.account@secure-verify.xyz",
        "subject": "Unusual sign-in detected",
        "content": "Someone signed in. Verify your password right away or your account will be locked.",
        "urls": ["http://secure-verify.xyz/login?u=ms"]
      }'
```

**Response (abridged):**

```json
{
  "report_id": 12,
  "classification": "phishing",
  "confidence": 0.888,
  "engine": "ml-ensemble",
  "reasons": [
    "Link uses a top-level domain often abused by attackers",
    "Domain looks random/auto-generated",
    "Message pressures you to act urgently",
    "Multiple known phishing keywords present"
  ],
  "sender_issues": ["Sender uses a top-level domain often abused by attackers"],
  "detected_features": { "has_suspicious_tld": 1, "has_urgent_language": 1, "...": "..." }
}
```

`confidence` is the phishing probability in `[0, 1]`; `classification` is
`"phishing"` when it crosses 0.5. `engine` is `"ml-ensemble"` when the trained
model is loaded, or `"heuristic"` when running on the fallback.

---

## 🧪 Tests

```bash
python tests/test_system.py          # or: python -m pytest -q
```

Covers the feature pipeline, the detector (ML + reasons), the sqlite3 database
round-trip, and end-to-end API behaviour. To run the real-world holdout:

```bash
python -c "import sys; sys.path.insert(0,'tests'); \
from eval_set import EVAL; from phishguard.detector import PhishingDetector; \
d=PhishingDetector(); ok=sum((d.predict(e)['classification']=='phishing')==bool(y) for e,y in EVAL); \
print(f'{ok}/{len(EVAL)} correct')"
```

---

## 🔍 What was fixed from the original hackathon build

The recovered project had the right ideas but did not actually work. Key fixes:

- **Models were never trained** — `rf_model.pkl` was a 0-byte file and the SVM
  and scaler were missing, so the app silently ran on a weak heuristic. Models
  are now properly trained and persisted.
- **Train/inference feature mismatch** — the app scaled some features
  (dividing lengths) before prediction while the scaler had been fit on raw
  values, so predictions were meaningless. Both paths now share one feature
  module.
- **Training read columns that did not exist** in the referenced dataset, so it
  would have trained on all-zero features. Training now validates/auto-detects
  columns and falls back to synthetic data.
- **An alert was built from a report id before it was assigned**, so alerts
  referenced `None`. The database layer now returns ids correctly.
- **Dependency cleanup** — removed unused `opencv-python` / `Pillow`, and
  replaced the Flask-SQLAlchemy/ORM stack with the standard-library `sqlite3`
  for a lighter, fully offline-testable footprint.

---

## 👥 Credits

Originally created as a hackathon project led by **me, leading a team of 4
members**, during my first year. Rebuilt and modernised into the system in
this repository.

## 📄 License

Released under the [MIT License](LICENSE).
