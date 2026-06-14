"""
PhishGuard AI - Detector
========================

Loads the trained ensemble (Random Forest + Gradient Boosting + scaler) and
classifies emails. If no trained models are present it falls back to a
transparent heuristic so the app still runs out of the box - but the whole
point of `train_models.py` is to replace that heuristic with real ML.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np

from .features import (
    FEATURE_NAMES,
    email_to_vector,
    extract_features,
    features_to_vector,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Human-readable reasons shown in the UI when a feature fires.
FEATURE_EXPLANATIONS = {
    "has_ip_address": "Link points to a raw IP address instead of a domain",
    "has_shortener": "Link uses a URL shortener that hides the destination",
    "has_at_symbol": "URL contains an '@' that can redirect to another host",
    "has_suspicious_tld": "Link uses a top-level domain often abused by attackers",
    "has_punycode": "Domain uses punycode, a common look-alike trick",
    "max_subdomains": "Unusually deep subdomain nesting",
    "domain_entropy": "Domain looks random/auto-generated",
    "domain_digit_ratio": "Domain is heavy with digits",
    "has_urgent_language": "Message pressures you to act urgently",
    "num_suspicious_keywords": "Multiple known phishing keywords present",
    "money_term_count": "Mentions money, prizes or rewards",
    "credential_term_count": "Asks for passwords or sensitive credentials",
    "uppercase_ratio": "Heavy use of ALL-CAPS shouting",
    "exclamation_count": "Excessive exclamation marks",
    "sender_has_digits": "Sender address contains digits",
    "suspicious_sender_format": "Sender domain looks irregular",
}


class PhishingDetector:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.rf_model = None
        self.gb_model = None
        self.scaler = None
        self.metadata = {}
        self.using_ml = False
        self._load_models()

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def _load_models(self) -> None:
        try:
            rf_p = os.path.join(self.models_dir, "rf_model.pkl")
            gb_p = os.path.join(self.models_dir, "gb_model.pkl")
            sc_p = os.path.join(self.models_dir, "scaler.pkl")
            meta_p = os.path.join(self.models_dir, "metadata.json")

            if not all(os.path.exists(p) and os.path.getsize(p) > 0
                       for p in (rf_p, gb_p, sc_p)):
                raise FileNotFoundError("trained model files missing or empty")

            self.rf_model = joblib.load(rf_p)
            self.gb_model = joblib.load(gb_p)
            self.scaler = joblib.load(sc_p)
            if os.path.exists(meta_p):
                with open(meta_p) as fh:
                    self.metadata = json.load(fh)
            self.using_ml = True
            print("[PhishGuard] Trained ensemble loaded - ML detection active.")
        except Exception as exc:  # noqa: BLE001
            self.using_ml = False
            print(f"[PhishGuard] No trained models ({exc}).")
            print("[PhishGuard] Falling back to heuristic. Run "
                  "`python train_models.py` to enable ML detection.")

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(self, email: dict) -> dict:
        vector, feats = email_to_vector(email)

        if self.using_ml:
            confidence = self._predict_ml(vector)
            engine = "ml-ensemble"
        else:
            confidence = self._predict_heuristic(feats)
            engine = "heuristic"

        confidence = float(min(max(confidence, 0.0), 1.0))
        classification = "phishing" if confidence >= 0.5 else "legitimate"

        return {
            "classification": classification,
            "confidence_score": confidence,
            "engine": engine,
            "features": feats,
            "reasons": self._reasons(feats),
            "sender_issues": self.analyze_sender(email.get("sender", "")),
        }

    def _predict_ml(self, vector) -> float:
        X = self.scaler.transform(np.array(vector, dtype=float).reshape(1, -1))
        rf = self.rf_model.predict_proba(X)[0][1]
        gb = self.gb_model.predict_proba(X)[0][1]
        return (rf + gb) / 2.0

    @staticmethod
    def _predict_heuristic(feats: dict) -> float:
        """Transparent weighted score used only when no ML model is present."""
        score = 0.0
        score += 0.20 if feats.get("has_ip_address") else 0
        score += 0.15 if feats.get("has_shortener") else 0
        score += 0.15 if feats.get("has_suspicious_tld") else 0
        score += 0.10 if feats.get("has_at_symbol") else 0
        score += 0.10 if feats.get("has_punycode") else 0
        score += 0.10 if feats.get("has_urgent_language") else 0
        score += 0.05 * min(feats.get("num_suspicious_keywords", 0), 4)
        score += 0.05 * min(feats.get("credential_term_count", 0), 3)
        score += 0.05 * min(feats.get("money_term_count", 0), 3)
        score += 0.10 if feats.get("suspicious_sender_format") else 0
        score += 0.05 if feats.get("domain_entropy", 0) > 3.5 else 0
        if feats.get("has_https") and feats.get("num_urls"):
            score -= 0.05
        return score

    # ------------------------------------------------------------------ #
    # Explainability
    # ------------------------------------------------------------------ #
    @staticmethod
    def _reasons(feats: dict) -> list[str]:
        reasons = []
        for key, text in FEATURE_EXPLANATIONS.items():
            val = feats.get(key, 0)
            if key == "domain_entropy" and val > 3.5:
                reasons.append(text)
            elif key == "domain_digit_ratio" and val > 0.3:
                reasons.append(text)
            elif key == "uppercase_ratio" and val > 0.3:
                reasons.append(text)
            elif key == "exclamation_count" and val >= 3:
                reasons.append(text)
            elif key == "max_subdomains" and val >= 3:
                reasons.append(text)
            elif key in ("num_suspicious_keywords", "money_term_count",
                         "credential_term_count") and val >= 2:
                reasons.append(text)
            elif key in ("has_ip_address", "has_shortener", "has_at_symbol",
                         "has_suspicious_tld", "has_punycode",
                         "has_urgent_language", "sender_has_digits",
                         "suspicious_sender_format") and val:
                reasons.append(text)
        return reasons

    @staticmethod
    def analyze_sender(sender: str) -> list[str]:
        issues = []
        sender = (sender or "").strip()
        if "@" not in sender:
            issues.append("Sender is not a valid email address")
            return issues
        domain = sender.split("@", 1)[1].lower()
        if len(domain) > 40:
            issues.append("Unusually long sender domain")
        if domain.count("-") > 2 or "--" in domain:
            issues.append("Sender domain has a suspicious hyphen pattern")
        if any(domain.endswith(t) for t in
               (".xyz", ".top", ".tk", ".zip", ".click", ".work")):
            issues.append("Sender uses a top-level domain often abused by attackers")
        if any(ch.isdigit() for ch in domain.split(".")[0]):
            issues.append("Sender domain name contains digits")
        return issues

    # convenience for callers / tests
    def explain_vector(self, email: dict) -> dict:
        return dict(zip(FEATURE_NAMES, features_to_vector(extract_features(email))))
