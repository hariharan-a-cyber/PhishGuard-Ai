"""
PhishGuard AI - Test suite
==========================
Run with:  python -m pytest -q     (or: python tests/test_system.py)
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phishguard.features import FEATURE_NAMES, extract_features, features_to_vector
from phishguard.detector import PhishingDetector
from phishguard.database import Database

PHISH = {
    "sender": "security-alert42@paypal-secure.xyz",
    "subject": "URGENT: Your account has been suspended",
    "content": "Verify your password and billing details within 24 hours!!! "
               "Unusual activity detected. Confirm now to avoid suspension.",
    "urls": ["http://192.168.4.21/paypal/secure-login.php"],
}
LEGIT = {
    "sender": "no-reply@amazon.com",
    "subject": "Your receipt for order #482910",
    "content": "Hi Alex, thanks for your purchase. Your order has shipped and "
               "is on its way. You can track it from your account.",
    "urls": ["https://www.amazon.com/orders"],
}


def test_feature_vector_shape():
    vec = features_to_vector(extract_features(PHISH))
    assert len(vec) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in vec)


def test_url_signals_detected():
    # The PHISH sample's link is a raw IP, so has_ip_address fires while
    # has_suspicious_tld (a URL-host check) is correctly 0 here.
    f = extract_features(PHISH)
    assert f["has_ip_address"] == 1
    assert f["has_https"] == 0          # IP link over plain http
    assert f["has_urgent_language"] == 1
    assert f["credential_term_count"] >= 1

    # A bad-TLD URL must independently raise has_suspicious_tld.
    g = extract_features({
        "sender": "support@parcel-track.tk",
        "subject": "Your package is on hold",
        "content": "Pay the customs fee here: http://parcel-track.tk/pay",
        "urls": ["http://parcel-track.tk/pay"],
    })
    assert g["has_suspicious_tld"] == 1


def test_detector_classifies():
    det = PhishingDetector()
    p = det.predict(PHISH)
    l = det.predict(LEGIT)
    assert p["classification"] == "phishing"
    assert l["classification"] == "legitimate"
    assert 0.0 <= p["confidence_score"] <= 1.0
    assert p["confidence_score"] > l["confidence_score"]


def test_detector_reasons():
    det = PhishingDetector()
    assert len(det.predict(PHISH)["reasons"]) >= 1


def test_database_roundtrip():
    tmp = tempfile.mkdtemp()
    db = Database(os.path.join(tmp, "t.db"))
    det = PhishingDetector()
    rid = db.add_report(PHISH, det.predict(PHISH))
    assert rid > 0
    reps = db.get_reports()
    assert reps["total"] == 1
    assert reps["reports"][0]["classification"] == "phishing"
    assert db.get_alerts(), "phishing report should raise an alert"
    assert db.set_feedback(rid, "correct")
    assert db.statistics()["total_analyzed"] == 1


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
    print(f"\n{'ALL PASSED' if not failures else str(failures)+' FAILED'}")
    sys.exit(1 if failures else 0)
