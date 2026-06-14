"""
PhishGuard AI - Feature Extraction Pipeline
===========================================

This module is the SINGLE SOURCE OF TRUTH for turning a raw email
(subject + sender + body + URLs) into the numeric feature vector that the
machine-learning models consume.

Both `train_models.py` (training time) and `phishguard/detector.py`
(inference time) import `extract_features` from here, so the two can never
drift apart. The original hackathon code had a subtle bug where training and
inference scaled features differently; centralising the logic removes that
whole class of error.

Every feature is a plain number, and `FEATURE_NAMES` defines the exact order
of the output vector.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Constants / lexicons
# ---------------------------------------------------------------------------

URL_SHORTENERS = (
    "bit.ly", "tinyurl", "ow.ly", "t.co", "goo.gl", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "short.link", "tiny.cc", "rb.gy",
)

# TLDs disproportionately abused by phishing / malware campaigns.
SUSPICIOUS_TLDS = (
    ".zip", ".xyz", ".top", ".tk", ".ml", ".ga", ".cf", ".gq", ".work",
    ".click", ".country", ".kim", ".loan", ".men", ".mom", ".party",
    ".review", ".stream", ".trade", ".date", ".racing", ".support",
)

FREE_EMAIL_DOMAINS = (
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com",
    "protonmail.com", "icloud.com", "mail.com", "gmx.com", "yandex.com",
)

# Words/phrases that pressure a reader to act quickly.
URGENT_TERMS = (
    "urgent", "immediately", "immediate", "action required", "verify now",
    "as soon as possible", "right away", "expire", "expires", "expiring",
    "suspended", "suspension", "deactivat", "within 24 hours", "final notice",
    "last warning", "act now", "limited time",
)

# Generic phishing keyword lexicon.
SUSPICIOUS_KEYWORDS = (
    "verify", "confirm", "urgent", "action required", "click here",
    "update account", "validate", "suspended", "limited time", "claim",
    "reward", "winner", "congratulations", "unusual activity", "security alert",
    "log in", "login", "sign in", "password", "credential", "billing",
    "invoice", "payment failed", "refund", "gift card",
)

# Words that ask for money or financial value.
MONEY_TERMS = (
    "$", "usd", "bitcoin", "btc", "wire transfer", "gift card", "prize",
    "lottery", "inheritance", "deposit", "refund", "bonus", "cash", "reward",
)

# Words that ask for secrets / credentials.
CREDENTIAL_TERMS = (
    "password", "ssn", "social security", "account number", "pin",
    "credit card", "cvv", "login", "username", "one-time code", "otp",
    "security code", "banking", "credentials",
)

# Order MUST stay stable: the trained model depends on it.
FEATURE_NAMES = [
    # --- URL features ---
    "num_urls",
    "max_url_length",
    "has_ip_address",
    "has_shortener",
    "has_at_symbol",
    "max_subdomains",
    "max_domain_dots",
    "has_https",
    "has_suspicious_tld",
    "has_punycode",
    "domain_digit_ratio",
    "domain_entropy",
    # --- Text / body features ---
    "text_length",
    "num_suspicious_keywords",
    "has_urgent_language",
    "exclamation_count",
    "uppercase_ratio",
    "money_term_count",
    "credential_term_count",
    "links_mentioned",
    # --- Sender features ---
    "sender_has_digits",
    "sender_domain_length",
    "sender_is_freemail",
    "suspicious_sender_format",
    # --- Subject features ---
    "subject_length",
    "subject_is_urgent",
    "subject_has_reply_prefix",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(text: str) -> float:
    """Shannon entropy of a string. Random-looking phishing domains score high."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _safe_domain(url: str) -> str:
    """Best-effort netloc extraction that tolerates missing schemes."""
    if "://" not in url:
        url = "http://" + url
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


_IP_RE = re.compile(r"https?://(?:[^/@]+@)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
_URL_IN_TEXT_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)


def _normalise_urls(urls) -> list[str]:
    """Accept a list, a newline string, or None and return a clean URL list."""
    if not urls:
        return []
    if isinstance(urls, str):
        urls = re.split(r"[\n,]+", urls)
    return [u.strip() for u in urls if u and str(u).strip()]


# ---------------------------------------------------------------------------
# Per-group extractors (also returned for explainability in the UI)
# ---------------------------------------------------------------------------

def url_features(urls) -> dict:
    urls = _normalise_urls(urls)
    f = {
        "num_urls": len(urls),
        "max_url_length": 0,
        "has_ip_address": 0,
        "has_shortener": 0,
        "has_at_symbol": 0,
        "max_subdomains": 0,
        "max_domain_dots": 0,
        "has_https": 0,
        "has_suspicious_tld": 0,
        "has_punycode": 0,
        "domain_digit_ratio": 0.0,
        "domain_entropy": 0.0,
    }
    if not urls:
        return f

    worst_digit_ratio = 0.0
    worst_entropy = 0.0
    for url in urls:
        low = url.lower()
        f["max_url_length"] = max(f["max_url_length"], len(url))
        if _IP_RE.match(low):
            f["has_ip_address"] = 1
        if any(s in low for s in URL_SHORTENERS):
            f["has_shortener"] = 1
        if "@" in url.split("//")[-1]:
            f["has_at_symbol"] = 1
        if low.startswith("https://"):
            f["has_https"] = 1
        if any(low.split("?")[0].rstrip("/").endswith(t) or t + "/" in low
               for t in SUSPICIOUS_TLDS):
            f["has_suspicious_tld"] = 1

        domain = _safe_domain(url)
        if domain:
            if "xn--" in domain:
                f["has_punycode"] = 1
            dots = domain.count(".")
            f["max_domain_dots"] = max(f["max_domain_dots"], dots)
            f["max_subdomains"] = max(f["max_subdomains"], max(0, dots - 1))
            digits = sum(ch.isdigit() for ch in domain)
            worst_digit_ratio = max(worst_digit_ratio, digits / len(domain))
            worst_entropy = max(worst_entropy, _shannon_entropy(domain))

    f["domain_digit_ratio"] = round(worst_digit_ratio, 4)
    f["domain_entropy"] = round(worst_entropy, 4)
    return f


def text_features(content: str, urls) -> dict:
    text = content or ""
    low = text.lower()
    letters = [c for c in text if c.isalpha()]
    uppercase_ratio = (
        sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
    )
    links_in_body = len(_URL_IN_TEXT_RE.findall(text)) + len(_normalise_urls(urls))
    return {
        "text_length": len(text),
        "num_suspicious_keywords": sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in low),
        "has_urgent_language": int(any(w in low for w in URGENT_TERMS)),
        "exclamation_count": text.count("!"),
        "uppercase_ratio": round(uppercase_ratio, 4),
        "money_term_count": sum(1 for w in MONEY_TERMS if w in low),
        "credential_term_count": sum(1 for w in CREDENTIAL_TERMS if w in low),
        "links_mentioned": links_in_body,
    }


def sender_features(sender: str) -> dict:
    sender = (sender or "").strip().lower()
    domain = sender.split("@", 1)[1] if "@" in sender else ""
    suspicious_format = 0
    if domain:
        if len(domain) > 30 or domain.count(".") > 2 or domain.count("-") > 2:
            suspicious_format = 1
    return {
        "sender_has_digits": int(bool(re.search(r"\d", sender))),
        "sender_domain_length": len(domain),
        "sender_is_freemail": int(domain in FREE_EMAIL_DOMAINS),
        "suspicious_sender_format": suspicious_format,
    }


def subject_features(subject: str) -> dict:
    subject = subject or ""
    low = subject.lower()
    return {
        "subject_length": len(subject),
        "subject_is_urgent": int(any(w in low for w in URGENT_TERMS)),
        "subject_has_reply_prefix": int(bool(re.match(r"\s*(re|fwd)\s*:", low))),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(email: dict) -> dict:
    """
    Turn one email dict into a flat {feature_name: value} dictionary.

    `email` keys (all optional): subject, sender, content, urls.
    `urls` may be a list or a newline/comma separated string.
    """
    feats = {}
    feats.update(url_features(email.get("urls")))
    feats.update(text_features(email.get("content", ""), email.get("urls")))
    feats.update(sender_features(email.get("sender", "")))
    feats.update(subject_features(email.get("subject", "")))
    return feats


def features_to_vector(feats: dict) -> list[float]:
    """Project a feature dict onto the canonical FEATURE_NAMES order."""
    return [float(feats.get(name, 0)) for name in FEATURE_NAMES]


def email_to_vector(email: dict):
    """Convenience: raw email -> (vector, feature_dict)."""
    feats = extract_features(email)
    return features_to_vector(feats), feats
