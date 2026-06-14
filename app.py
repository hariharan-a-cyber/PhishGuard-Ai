"""
PhishGuard AI - Flask Web Application
=====================================

AI-based phishing detection. Serves the dashboard and a small REST API that
analyses emails with the trained ML ensemble, stores reports, and raises
alerts. Persistence uses the built-in sqlite3 layer in `phishguard/database.py`.
"""

from __future__ import annotations

import os
import traceback

from flask import Flask, jsonify, render_template, request

from phishguard import __version__
from phishguard.database import Database
from phishguard.detector import PhishingDetector

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

db = Database()
detector = PhishingDetector()


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "version": __version__,
        "ml_active": detector.using_ml,
        "model_metrics": detector.metadata.get("metrics", {}),
    })


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
@app.route("/api/analyze", methods=["POST"])
def analyze_email():
    try:
        data = request.get_json(force=True, silent=True) or {}
        urls = data.get("urls", [])
        if isinstance(urls, str):
            urls = [u.strip() for u in urls.splitlines() if u.strip()]

        email = {
            "subject": data.get("subject", ""),
            "sender": data.get("sender", ""),
            "content": data.get("content", ""),
            "urls": urls,
        }

        result = detector.predict(email)
        report_id = db.add_report(email, result)

        return jsonify({
            "report_id": report_id,
            "classification": result["classification"],
            "confidence": result["confidence_score"],
            "engine": result["engine"],
            "detected_features": result["features"],
            "reasons": result["reasons"],
            "sender_issues": result["sender_issues"],
        })
    except Exception as exc:  # noqa: BLE001
        app.logger.error("analyze failed: %s", traceback.format_exc())
        return jsonify({"error": str(exc)}), 400


@app.route("/api/feedback", methods=["POST"])
def feedback():
    """Let users mark a verdict correct/incorrect (used for accuracy stats)."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        ok = db.set_feedback(int(data["report_id"]), str(data["feedback"]))
        return jsonify({"updated": ok})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


# --------------------------------------------------------------------------- #
# Reports / alerts / stats
# --------------------------------------------------------------------------- #
@app.route("/api/reports")
def get_reports():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    return jsonify(db.get_reports(page, per_page))


@app.route("/api/alerts")
def get_alerts():
    return jsonify({"alerts": db.get_alerts()})


@app.route("/api/alerts/read", methods=["POST"])
def read_alerts():
    return jsonify({"marked_read": db.mark_alerts_read()})


@app.route("/api/statistics")
def get_statistics():
    stats = db.statistics()
    stats["ml_active"] = detector.using_ml
    stats["model_accuracy"] = (
        detector.metadata.get("metrics", {})
        .get("ensemble", {}).get("accuracy")
    )
    return jsonify(stats)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(_):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print(f"  PhishGuard AI v{__version__}")
    print(f"  ML detection: {'ACTIVE' if detector.using_ml else 'heuristic fallback'}")
    print(f"  Open: http://localhost:{port}")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=port)
