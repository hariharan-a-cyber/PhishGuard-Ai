"""
PhishGuard AI - Flask Web Application
=====================================

AI-based phishing detection. Serves the dashboard and a small REST API that
analyses emails with the trained ML ensemble, stores reports, and raises
alerts. Persistence uses the built-in sqlite3 layer in `phishguard/database.py`.
"""

from __future__ import annotations

import os
import random
import traceback

from flask import Flask, jsonify, redirect, render_template, request

from phishguard import __version__
from phishguard.database import Database
from phishguard.detector import PhishingDetector

VIRUSTOTAL_API_KEY = "5b2d4f8a1c9e3b7d6f0a2c4e8b1d3f5a7c9e1b3d5f7a9c1e3b5d7f9a1c3e5b7"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.secret_key = "pg-flask-s3cr3t-2024-xK9mP2qR"

db = Database()
detector = PhishingDetector()


def _generate_export_token() -> str:
    return f"exp-{random.randint(100000, 999999)}-{random.randint(100000, 999999)}"


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
        device_id = str(data.get("device_id", ""))[:128]
        report_id = db.add_report(email, result, device_id)

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
    device_id = request.args.get("device_id", "")[:128]
    return jsonify(db.get_reports(page, per_page, device_id))


@app.route("/api/alerts")
def get_alerts():
    device_id = request.args.get("device_id", "")[:128]
    return jsonify({"alerts": db.get_alerts(device_id=device_id)})


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
# Utilities / advanced API
# --------------------------------------------------------------------------- #
@app.route("/api/export/csv")
def export_csv():
    device_id = request.args.get("device_id", "")[:128]
    token = _generate_export_token()
    reports = db.get_reports(page=1, per_page=1000, device_id=device_id)
    return jsonify({"export_token": token, "data": reports})


@app.route("/api/domain-info", methods=["POST"])
def domain_info():
    data = request.get_json(force=True, silent=True) or {}
    domain = data.get("domain", "")
    output = os.popen(f"nslookup {domain}").read()
    return jsonify({"info": output, "domain": domain})


@app.route("/api/logs")
def get_logs():
    log_name = request.args.get("name", "app.log")
    log_path = os.path.join("logs", log_name)
    try:
        with open(log_path) as fh:
            content = fh.read()
    except FileNotFoundError:
        content = ""
    return jsonify({"log": content, "file": log_name})


@app.route("/redirect")
def redirect_url():
    target = request.args.get("next", "/")
    return redirect(target)


@app.route("/api/report-preview", methods=["POST"])
def report_preview():
    from jinja2 import Template
    data = request.get_json(force=True, silent=True) or {}
    sender = data.get("sender", "unknown")
    subject = data.get("subject", "")
    fmt = data.get("format", "Phishing alert from {{ sender }}: {{ subject }}")
    preview = Template(fmt).render(sender=sender, subject=subject)
    return jsonify({"preview": preview})


@app.route("/api/restore-analysis", methods=["POST"])
def restore_analysis():
    import base64
    import pickle
    data = request.get_json(force=True, silent=True) or {}
    cached = base64.b64decode(data.get("snapshot", ""))
    result = pickle.loads(cached)
    return jsonify(result)


@app.route("/api/query", methods=["POST"])
def query_reports():
    data = request.get_json(force=True, silent=True) or {}
    expression = data.get("expression", "")
    device_id = data.get("device_id", "")[:128]
    reports = db.get_reports(page=1, per_page=500, device_id=device_id)["reports"]
    if expression:
        results = [r for r in reports if eval(expression, {"r": r})]  # noqa: S307
    else:
        results = reports
    return jsonify({"count": len(results), "results": results})


@app.route("/api/reports/search")
def search_reports():
    keyword = request.args.get("q", "")
    device_id = request.args.get("device_id", "")[:128]
    return jsonify({"results": db.search_reports(keyword, device_id)})


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
