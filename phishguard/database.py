"""
PhishGuard AI - Database layer
==============================

A thin wrapper over Python's built-in sqlite3. It keeps the same logical
tables the original project defined (reports, alerts, blacklist) but removes
the Flask-SQLAlchemy/SQLAlchemy dependency, so the app installs and runs with
far fewer moving parts.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "instance")
# Allow the storage location to be overridden (deployment / isolated tests).
DB_PATH = os.environ.get("PHISHGUARD_DB") or os.path.join(DB_DIR, "phishguard.db")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class Database:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS phishing_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_subject TEXT,
                    sender_email TEXT,
                    message_content TEXT,
                    urls TEXT,
                    classification TEXT,
                    confidence_score REAL,
                    engine TEXT,
                    detected_features TEXT,
                    reasons TEXT,
                    timestamp TEXT,
                    user_feedback TEXT
                );

                CREATE TABLE IF NOT EXISTS user_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT,
                    message TEXT,
                    severity TEXT,
                    report_id INTEGER,
                    is_read INTEGER DEFAULT 0,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS url_blacklist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    threat_level TEXT,
                    added_date TEXT
                );
                """
            )

    # ------------------------------------------------------------------ #
    # Reports
    # ------------------------------------------------------------------ #
    def add_report(self, email: dict, result: dict) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO phishing_reports
                   (email_subject, sender_email, message_content, urls,
                    classification, confidence_score, engine,
                    detected_features, reasons, timestamp)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    email.get("subject", ""),
                    email.get("sender", ""),
                    email.get("content", ""),
                    json.dumps(email.get("urls", [])),
                    result["classification"],
                    round(float(result["confidence_score"]), 4),
                    result.get("engine", "unknown"),
                    json.dumps(result.get("features", {})),
                    json.dumps(result.get("reasons", [])),
                    _now(),
                ),
            )
            report_id = cur.lastrowid

            if result["classification"] == "phishing":
                c.execute(
                    """INSERT INTO user_alerts
                       (alert_type, message, severity, report_id, created_at)
                       VALUES (?,?,?,?,?)""",
                    (
                        "Phishing Detected",
                        f"Potential phishing email from "
                        f"{email.get('sender', 'unknown sender')}",
                        "high" if result["confidence_score"] >= 0.8 else "medium",
                        report_id,
                        _now(),
                    ),
                )
            return report_id

    def get_reports(self, page: int = 1, per_page: int = 20) -> dict:
        offset = (page - 1) * per_page
        with self._conn() as c:
            total = c.execute(
                "SELECT COUNT(*) AS n FROM phishing_reports").fetchone()["n"]
            rows = c.execute(
                """SELECT * FROM phishing_reports
                   ORDER BY id DESC LIMIT ? OFFSET ?""",
                (per_page, offset),
            ).fetchall()
        reports = [
            {
                "id": r["id"],
                "sender_email": r["sender_email"],
                "subject": r["email_subject"],
                "classification": r["classification"],
                "confidence_score": round(r["confidence_score"], 4),
                "engine": r["engine"],
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]
        pages = max(1, (total + per_page - 1) // per_page)
        return {"total": total, "pages": pages,
                "current_page": page, "reports": reports}

    def set_feedback(self, report_id: int, feedback: str) -> bool:
        with self._conn() as c:
            cur = c.execute(
                "UPDATE phishing_reports SET user_feedback=? WHERE id=?",
                (feedback, report_id),
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------------ #
    # Alerts
    # ------------------------------------------------------------------ #
    def get_alerts(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT * FROM user_alerts WHERE is_read=0
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "type": r["alert_type"],
                "message": r["message"],
                "severity": r["severity"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def mark_alerts_read(self) -> int:
        with self._conn() as c:
            cur = c.execute("UPDATE user_alerts SET is_read=1 WHERE is_read=0")
            return cur.rowcount

    # ------------------------------------------------------------------ #
    # Blacklist
    # ------------------------------------------------------------------ #
    def add_blacklist(self, url: str, threat_level: str = "high") -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR IGNORE INTO url_blacklist
                   (url, threat_level, added_date) VALUES (?,?,?)""",
                (url, threat_level, _now()),
            )

    def is_blacklisted(self, url: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM url_blacklist WHERE url=?", (url,)).fetchone()
        return row is not None

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #
    def statistics(self) -> dict:
        with self._conn() as c:
            total = c.execute(
                "SELECT COUNT(*) AS n FROM phishing_reports").fetchone()["n"]
            phishing = c.execute(
                "SELECT COUNT(*) AS n FROM phishing_reports "
                "WHERE classification='phishing'").fetchone()["n"]
            correct = c.execute(
                "SELECT COUNT(*) AS n FROM phishing_reports "
                "WHERE user_feedback='correct'").fetchone()["n"]
            graded = c.execute(
                "SELECT COUNT(*) AS n FROM phishing_reports "
                "WHERE user_feedback IS NOT NULL").fetchone()["n"]
        legitimate = total - phishing
        accuracy = (correct / graded * 100) if graded else 0.0
        return {
            "total_analyzed": total,
            "phishing_detected": phishing,
            "legitimate": legitimate,
            "feedback_accuracy": round(accuracy, 2),
            "phishing_percentage": round((phishing / total * 100) if total else 0, 2),
        }
