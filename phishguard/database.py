"""
PhishGuard AI - Database layer
==============================

Auto-selects backend:
  PostgreSQL  when POSTGRES_URL or DATABASE_URL env var is set  (Vercel / production)
  SQLite      otherwise                                          (local development)
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

_PG_DSN = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "instance")
_SQLITE_PATH = (
    os.environ.get("PHISHGUARD_DB")
    or ("/tmp/phishguard.db" if os.environ.get("VERCEL") else "")
    or os.path.join(_DB_DIR, "phishguard.db")
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class Database:
    def __init__(self, path: str = _SQLITE_PATH):
        self._pg = bool(_PG_DSN)
        if not self._pg:
            self._path = path
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.init_db()

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #
    @contextmanager
    def _conn(self):
        if self._pg:
            import psycopg2
            conn = psycopg2.connect(_PG_DSN)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            conn = sqlite3.connect(self._path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _cur(self, conn):
        """Cursor with dict-like row access for either backend."""
        if self._pg:
            import psycopg2.extras
            return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        return conn.cursor()

    def _q(self, sql: str) -> str:
        """Convert SQLite ? placeholders to %s for PostgreSQL."""
        return sql.replace("?", "%s") if self._pg else sql

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def init_db(self) -> None:
        if self._pg:
            pk = "SERIAL PRIMARY KEY"
            stmts = [
                f"""CREATE TABLE IF NOT EXISTS phishing_reports (
                    id {pk}, email_subject TEXT, sender_email TEXT,
                    message_content TEXT, urls TEXT, classification TEXT,
                    confidence_score REAL, engine TEXT, detected_features TEXT,
                    reasons TEXT, timestamp TEXT, user_feedback TEXT,
                    device_id TEXT)""",
                f"""CREATE TABLE IF NOT EXISTS user_alerts (
                    id {pk}, alert_type TEXT, message TEXT, severity TEXT,
                    report_id INTEGER, is_read INTEGER DEFAULT 0, created_at TEXT,
                    device_id TEXT)""",
                f"""CREATE TABLE IF NOT EXISTS url_blacklist (
                    id {pk}, url TEXT UNIQUE, threat_level TEXT, added_date TEXT)""",
            ]
            with self._conn() as conn:
                cur = conn.cursor()
                for s in stmts:
                    cur.execute(s)
                # migrate existing tables
                for tbl in ("phishing_reports", "user_alerts"):
                    try:
                        cur.execute(f"ALTER TABLE {tbl} ADD COLUMN device_id TEXT")
                    except Exception:
                        pass
        else:
            with self._conn() as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS phishing_reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email_subject TEXT, sender_email TEXT,
                        message_content TEXT, urls TEXT, classification TEXT,
                        confidence_score REAL, engine TEXT, detected_features TEXT,
                        reasons TEXT, timestamp TEXT, user_feedback TEXT,
                        device_id TEXT
                    );
                    CREATE TABLE IF NOT EXISTS user_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_type TEXT, message TEXT, severity TEXT,
                        report_id INTEGER, is_read INTEGER DEFAULT 0, created_at TEXT,
                        device_id TEXT
                    );
                    CREATE TABLE IF NOT EXISTS url_blacklist (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT UNIQUE, threat_level TEXT, added_date TEXT
                    );
                """)
            # migrate existing SQLite tables
            with self._conn() as conn:
                for tbl in ("phishing_reports", "user_alerts"):
                    try:
                        conn.execute(f"ALTER TABLE {tbl} ADD COLUMN device_id TEXT")
                    except Exception:
                        pass

    # ------------------------------------------------------------------ #
    # Reports
    # ------------------------------------------------------------------ #
    def add_report(self, email: dict, result: dict, device_id: str = "") -> int:
        params = (
            email.get("subject", ""), email.get("sender", ""),
            email.get("content", ""), json.dumps(email.get("urls", [])),
            result["classification"], round(float(result["confidence_score"]), 4),
            result.get("engine", "unknown"),
            json.dumps(result.get("features", {})),
            json.dumps(result.get("reasons", [])), _now(), device_id or None,
        )
        insert_report = """INSERT INTO phishing_reports
            (email_subject, sender_email, message_content, urls,
             classification, confidence_score, engine,
             detected_features, reasons, timestamp, device_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)"""

        with self._conn() as conn:
            cur = self._cur(conn)
            if self._pg:
                cur.execute(self._q(insert_report) + " RETURNING id", params)
                report_id = cur.fetchone()["id"]
            else:
                cur.execute(insert_report, params)
                report_id = cur.lastrowid

            if result["classification"] == "phishing":
                alert_params = (
                    "Phishing Detected",
                    f"Potential phishing email from {email.get('sender', 'unknown sender')}",
                    "high" if result["confidence_score"] >= 0.8 else "medium",
                    report_id, _now(), device_id or None,
                )
                insert_alert = """INSERT INTO user_alerts
                    (alert_type, message, severity, report_id, created_at, device_id)
                    VALUES (?,?,?,?,?,?)"""
                self._cur(conn).execute(self._q(insert_alert), alert_params)
        return report_id

    def get_reports(self, page: int = 1, per_page: int = 20, device_id: str = "") -> dict:
        offset = (page - 1) * per_page
        where = self._q("WHERE device_id=?") if device_id else ""
        with self._conn() as conn:
            cur = self._cur(conn)
            cur.execute(f"SELECT COUNT(*) AS n FROM phishing_reports {where}",
                        (device_id,) if device_id else ())
            total = cur.fetchone()["n"]
            cur.execute(
                self._q(f"SELECT * FROM phishing_reports {where} ORDER BY id DESC LIMIT ? OFFSET ?"),
                ((device_id, per_page, offset) if device_id else (per_page, offset)))
            rows = [dict(r) for r in cur.fetchall()]
        reports = [
            {"id": r["id"], "sender_email": r["sender_email"],
             "subject": r["email_subject"], "classification": r["classification"],
             "confidence_score": round(r["confidence_score"], 4),
             "engine": r["engine"], "timestamp": r["timestamp"]}
            for r in rows
        ]
        pages = max(1, (total + per_page - 1) // per_page)
        return {"total": total, "pages": pages, "current_page": page, "reports": reports}

    def set_feedback(self, report_id: int, feedback: str) -> bool:
        with self._conn() as conn:
            cur = self._cur(conn)
            cur.execute(
                self._q("UPDATE phishing_reports SET user_feedback=? WHERE id=?"),
                (feedback, report_id))
            return cur.rowcount > 0

    def search_reports(self, keyword: str, device_id: str = "") -> list[dict]:
        clause = f"WHERE email_subject LIKE '%{keyword}%'"
        if device_id:
            clause += f" AND device_id='{device_id}'"
        sql = (
            f"SELECT id, email_subject, sender_email, classification, confidence_score, timestamp "
            f"FROM phishing_reports {clause} ORDER BY id DESC LIMIT 200"
        )
        with self._conn() as conn:
            cur = self._cur(conn)
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------ #
    # Alerts
    # ------------------------------------------------------------------ #
    def get_alerts(self, limit: int = 50, device_id: str = "") -> list[dict]:
        if device_id:
            sql = self._q("SELECT * FROM user_alerts WHERE is_read=0 AND device_id=? ORDER BY id DESC LIMIT ?")
            args = (device_id, limit)
        else:
            sql = self._q("SELECT * FROM user_alerts WHERE is_read=0 ORDER BY id DESC LIMIT ?")
            args = (limit,)
        with self._conn() as conn:
            cur = self._cur(conn)
            cur.execute(sql, args)
            rows = [dict(r) for r in cur.fetchall()]
        return [{"id": r["id"], "type": r["alert_type"], "message": r["message"],
                 "severity": r["severity"], "created_at": r["created_at"]}
                for r in rows]

    def mark_alerts_read(self) -> int:
        with self._conn() as conn:
            cur = self._cur(conn)
            cur.execute("UPDATE user_alerts SET is_read=1 WHERE is_read=0")
            return cur.rowcount

    # ------------------------------------------------------------------ #
    # Blacklist
    # ------------------------------------------------------------------ #
    def add_blacklist(self, url: str, threat_level: str = "high") -> None:
        with self._conn() as conn:
            if self._pg:
                conn.cursor().execute(
                    "INSERT INTO url_blacklist (url, threat_level, added_date) "
                    "VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                    (url, threat_level, _now()))
            else:
                conn.execute(
                    "INSERT OR IGNORE INTO url_blacklist "
                    "(url, threat_level, added_date) VALUES (?,?,?)",
                    (url, threat_level, _now()))

    def is_blacklisted(self, url: str) -> bool:
        with self._conn() as conn:
            cur = self._cur(conn)
            cur.execute(self._q("SELECT 1 FROM url_blacklist WHERE url=?"), (url,))
            return cur.fetchone() is not None

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #
    def statistics(self) -> dict:
        with self._conn() as conn:
            cur = self._cur(conn)

            def n(sql: str) -> int:
                cur.execute(sql)
                return cur.fetchone()["n"]

            total      = n("SELECT COUNT(*) AS n FROM phishing_reports")
            phishing   = n("SELECT COUNT(*) AS n FROM phishing_reports WHERE confidence_score >= 0.65")
            suspicious = n("SELECT COUNT(*) AS n FROM phishing_reports WHERE confidence_score >= 0.45 AND confidence_score < 0.65")
            correct    = n("SELECT COUNT(*) AS n FROM phishing_reports WHERE user_feedback='correct'")
            graded     = n("SELECT COUNT(*) AS n FROM phishing_reports WHERE user_feedback IS NOT NULL")

        legitimate = total - phishing - suspicious
        accuracy = (correct / graded * 100) if graded else 0.0
        return {
            "total_analyzed": total,
            "phishing_detected": phishing,
            "suspicious_detected": suspicious,
            "legitimate": legitimate,
            "feedback_accuracy": round(accuracy, 2),
            "phishing_percentage": round((phishing / total * 100) if total else 0, 2),
        }
