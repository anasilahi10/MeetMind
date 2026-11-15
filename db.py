import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data/meetmind.db")


@contextmanager
def _db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _meeting(row):
    if row is None:
        return None
    d = dict(row)
    try:
        d["action_items"] = json.loads(d.get("action_items") or "[]")
    except Exception:
        d["action_items"] = []
    return d


def init_db():
    with _db() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                filename TEXT,
                created_at TEXT NOT NULL,
                duration_sec REAL,
                transcript TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT ''
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS chat (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )""")
        cols = [r["name"]
                for r in c.execute("PRAGMA table_info(meetings)").fetchall()]
        if "action_items" not in cols:
            c.execute(
                "ALTER TABLE meetings ADD COLUMN action_items TEXT NOT NULL DEFAULT '[]'")


# ---------- meetings ----------

def save_meeting(title, filename, duration_sec, transcript, summary="", action_items=None):
    with _db() as c:
        cur = c.execute(
            "INSERT INTO meetings (title, filename, created_at, duration_sec, transcript, summary, action_items) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title, filename, datetime.now().strftime("%Y-%m-%d %H:%M"), duration_sec,
             transcript, summary, json.dumps(action_items or [])),
        )
        return cur.lastrowid


def list_meetings():
    with _db() as c:
        return [dict(r) for r in c.execute(
            "SELECT id, title, created_at FROM meetings ORDER BY id DESC").fetchall()]


def all_meetings():
    with _db() as c:
        return [_meeting(r) for r in c.execute(
            "SELECT * FROM meetings ORDER BY id DESC").fetchall()]


def get_meeting(meeting_id):
    with _db() as c:
        return _meeting(c.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone())


def update_summary(meeting_id, summary, action_items=None):
    with _db() as c:
        if action_items is None:
            c.execute("UPDATE meetings SET summary = ? WHERE id = ?",
                      (summary, meeting_id))
        else:
            c.execute("UPDATE meetings SET summary = ?, action_items = ? WHERE id = ?",
                      (summary, json.dumps(action_items), meeting_id))


def update_text(meeting_id, transcript, summary, action_items):
    with _db() as c:
        c.execute("UPDATE meetings SET transcript = ?, summary = ?, action_items = ? WHERE id = ?",
                  (transcript, summary, json.dumps(action_items), meeting_id))


def update_title(meeting_id, title):
    with _db() as c:
        c.execute("UPDATE meetings SET title = ? WHERE id = ?",
                  (title, meeting_id))


def delete_meeting(meeting_id):
    with _db() as c:
        c.execute("DELETE FROM chat WHERE meeting_id = ?", (meeting_id,))
        c.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))


# ---------- chat ----------

def add_chat(meeting_id, role, content):
    with _db() as c:
        c.execute("INSERT INTO chat (meeting_id, role, content) VALUES (?, ?, ?)",
                  (meeting_id, role, content))


def get_chat(meeting_id):
    with _db() as c:
        return [dict(r) for r in c.execute(
            "SELECT role, content FROM chat WHERE meeting_id = ? ORDER BY id", (meeting_id,)).fetchall()]


def clear_chat(meeting_id):
    with _db() as c:
        c.execute("DELETE FROM chat WHERE meeting_id = ?", (meeting_id,))


# ---------- settings (never stores API keys) ----------

def get_setting(key, default=None):
    with _db() as c:
        row = c.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row and row["value"] is not None else default


def set_setting(key, value):
    with _db() as c:
        c.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                  "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, str(value)))
