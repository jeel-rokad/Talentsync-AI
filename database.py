"""
TalentSync AI — Database Layer v3.0
SQLite-backed persistent storage: candidates, jobs, skill cache,
batch jobs, emerging skills, and chat sessions.
"""

import sqlite3
import json
import os
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.environ.get("TALENTSYNC_DB", "talentsync.db")


class Database:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self._init_tables()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    filename TEXT,
                    name TEXT,
                    title TEXT,
                    email TEXT,
                    match_score INTEGER,
                    verdict TEXT,
                    parsed_json TEXT,
                    normalized_skills_json TEXT,
                    match_data_json TEXT,
                    job_description TEXT,
                    pipeline_time_ms INTEGER,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    company TEXT,
                    description TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS skill_cache (
                    raw_skill TEXT PRIMARY KEY,
                    normalized_json TEXT,
                    cached_at TEXT
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    candidate_id TEXT,
                    messages_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS batch_jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT DEFAULT 'pending',
                    total INTEGER DEFAULT 0,
                    completed_count INTEGER DEFAULT 0,
                    results_json TEXT DEFAULT '[]',
                    error TEXT DEFAULT '',
                    webhook_url TEXT DEFAULT '',
                    webhook_delivered INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS emerging_skills (
                    id TEXT PRIMARY KEY,
                    raw_skill TEXT UNIQUE,
                    detected_count INTEGER DEFAULT 1,
                    first_seen TEXT,
                    last_seen TEXT,
                    reviewed INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_candidates_created ON candidates(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_candidates_score ON candidates(match_score DESC);
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_batch_created ON batch_jobs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_emerging_count ON emerging_skills(detected_count DESC);
            """)

    # ─────────────────────────────────────────────────────────────────────────
    # Candidates
    # ─────────────────────────────────────────────────────────────────────────

    def save_candidate(self, data: Dict[str, Any]) -> str:
        parsed = data.get("parsed", {})
        match_data = data.get("match_data", {})
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO candidates
                (id, filename, name, title, email, match_score, verdict,
                 parsed_json, normalized_skills_json, match_data_json,
                 job_description, pipeline_time_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["id"],
                data.get("filename", ""),
                parsed.get("name", ""),
                parsed.get("title", ""),
                parsed.get("email", ""),
                match_data.get("match_score", 0),
                match_data.get("verdict", ""),
                json.dumps(parsed),
                json.dumps(data.get("normalized_skills", [])),
                json.dumps(match_data),
                data.get("job_description", ""),
                data.get("pipeline_time_ms", 0),
                data.get("created_at", datetime.utcnow().isoformat()),
            ))
        return data["id"]

    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
        return self._deserialize_candidate(dict(row)) if row else None

    def list_candidates(self, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM candidates ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, skip)
            ).fetchall()
        return [self._deserialize_candidate(dict(r)) for r in rows]

    def delete_candidate(self, candidate_id: str) -> bool:
        with self._conn() as conn:
            result = conn.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
        return result.rowcount > 0

    def update_candidate_match(self, candidate_id: str, match_data: Dict[str, Any]):
        with self._conn() as conn:
            conn.execute("""
                UPDATE candidates SET match_score=?, verdict=?, match_data_json=? WHERE id=?
            """, (match_data.get("match_score", 0), match_data.get("verdict", ""),
                  json.dumps(match_data), candidate_id))

    def _deserialize_candidate(self, row: Dict) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "filename": row.get("filename", ""),
            "name": row.get("name", ""),
            "title": row.get("title", ""),
            "email": row.get("email", ""),
            "match_score": row.get("match_score", 0),
            "verdict": row.get("verdict", ""),
            "created_at": row.get("created_at", ""),
            "pipeline_time_ms": row.get("pipeline_time_ms", 0),
            "parsed": json.loads(row.get("parsed_json") or "{}"),
            "normalized_skills": json.loads(row.get("normalized_skills_json") or "[]"),
            "match_data": json.loads(row.get("match_data_json") or "{}"),
            "job_description": row.get("job_description", ""),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Jobs
    # ─────────────────────────────────────────────────────────────────────────

    def save_job(self, data: Dict[str, Any]) -> str:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO jobs (id, title, company, description, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (data["id"], data.get("title", ""), data.get("company", ""),
                  data.get("description", ""), data.get("created_at", datetime.utcnow().isoformat())))
        return data["id"]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────────────
    # Skill Cache
    # ─────────────────────────────────────────────────────────────────────────

    def cache_skill(self, raw_skill: str, normalized: Dict[str, Any]):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO skill_cache (raw_skill, normalized_json, cached_at)
                VALUES (?, ?, ?)
            """, (raw_skill.lower(), json.dumps(normalized), datetime.utcnow().isoformat()))

    def get_cached_skill(self, raw_skill: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT normalized_json FROM skill_cache WHERE raw_skill = ?",
                (raw_skill.lower(),)
            ).fetchone()
        return json.loads(row["normalized_json"]) if row else None

    def list_skill_cache(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM skill_cache ORDER BY cached_at DESC").fetchall()
        results = []
        for r in rows:
            try:
                data = json.loads(r["normalized_json"])
                data["raw"] = r["raw_skill"]
                results.append(data)
            except Exception:
                pass
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Batch Jobs
    # ─────────────────────────────────────────────────────────────────────────

    def save_batch_job(self, data: Dict[str, Any]) -> str:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO batch_jobs
                (id, status, total, completed_count, results_json, error, webhook_url,
                 webhook_delivered, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["id"], data.get("status", "pending"), data.get("total", 0),
                0, "[]", "", data.get("webhook_url", ""), 0,
                data.get("created_at", now), now
            ))
        return data["id"]

    def get_batch_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM batch_jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["results"] = json.loads(d.get("results_json") or "[]")
        except Exception:
            d["results"] = []
        return d

    def update_batch_job(self, job_id: str, updates: Dict[str, Any]):
        updates["updated_at"] = datetime.utcnow().isoformat()
        set_clause = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [job_id]
        with self._conn() as conn:
            conn.execute(f"UPDATE batch_jobs SET {set_clause} WHERE id=?", values)

    def list_batch_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, status, total, completed_count, created_at, updated_at, webhook_url "
                "FROM batch_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────────────
    # Emerging Skills
    # ─────────────────────────────────────────────────────────────────────────

    def record_emerging_skill(self, raw_skill: str):
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id, detected_count FROM emerging_skills WHERE raw_skill = ?",
                (raw_skill.lower(),)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE emerging_skills SET detected_count=?, last_seen=? WHERE raw_skill=?",
                    (existing["detected_count"] + 1, now, raw_skill.lower())
                )
            else:
                import uuid
                conn.execute("""
                    INSERT INTO emerging_skills (id, raw_skill, detected_count, first_seen, last_seen, reviewed)
                    VALUES (?, ?, 1, ?, ?, 0)
                """, (str(uuid.uuid4()), raw_skill.lower(), now, now))

    def list_emerging_skills(self, reviewed: bool = False) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM emerging_skills WHERE reviewed=? ORDER BY detected_count DESC",
                (1 if reviewed else 0,)
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_emerging_skill_reviewed(self, skill_id: str) -> bool:
        with self._conn() as conn:
            result = conn.execute(
                "UPDATE emerging_skills SET reviewed=1 WHERE id=?", (skill_id,)
            )
        return result.rowcount > 0

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        with self._conn() as conn:
            total_cands = conn.execute("SELECT COUNT(*) as c FROM candidates").fetchone()["c"]
            total_jobs = conn.execute("SELECT COUNT(*) as c FROM jobs").fetchone()["c"]
            avg_score = conn.execute(
                "SELECT AVG(match_score) as a FROM candidates WHERE match_score > 0"
            ).fetchone()["a"]
            total_batches = conn.execute("SELECT COUNT(*) as c FROM batch_jobs").fetchone()["c"]
            emerging_count = conn.execute(
                "SELECT COUNT(*) as c FROM emerging_skills WHERE reviewed=0"
            ).fetchone()["c"]
        return {
            "total_candidates": total_cands,
            "total_jobs": total_jobs,
            "avg_match_score": round(avg_score or 0),
            "total_batch_jobs": total_batches,
            "pending_emerging_skills": emerging_count,
        }


db = Database()
