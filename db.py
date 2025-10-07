"""Simple SQLite wrapper for candidates, jobs, and matches with context manager support.""" 
import sqlite3
import json
from typing import Dict, List, Optional
from .config import DB_PATH
from .utils import logger

class DatabaseManager:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            title TEXT,
            company TEXT,
            text TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS candidates (
            id TEXT PRIMARY KEY,
            text TEXT,
            skills TEXT,
            education TEXT,
            experience TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT,
            job_id TEXT,
            score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        self.conn.commit()
        logger.info("Database initialized.")

    def insert_job(self, job: Dict) -> str:
        cur = self.conn.cursor()
        job_id = job.get('id') or job.get('job_id')
        cur.execute("INSERT OR REPLACE INTO jobs (id, title, company, text) VALUES (?, ?, ?, ?)",
                    (job_id, job.get('title') or job.get('job_title'),
                     job.get('company'), job.get('text') or job.get('job_description')))
        self.conn.commit()
        return job_id

    def list_jobs(self) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, title, company, text FROM jobs")
        rows = cur.fetchall()
        return [{'id': r[0], 'title': r[1], 'company': r[2], 'text': r[3]} for r in rows]

    def insert_candidate(self, candidate: Dict) -> str:
        cur = self.conn.cursor()
        cid = candidate.get('id') or candidate.get('candidate_id')
        cur.execute("INSERT OR REPLACE INTO candidates (id, text, skills, education, experience) VALUES (?, ?, ?, ?, ?)",
                    (cid, candidate.get('text'), json.dumps(candidate.get('skills', [])),
                     json.dumps(candidate.get('education', [])), json.dumps(candidate.get('experience', []))))
        self.conn.commit()
        return cid

    def insert_match(self, candidate_id: str, job_id: str, score: float):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO matches (candidate_id, job_id, score) VALUES (?, ?, ?)",
                    (candidate_id, job_id, float(score)))
        self.conn.commit()

    def get_matches_for_job(self, job_id: str):
        cur = self.conn.cursor()
        cur.execute("SELECT candidate_id, job_id, score FROM matches WHERE job_id = ? ORDER BY score DESC", (job_id,))
        return cur.fetchall()
