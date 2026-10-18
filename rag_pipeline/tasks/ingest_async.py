"""Async ingestion with job tracking."""

import asyncio
import hashlib
import sqlite3
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from rag_pipeline.pipeline import get_pipeline


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestJobDB:
    """SQLite-based job database for async ingestion."""

    def __init__(self, db_path: Path = Path("./data/ingest_jobs.db")):
        """Initialize job database.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_hash TEXT,
                    status TEXT NOT NULL,
                    progress INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                )
            """)
            conn.commit()

    def create_job(self, file_name: str, file_hash: str) -> str:
        """Create new ingest job.

        Args:
            file_name: Name of file being ingested.
            file_hash: SHA256 hash of file content.

        Returns:
            Job ID.
        """
        job_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO ingest_jobs
                (job_id, file_name, file_hash, status, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (job_id, file_name, file_hash, JobStatus.PENDING, now))
            conn.commit()

        return job_id

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job by ID.

        Args:
            job_id: Job identifier.

        Returns:
            Job dict or None if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM ingest_jobs WHERE job_id = ?",
                (job_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: int = 0,
        chunk_count: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Update job status.

        Args:
            job_id: Job identifier.
            status: New status.
            progress: Progress percentage.
            chunk_count: Number of chunks ingested.
            error: Error message if failed.
        """
        completed_at = None
        if status in (JobStatus.COMPLETED, JobStatus.FAILED):
            completed_at = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE ingest_jobs
                SET status = ?, progress = ?, chunk_count = ?,
                    error = ?, completed_at = ?
                WHERE job_id = ?
            """, (status, progress, chunk_count, error, completed_at, job_id))
            conn.commit()

    def get_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file.

        Args:
            file_path: Path to file.

        Returns:
            Hex digest of file hash.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


async def ingest_document_async(
    job_id: str,
    file_path: Path,
    force: bool = False,
) -> dict:
    """Async document ingestion with job tracking.

    Args:
        job_id: Job identifier.
        file_path: Path to file to ingest.
        force: Force ingest even if file previously ingested.

    Returns:
        Ingestion result dict.
    """
    job_db = IngestJobDB()
    job = job_db.get_job(job_id)

    if not job:
        return {"status": "error", "error": "Job not found"}

    try:
        job_db.update_job_status(job_id, JobStatus.PROCESSING)

        # Compute file hash for deduplication
        file_hash = job_db.get_file_hash(file_path)

        # Check if already ingested
        if not force and job["file_hash"] == file_hash:
            job_db.update_job_status(job_id, JobStatus.COMPLETED)
            return {
                "status": "skipped",
                "message": "File already ingested (same content hash)",
            }

        # Ingest document (runs in default executor - non-blocking)
        pipeline = get_pipeline()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            pipeline.ingest_document,
            file_path
        )

        chunk_count = result.get("chunk_count", 0)
        job_db.update_job_status(
            job_id,
            JobStatus.COMPLETED,
            progress=100,
            chunk_count=chunk_count,
        )

        return {
            "status": "completed",
            "job_id": job_id,
            "chunk_count": chunk_count,
            "file_hash": file_hash,
        }

    except Exception as e:
        job_db.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=str(e),
        )
        return {
            "status": "error",
            "job_id": job_id,
            "error": str(e),
        }
