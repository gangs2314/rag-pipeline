"""Tests for async ingestion."""

import tempfile
from pathlib import Path

from rag_pipeline.tasks.ingest_async import IngestJobDB, JobStatus


def test_job_db_init():
    """Test IngestJobDB initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = IngestJobDB(db_path)
        assert db_path.exists()


def test_create_job():
    """Test creating an ingest job."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = IngestJobDB(Path(tmpdir) / "test.db")
        job_id = db.create_job("test.pdf", "hash123")
        assert job_id is not None


def test_get_job():
    """Test retrieving a job."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = IngestJobDB(Path(tmpdir) / "test.db")
        job_id = db.create_job("test.pdf", "hash123")
        job = db.get_job(job_id)
        assert job is not None
        assert job["file_name"] == "test.pdf"
        assert job["status"] == JobStatus.PENDING


def test_update_job_status():
    """Test updating job status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = IngestJobDB(Path(tmpdir) / "test.db")
        job_id = db.create_job("test.pdf", "hash123")
        db.update_job_status(job_id, JobStatus.PROCESSING, progress=50)
        job = db.get_job(job_id)
        assert job["status"] == JobStatus.PROCESSING
        assert job["progress"] == 50


def test_job_completion():
    """Test marking job as completed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = IngestJobDB(Path(tmpdir) / "test.db")
        job_id = db.create_job("test.pdf", "hash123")
        db.update_job_status(
            job_id,
            JobStatus.COMPLETED,
            progress=100,
            chunk_count=42,
        )
        job = db.get_job(job_id)
        assert job["status"] == JobStatus.COMPLETED
        assert job["chunk_count"] == 42
        assert job["completed_at"] is not None


def test_job_failure():
    """Test marking job as failed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = IngestJobDB(Path(tmpdir) / "test.db")
        job_id = db.create_job("test.pdf", "hash123")
        db.update_job_status(
            job_id,
            JobStatus.FAILED,
            error="File format not supported",
        )
        job = db.get_job(job_id)
        assert job["status"] == JobStatus.FAILED
        assert "not supported" in job["error"]


def test_file_hash():
    """Test file hash computation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = IngestJobDB(Path(tmpdir) / "test.db")
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("test content")

        hash1 = db.get_file_hash(test_file)
        hash2 = db.get_file_hash(test_file)

        # Same file should produce same hash
        assert hash1 == hash2
        # Hash should be hex string
        assert len(hash1) == 64  # SHA256 hex is 64 chars
