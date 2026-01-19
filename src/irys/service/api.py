"""FastAPI REST API for Irys RLM service."""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
import httpx

from .config import ServiceConfig, get_config
from .models import (
    InvestigateRequest,
    InvestigateResponse,
    JobResult,
    JobStatus,
    SearchRequest,
    SearchResponse,
    SearchResult,
    HealthResponse,
    ErrorResponse,
)
from .s3_repository import S3Repository

logger = logging.getLogger(__name__)

# In-memory job storage (use Redis in production for multi-worker)
_jobs: dict[str, JobResult] = {}
_start_time: float = time.time()

# Version
VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    config = get_config()

    # Validate config on startup
    errors = config.validate()
    if errors and not config.debug:
        for error in errors:
            logger.error(f"Config error: {error}")

    # Create temp directory
    Path(config.temp_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Irys RLM Service v{VERSION} starting...")
    logger.info(f"S3 Bucket: {config.s3_bucket}")
    logger.info(f"Temp Dir: {config.temp_dir}")

    # Start cleanup task
    cleanup_task = asyncio.create_task(_cleanup_loop(config))

    yield

    # Cleanup on shutdown
    cleanup_task.cancel()
    logger.info("Shutting down...")


async def _cleanup_loop(config: ServiceConfig):
    """Background task to clean up expired temp files."""
    while True:
        await asyncio.sleep(60)  # Check every minute
        try:
            # Clean old jobs from memory
            now = datetime.now()
            expired = [
                job_id for job_id, job in _jobs.items()
                if job.completed_at and
                (now - job.completed_at).seconds > config.cleanup_after_seconds
            ]
            for job_id in expired:
                del _jobs[job_id]
                logger.debug(f"Cleaned up job {job_id}")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def create_app(config: Optional[ServiceConfig] = None) -> FastAPI:
    """Create FastAPI application."""
    config = config or get_config()

    app = FastAPI(
        title="Irys RLM API",
        description="Legal document investigation service with recursive language model",
        version=VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store config in app state
    app.state.config = config

    return app


# Create default app instance
app = create_app()


# === ENDPOINTS ===


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    config = get_config()

    # Check S3 connection
    s3_connected = False
    if config.s3_bucket:
        try:
            import boto3
            s3 = boto3.client("s3", region_name=config.s3_region)
            s3.head_bucket(Bucket=config.s3_bucket)
            s3_connected = True
        except Exception:
            pass

    # Check Gemini connection
    gemini_connected = bool(config.gemini_api_key)

    # Calculate temp storage usage
    temp_dir = Path(config.temp_dir)
    temp_size_mb = 0.0
    if temp_dir.exists():
        temp_size_mb = sum(
            f.stat().st_size for f in temp_dir.rglob("*") if f.is_file()
        ) / (1024 * 1024)

    # Count active jobs
    active_jobs = sum(
        1 for job in _jobs.values()
        if job.status in (JobStatus.PENDING, JobStatus.PROCESSING)
    )

    return HealthResponse(
        status="healthy",
        version=VERSION,
        gemini_connected=gemini_connected,
        s3_connected=s3_connected,
        active_jobs=active_jobs,
        temp_storage_mb=round(temp_size_mb, 2),
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.post(
    "/investigate",
    response_model=InvestigateResponse,
    tags=["Investigation"],
    responses={400: {"model": ErrorResponse}},
)
async def start_investigation(
    request: InvestigateRequest,
    background_tasks: BackgroundTasks,
):
    """Start a new document investigation.

    Downloads documents from S3, runs investigation using Gemini,
    and returns results asynchronously.
    """
    config = get_config()

    # Check concurrent job limit
    active_count = sum(
        1 for job in _jobs.values()
        if job.status in (JobStatus.PENDING, JobStatus.PROCESSING)
    )
    if active_count >= config.max_concurrent_jobs:
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent jobs. Max: {config.max_concurrent_jobs}",
        )

    # Generate job ID
    job_id = f"inv_{uuid.uuid4().hex[:12]}"

    # Create job record
    job = JobResult(
        job_id=job_id,
        status=JobStatus.PENDING,
        query=request.query,
        s3_prefix=request.s3_prefix,
        created_at=datetime.now(),
    )
    _jobs[job_id] = job

    # Start background investigation
    background_tasks.add_task(
        _run_investigation,
        job_id,
        request,
        config,
    )

    return InvestigateResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Investigation started",
        estimated_seconds=60,
    )


async def _run_investigation(
    job_id: str,
    request: InvestigateRequest,
    config: ServiceConfig,
):
    """Background task to run investigation."""
    job = _jobs[job_id]
    job.status = JobStatus.PROCESSING
    s3_repo = None
    temp_dir = None

    try:
        # Download documents from S3
        s3_repo = S3Repository(
            bucket=config.s3_bucket,
            prefix=request.s3_prefix,
            config=config,
        )
        temp_dir = await s3_repo.download_to_temp(job_id)

        # Run investigation
        from irys import Irys
        irys = Irys(api_key=config.gemini_api_key)

        result = await irys.investigate(
            query=request.query,
            repository=str(temp_dir),
        )

        # Extract results
        job.analysis = result.get("analysis", "")
        job.citations = result.get("citations", [])
        job.entities = result.get("entities", {})
        job.documents_processed = result.get("documents_read", 0)
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        job.duration_seconds = (
            job.completed_at - job.created_at).total_seconds()

        logger.info(f"Job {job_id} completed in {job.duration_seconds:.1f}s")

        # Call webhook if provided
        if request.callback_url:
            await _send_callback(request.callback_url, job)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now()

    finally:
        # Cleanup temp files
        if s3_repo and temp_dir:
            await s3_repo.cleanup(job_id)


async def _send_callback(url: str, job: JobResult):
    """Send results to callback URL."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                url,
                json=job.model_dump(mode="json"),
                timeout=30,
            )
            logger.info(f"Sent callback for job {job.job_id}")
    except Exception as e:
        logger.error(f"Callback failed for job {job.job_id}: {e}")


@app.get(
    "/investigate/{job_id}",
    response_model=JobResult,
    tags=["Investigation"],
    responses={404: {"model": ErrorResponse}},
)
async def get_investigation(job_id: str):
    """Get investigation status and results."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@app.get("/jobs", response_model=list[JobResult], tags=["Investigation"])
async def list_jobs(
    status: Optional[JobStatus] = None,
    limit: int = 20,
):
    """List recent investigation jobs."""
    jobs = list(_jobs.values())

    if status:
        jobs = [j for j in jobs if j.status == status]

    # Sort by created_at descending
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return jobs[:limit]


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Search"],
)
async def quick_search(request: SearchRequest):
    """Quick keyword search across documents."""
    config = get_config()
    job_id = f"search_{uuid.uuid4().hex[:8]}"

    try:
        # Download documents
        s3_repo = S3Repository(
            bucket=config.s3_bucket,
            prefix=request.s3_prefix,
            config=config,
        )
        temp_dir = await s3_repo.download_to_temp(job_id)

        # Run search
        from irys import quick_search as irys_search
        results = await irys_search(
            query=request.query,
            repository=str(temp_dir),
            api_key=config.gemini_api_key,
        )

        # Cleanup
        await s3_repo.cleanup(job_id)

        return SearchResponse(
            results=[SearchResult(**r) for r in results[:request.max_results]],
            total_matches=len(results),
            documents_searched=len(list(temp_dir.rglob("*"))),
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
