"""S3-backed repository for document storage.

Downloads documents from S3 to temp storage for processing,
with automatic cleanup to keep disk usage low.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator, Optional
from contextlib import asynccontextmanager

import boto3
from botocore.exceptions import ClientError

from .config import ServiceConfig, get_config

logger = logging.getLogger(__name__)


class S3Repository:
    """S3-backed document repository with temp storage management."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        config: Optional[ServiceConfig] = None,
    ):
        """Initialize S3 repository.

        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix (folder path)
            config: Service configuration
        """
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.config = config or get_config()

        # Initialize S3 client
        self._s3 = boto3.client(
            "s3",
            region_name=self.config.s3_region,
            aws_access_key_id=self.config.aws_access_key_id,
            aws_secret_access_key=self.config.aws_secret_access_key,
        )

        # Temp storage tracking
        self._temp_dirs: dict[str, tuple[Path, float]] = {}

        logger.info(f"Initialized S3Repository: s3://{bucket}/{prefix}")

    async def list_documents(
        self,
        extensions: Optional[list[str]] = None,
    ) -> list[str]:
        """List documents in S3 prefix.

        Args:
            extensions: Filter by file extensions (e.g., ['.pdf', '.txt'])

        Returns:
            List of S3 keys (relative to prefix)
        """
        extensions = extensions or [".txt", ".pdf", ".docx", ".md"]

        def _list():
            documents = []
            paginator = self._s3.get_paginator("list_objects_v2")

            prefix = f"{self.prefix}/" if self.prefix else ""

            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if any(key.lower().endswith(ext) for ext in extensions):
                        # Return key relative to prefix
                        rel_key = key[len(prefix):] if prefix else key
                        documents.append(rel_key)

            return documents

        return await asyncio.to_thread(_list)

    async def download_to_temp(self, job_id: str) -> Path:
        """Download all documents to temp directory.

        Args:
            job_id: Unique job identifier for tracking

        Returns:
            Path to temp directory containing downloaded files
        """
        # Create temp directory
        temp_dir = Path(self.config.temp_dir) / job_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Track for cleanup
        self._temp_dirs[job_id] = (temp_dir, time.time())

        # List and download documents
        documents = await self.list_documents()
        logger.info(f"Downloading {len(documents)} documents for job {job_id}")

        # Check limits
        if len(documents) > self.config.max_documents_per_job:
            raise ValueError(
                f"Too many documents ({len(documents)}). "
                f"Max: {self.config.max_documents_per_job}"
            )

        # Download each document
        for doc_key in documents:
            await self._download_file(doc_key, temp_dir)

        logger.info(f"Downloaded {len(documents)} documents to {temp_dir}")
        return temp_dir

    async def _download_file(self, key: str, dest_dir: Path) -> Path:
        """Download a single file from S3."""
        s3_key = f"{self.prefix}/{key}" if self.prefix else key
        dest_path = dest_dir / key

        # Create parent directories
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        def _download():
            # Check file size first
            head = self._s3.head_object(Bucket=self.bucket, Key=s3_key)
            size_mb = head["ContentLength"] / (1024 * 1024)

            if size_mb > self.config.max_document_size_mb:
                logger.warning(f"Skipping {key}: {size_mb:.1f}MB exceeds limit")
                return None

            self._s3.download_file(self.bucket, s3_key, str(dest_path))
            return dest_path

        return await asyncio.to_thread(_download)

    async def cleanup(self, job_id: str) -> None:
        """Clean up temp directory for a job."""
        if job_id in self._temp_dirs:
            temp_dir, _ = self._temp_dirs.pop(job_id)
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")

    async def cleanup_expired(self) -> int:
        """Clean up expired temp directories. Returns count cleaned."""
        now = time.time()
        expired = []

        for job_id, (temp_dir, created_at) in self._temp_dirs.items():
            if now - created_at > self.config.cleanup_after_seconds:
                expired.append(job_id)

        for job_id in expired:
            await self.cleanup(job_id)

        return len(expired)

