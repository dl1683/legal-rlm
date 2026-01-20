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

    @staticmethod
    def parse_s3_url(url: str) -> tuple[str, str]:
        """Parse S3 URL into bucket and key.

        Supports:
            - s3://bucket/key
            - https://bucket.s3.region.amazonaws.com/key
            - https://s3.region.amazonaws.com/bucket/key

        Returns:
            Tuple of (bucket, key)

        Raises:
            ValueError if URL format is not recognized
        """
        import re

        # s3:// format
        if url.startswith("s3://"):
            parts = url[5:].split("/", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid S3 URL format: {url}")
            return parts[0], parts[1]

        # https://bucket.s3.region.amazonaws.com/key format
        match = re.match(
            r"https://([a-zA-Z0-9.-]+)\.s3\.[a-z0-9-]+\.amazonaws\.com/(.+)",
            url,
        )
        if match:
            return match.group(1), match.group(2)

        # https://s3.region.amazonaws.com/bucket/key format
        match = re.match(
            r"https://s3\.[a-z0-9-]+\.amazonaws\.com/([a-zA-Z0-9.-]+)/(.+)",
            url,
        )
        if match:
            return match.group(1), match.group(2)

        raise ValueError(f"Unrecognized S3 URL format: {url}")

    async def download_urls_to_temp(
        self,
        job_id: str,
        s3_urls: list[str],
    ) -> Path:
        """Download specific S3 URLs to temp directory.

        Args:
            job_id: Unique job identifier for tracking
            s3_urls: List of S3 URLs to download

        Returns:
            Path to temp directory containing downloaded files
        """
        # Create temp directory
        temp_dir = Path(self.config.temp_dir) / job_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Track for cleanup
        self._temp_dirs[job_id] = (temp_dir, time.time())

        # Check limits
        if len(s3_urls) > self.config.max_documents_per_job:
            raise ValueError(
                f"Too many documents ({len(s3_urls)}). "
                f"Max: {self.config.max_documents_per_job}"
            )

        logger.info(f"Downloading {len(s3_urls)} documents for job {job_id}")

        # Download each URL
        downloaded = 0
        for url in s3_urls:
            try:
                bucket, key = self.parse_s3_url(url)
                await self._download_url(bucket, key, temp_dir)
                downloaded += 1
            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")

        logger.info(f"Downloaded {downloaded}/{len(s3_urls)} documents to {temp_dir}")
        return temp_dir

    async def _download_url(
        self,
        bucket: str,
        key: str,
        dest_dir: Path,
    ) -> Path:
        """Download a single file from S3 by bucket and key."""
        # Use filename from key, preserving subdirectory structure
        filename = Path(key).name
        dest_path = dest_dir / filename

        def _download():
            # Check file size first
            head = self._s3.head_object(Bucket=bucket, Key=key)
            size_mb = head["ContentLength"] / (1024 * 1024)

            if size_mb > self.config.max_document_size_mb:
                logger.warning(
                    f"Skipping {key}: {size_mb:.1f}MB exceeds limit"
                )
                return None

            self._s3.download_file(bucket, key, str(dest_path))
            return dest_path

        return await asyncio.to_thread(_download)

    async def upload_files(
        self,
        job_id: str,
        files: list[tuple[str, bytes]],
        prefix: str = "uploads",
    ) -> str:
        """Upload files to S3.

        Args:
            job_id: Unique job identifier
            files: List of (filename, content) tuples
            prefix: S3 prefix for uploads (default: "uploads")

        Returns:
            S3 prefix where files were uploaded
        """
        upload_prefix = f"{prefix}/{job_id}"

        def _upload():
            uploaded = 0
            for filename, content in files:
                key = f"{upload_prefix}/{filename}"
                self._s3.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=content,
                )
                uploaded += 1
                logger.debug(f"Uploaded {filename} to s3://{self.bucket}/{key}")
            return uploaded

        count = await asyncio.to_thread(_upload)
        logger.info(f"Uploaded {count} files to s3://{self.bucket}/{upload_prefix}/")
        return upload_prefix

    async def delete_prefix(self, prefix: str) -> int:
        """Delete all objects under a prefix.

        Args:
            prefix: S3 prefix to delete

        Returns:
            Number of objects deleted
        """
        def _delete():
            deleted = 0
            paginator = self._s3.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                objects = page.get("Contents", [])
                if objects:
                    delete_keys = [{"Key": obj["Key"]} for obj in objects]
                    self._s3.delete_objects(
                        Bucket=self.bucket,
                        Delete={"Objects": delete_keys},
                    )
                    deleted += len(delete_keys)

            return deleted

        count = await asyncio.to_thread(_delete)
        logger.info(f"Deleted {count} objects from s3://{self.bucket}/{prefix}")
        return count

