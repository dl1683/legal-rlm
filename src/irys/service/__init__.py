"""Irys RLM Production Service.

FastAPI-based REST API for document investigation with S3 integration.
Designed for lightweight deployment on small instances.
"""

from .api import app, create_app
from .config import ServiceConfig, get_config
from .s3_repository import S3Repository

__all__ = [
    "app",
    "create_app",
    "ServiceConfig",
    "get_config",
    "S3Repository",
]

