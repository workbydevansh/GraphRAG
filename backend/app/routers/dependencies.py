from backend.app.config import get_settings
from backend.app.database import BenchmarkDatabase
from backend.app.services.benchmark import BenchmarkService


def get_database() -> BenchmarkDatabase:
    settings = get_settings()
    return BenchmarkDatabase(settings.database_path)


def get_benchmark_service() -> BenchmarkService:
    settings = get_settings()
    return BenchmarkService(settings, get_database())

