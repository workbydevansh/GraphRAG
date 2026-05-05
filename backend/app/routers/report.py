from fastapi import APIRouter, Depends

from backend.app.config import Settings, get_settings
from backend.app.routers.dependencies import get_benchmark_service
from backend.app.schemas import ReportExportRequest, ReportExportResponse
from backend.app.services.benchmark import BenchmarkService


router = APIRouter(prefix="/api/report", tags=["report"])


@router.post("/export", response_model=ReportExportResponse)
async def export_report(
    payload: ReportExportRequest,
    service: BenchmarkService = Depends(get_benchmark_service),
    settings: Settings = Depends(get_settings),
):
    export_format, content = service.export(payload.format, payload.limit)
    suffix = "md" if export_format == "markdown" else export_format
    path = settings.report_dir / f"benchmark_report.{suffix}"
    path.write_text(content, encoding="utf-8")
    return ReportExportResponse(format=export_format, path=str(path), content=content)

