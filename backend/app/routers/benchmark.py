from fastapi import APIRouter, Depends

from backend.app.database import BenchmarkDatabase
from backend.app.routers.dependencies import get_benchmark_service, get_database
from backend.app.schemas import BenchmarkRequest, QueryRunResponse, SummaryResponse
from backend.app.services.benchmark import BenchmarkService


router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


@router.post("/run", response_model=list[QueryRunResponse])
async def run_benchmark(
    payload: BenchmarkRequest,
    service: BenchmarkService = Depends(get_benchmark_service),
):
    return await service.run_batch(payload.limit, payload.eval_set_path, payload.save)


@router.get("/results", response_model=list[QueryRunResponse])
async def results(limit: int = 200, database: BenchmarkDatabase = Depends(get_database)):
    return database.list_runs(limit)


@router.get("/summary", response_model=SummaryResponse)
async def summary(service: BenchmarkService = Depends(get_benchmark_service)):
    return service.summary()

