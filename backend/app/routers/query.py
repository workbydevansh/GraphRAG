from fastapi import APIRouter, Depends, HTTPException

from backend.app.schemas import QueryRunAllRequest, QueryRunOneRequest, QueryRunResponse
from backend.app.services.benchmark import BenchmarkService
from backend.app.routers.dependencies import get_benchmark_service


router = APIRouter(prefix="/api/query", tags=["query"])


@router.post("/run-all", response_model=QueryRunResponse)
async def run_all(
    payload: QueryRunAllRequest,
    service: BenchmarkService = Depends(get_benchmark_service),
):
    return await service.run_all(payload.question, payload.ground_truth, save=payload.save)


@router.post("/run-one", response_model=QueryRunResponse)
async def run_one(
    payload: QueryRunOneRequest,
    service: BenchmarkService = Depends(get_benchmark_service),
):
    pipelines = service.pipelines()
    if payload.pipeline_name not in pipelines:
        raise HTTPException(status_code=400, detail=f"Unknown pipeline: {payload.pipeline_name}")
    return await service.run_one(
        payload.pipeline_name,
        payload.question,
        payload.ground_truth,
        save=payload.save,
    )

