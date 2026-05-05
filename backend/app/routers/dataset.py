from fastapi import APIRouter, Depends

from backend.app.config import Settings, get_settings
from backend.app.schemas import DatasetStatus
from backend.app.services.dataset import DatasetManager


router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.get("/status", response_model=DatasetStatus)
async def status(settings: Settings = Depends(get_settings)):
    return DatasetManager(settings).status()

