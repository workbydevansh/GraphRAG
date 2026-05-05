from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import get_settings
from backend.app.routers import benchmark, dataset, query, report


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Benchmark LLM-only, Basic RAG, and TigerGraph GraphRAG pipelines.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "llm_provider": settings.llm_provider,
        "tigergraph_configured": bool(settings.tigergraph_graphrag_url),
    }


app.include_router(query.router)
app.include_router(benchmark.router)
app.include_router(dataset.router)
app.include_router(report.router)

