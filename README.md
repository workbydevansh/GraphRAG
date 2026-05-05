# GraphRAG Benchmark Lab

Full-stack benchmark product for the TigerGraph GraphRAG Inference Hackathon. It compares:

- `LLMOnlyPipeline`
- `BasicRAGPipeline`
- `GraphRAGPipeline`

Every pipeline returns the same production contract:

```json
{
  "pipeline_name": "string",
  "answer": "string",
  "contexts": [],
  "prompt_tokens": 0,
  "completion_tokens": 0,
  "total_tokens": 0,
  "latency_ms": 0,
  "estimated_cost_usd": 0,
  "metadata": {}
}
```

## Repository Structure

```text
frontend/              Next.js App Router dashboard
backend/app/           FastAPI API, routers, services, schemas, SQLite
backend/scripts/       HotpotQA prep, vector index, GraphRAG ingest, batch/report scripts
docs/                  Architecture, report template, blog draft, demo script
data/                  raw, processed, eval, indexes
```

## API Endpoints

- `GET /health`
- `POST /api/query/run-all`
- `POST /api/query/run-one`
- `POST /api/benchmark/run`
- `GET /api/benchmark/results`
- `GET /api/benchmark/summary`
- `GET /api/dataset/status`
- `POST /api/report/export`

## Quick Start

```powershell
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
cd frontend
npm install
```

Start backend from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path ".").Path
python -m uvicorn backend.app.main:app --reload --port 8000
```

Start frontend:

```powershell
cd frontend
npm run dev
```

Open `http://localhost:3000`.

## Dataset

```powershell
$env:PYTHONPATH = (Resolve-Path ".").Path
python backend\scripts\prepare_hotpotqa.py --target-tokens 2000000 --eval-size 40
python backend\scripts\build_vector_index.py
```

Offline wiring test:

```powershell
python backend\scripts\prepare_hotpotqa.py --allow-dev-seed
```

Dev seed rows are labelled and are not submission benchmark evidence.

## Benchmark Commands

Single query:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/api/query/run-all `
  -ContentType "application/json" `
  -Body '{"question":"Which country is the city where Marie Curie was born the capital of?","ground_truth":"Poland"}'
```

Batch:

```powershell
python backend\scripts\run_batch_benchmark.py --limit 40
python backend\scripts\generate_report.py --format markdown
```

## Docker

```powershell
docker compose up --build
```

The backend runs on `8000`, frontend on `3000`, and `./data` is mounted for durable artifacts.

