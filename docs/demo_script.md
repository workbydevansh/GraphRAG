# Demo Script

## 1. Opening

"This is GraphRAG Benchmark Lab, a dashboard for proving whether TigerGraph GraphRAG reduces token usage and cost compared with Basic RAG while preserving answer quality."

## 2. Show the Query Lab

Open `http://localhost:3000`.

Run:

```text
Which country is the city where Marie Curie was born the capital of?
```

Ground truth:

```text
Poland
```

Point out that the same question runs through LLM-only, Basic RAG, and TigerGraph GraphRAG.

## 3. Explain Mode Labels

"If live credentials are missing, the app labels the row as dev mode. It does not fake live TigerGraph, LLM, or evaluation metrics."

## 4. Compare Results

Show:

- Answers side by side
- Token totals
- Estimated cost
- Latency
- Judge state and BERTScore/recall
- Retrieved context and trace

## 5. Show the Dashboard

Open `/dashboard`.

Explain:

"These charts are generated from stored SQLite runs. The winner card summarizes GraphRAG token reduction, cost reduction, and quality delta against Basic RAG."

## 6. Run Batch Benchmark

Open `/reports` and click "Run batch", or run:

```powershell
python backend\scripts\run_batch_benchmark.py --limit 40
```

Then export JSON, CSV, or Markdown.

## 7. Closing

"The benchmark claim is only made from measured rows: GraphRAG must reduce tokens or cost against Basic RAG while matching or improving quality. The exports preserve the exact run conditions for judging."
