# GraphRAG Benchmark Report

## Run Configuration

- Date:
- Dataset:
- Corpus token count:
- Eval question count:
- LLM provider/model:
- Basic RAG index:
- TigerGraph GraphRAG deployment:
- Notes about any `dev` or `mixed` rows:

## Summary

| Metric | LLM-only | Basic RAG | TigerGraph GraphRAG |
| --- | ---: | ---: | ---: |
| Avg prompt tokens | | | |
| Avg completion tokens | | | |
| Avg total tokens | | | |
| Avg latency | | | |
| Avg estimated cost | | | |
| Avg BERTScore F1 | | | |
| Avg judge score | | | |

## Claim

TigerGraph GraphRAG reduced total tokens by `__%` and estimated cost by `__%` compared with Basic RAG while changing quality by `__` points.

## Evidence

Attach exported JSON/CSV/Markdown from:

- `POST http://localhost:8000/api/report/export` with `{ "format": "json" }`
- `POST http://localhost:8000/api/report/export` with `{ "format": "csv" }`
- `POST http://localhost:8000/api/report/export` with `{ "format": "markdown" }`

## Notable Examples

| Question | Basic RAG tokens | GraphRAG tokens | Basic quality | GraphRAG quality | Comment |
| --- | ---: | ---: | ---: | ---: | --- |
| | | | | | |

## Limitations

- Note any dev-mode rows.
- Note if the LLM provider estimated rather than returned token usage.
- Note if BERTScore could not run locally.
