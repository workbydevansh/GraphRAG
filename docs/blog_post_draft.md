# Building GraphRAG Benchmark Lab

Most RAG demos show a good answer. For the TigerGraph GraphRAG Inference Hackathon, I wanted to show a measurable answer: did GraphRAG actually reduce prompt tokens and cost compared with Basic RAG, and did answer quality hold up?

GraphRAG Benchmark Lab compares three pipelines:

1. LLM-only, as the control.
2. Basic RAG, using vector search over a HotpotQA-style Wikipedia corpus.
3. TigerGraph GraphRAG, using graph/vector retrieval through TigerGraph or a labelled local fallback during development.

The core design principle is honesty. If credentials are missing, the app does not pretend that TigerGraph or a paid LLM ran. It labels the row as `dev`, stores warnings, and keeps the UI usable for local testing. When live services are configured, token usage, latency, estimated cost, BERTScore, and judge results are stored in SQLite and surfaced in the dashboard.

The benchmark corpus comes from HotpotQA because multi-hop Wikipedia questions are exactly where graph structure should matter. Basic RAG tends to pass broad top-k chunks into the prompt. GraphRAG should retrieve compact connected evidence, reducing prompt size while preserving the facts needed to answer.

The dashboard makes the comparison judge-friendly:

- Side-by-side answers for all three pipelines
- Prompt/completion/total token counts
- Cost and latency charts
- Accuracy and judge-state reporting
- Winner insight card
- Query history and JSON/CSV/Markdown exports

The expected winning pattern is not "GraphRAG is always faster." Graph traversal can add work. The important result is that graph-grounded context can be smaller and more targeted, which lowers LLM token spend while maintaining or improving answer quality on multi-hop questions.

Next steps are to run the full 40-question eval set against a live TigerGraph GraphRAG deployment, tune graph retrieval parameters, and publish the exported benchmark report with every warning and configuration included.

