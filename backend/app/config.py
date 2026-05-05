from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "GraphRAG Benchmark Lab"
    environment: Literal["development", "test", "production"] = "development"
    frontend_origin: str = "http://localhost:3000"

    data_dir: Path = Field(default=REPO_ROOT / "data")
    raw_data_dir: Path = Field(default=REPO_ROOT / "data" / "raw")
    processed_data_dir: Path = Field(default=REPO_ROOT / "data" / "processed")
    eval_data_dir: Path = Field(default=REPO_ROOT / "data" / "eval")
    indexes_dir: Path = Field(default=REPO_ROOT / "data" / "indexes")
    database_path: Path = Field(default=REPO_ROOT / "data" / "benchmark_lab.sqlite")
    corpus_path: Path = Field(default=REPO_ROOT / "data" / "processed" / "corpus.jsonl")
    eval_set_path: Path = Field(default=REPO_ROOT / "data" / "eval" / "eval_set.json")
    chroma_path: Path = Field(default=REPO_ROOT / "data" / "indexes" / "chroma")
    report_dir: Path = Field(default=REPO_ROOT / "docs" / "generated_reports")

    llm_provider: Literal["mock", "openai", "openai_compatible", "ollama"] = "mock"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    llm_api_base: str = "https://api.openai.com/v1"
    llm_temperature: float = 0.0
    request_timeout_seconds: int = 60
    input_cost_per_1k: float = 0.00015
    output_cost_per_1k: float = 0.0006

    basic_rag_top_k: int = 8
    graphrag_top_k: int = 2
    max_context_chars: int = 1800
    max_graph_context_chars: int = 650

    enable_hf_eval: bool = True
    enable_llm_judge: bool = True
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"

    tigergraph_host: str | None = None
    tigergraph_username: str | None = None
    tigergraph_password: str | None = None
    tigergraph_secret: str | None = None
    tigergraph_graph_name: str = "TigerGraphRAG"
    tigergraph_restpp_port: int = 14240
    tigergraph_graphrag_url: str | None = None
    tigergraph_graphrag_api_key: str | None = None
    tigergraph_graphrag_path: str = "/query"
    tigergraph_method: Literal["hybrid", "community"] = "hybrid"

    default_batch_limit: int = 40

    def ensure_directories(self) -> None:
        for directory in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.eval_data_dir,
            self.indexes_dir,
            self.chroma_path,
            self.report_dir,
            self.database_path.parent,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings

