from __future__ import annotations

import math


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, math.ceil(len(text) / 4))


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_cost_per_1k: float,
    output_cost_per_1k: float,
) -> float:
    return round(
        (prompt_tokens / 1000) * input_cost_per_1k
        + (completion_tokens / 1000) * output_cost_per_1k,
        8,
    )

