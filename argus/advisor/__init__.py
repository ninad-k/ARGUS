"""AI advisor: pluggable LLM backends plus a single-pass daily-pick reviewer."""

from argus.advisor.llm import (
    AnthropicBackend,
    LLMBackend,
    LLMRequest,
    LLMResponse,
    NoOpBackend,
    OllamaBackend,
    OpenAIBackend,
    OpenAICompatibleBackend,
    build_backend,
    parse_llm_json,
)
from argus.advisor.pick_reviewer import PickVerdict, apply_verdicts, review_picks

__all__ = [
    "AnthropicBackend",
    "LLMBackend",
    "LLMRequest",
    "LLMResponse",
    "NoOpBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "OpenAICompatibleBackend",
    "PickVerdict",
    "apply_verdicts",
    "build_backend",
    "parse_llm_json",
    "review_picks",
]
