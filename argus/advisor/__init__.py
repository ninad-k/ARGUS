"""AI advisor: pluggable LLM backends, a single-pass daily-pick reviewer, and
an optional multi-persona council."""

from argus.advisor.council import (
    CouncilVerdict,
    CouncilVote,
    council_review,
    council_to_pick_verdicts,
)
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
from argus.advisor.personas import DEFAULT_COUNCIL, Persona, get_personas
from argus.advisor.pick_reviewer import PickVerdict, apply_verdicts, review_picks

__all__ = [
    "DEFAULT_COUNCIL",
    "AnthropicBackend",
    "CouncilVerdict",
    "CouncilVote",
    "LLMBackend",
    "LLMRequest",
    "LLMResponse",
    "NoOpBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "OpenAICompatibleBackend",
    "Persona",
    "PickVerdict",
    "apply_verdicts",
    "build_backend",
    "council_review",
    "council_to_pick_verdicts",
    "get_personas",
    "parse_llm_json",
    "review_picks",
]
