"""Real OpenAI-backed LLM client implementing the pipeline's LLMClient interface.

Drop-in replacement for :class:`~src.llm.mock_client.MockLLMClient` in
``full_agentic`` mode. The agents build a fully-rendered prompt (system + user,
already containing JSON-schema instructions) and call ``generate(prompt)``;
this client forwards that prompt to an OpenAI chat model and returns the raw
text response, which the agents parse.

Design notes
------------
* The ``openai`` package is imported **lazily** inside :meth:`_ensure_client`,
  so importing this module (e.g. in unit tests) never requires the SDK and
  never touches the network. Tests monkeypatch the client and must not call
  the real API.
* JSON mode (``response_format={"type": "json_object"}``) is enabled by default
  because every agent prompt requests a JSON object — this makes parse failures
  (which plagued the mock ``label_echo`` client) far less likely.
* Token usage is accumulated across calls so the caller can report cost.
* Any backend error is wrapped in :class:`~src.llm.base_client.LLMClientError`
  so the agents' existing error handling applies unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.llm.base_client import LLMClient, LLMClientError

log = logging.getLogger(__name__)

# Per-1M-token list prices (USD). gpt-4o-mini as of the pilot; override via
# PRICING for other models. Used only for the convenience cost estimate in
# usage_summary(); actual token counts come from the API response.
PRICING_USD_PER_1M = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
}


class OpenAIClient(LLMClient):
    """OpenAI chat-completions backend for the multi-agent pipeline.

    Parameters
    ----------
    model:
        OpenAI model id (default ``"gpt-4o-mini"``).
    api_key:
        Explicit key. When ``None`` the SDK resolves ``OPENAI_API_KEY`` from the
        environment.
    base_url:
        Optional override (proxy / Azure-compatible gateway). ``None`` → SDK
        default / ``OPENAI_BASE_URL``.
    temperature:
        Sampling temperature (default ``0.0`` for reproducibility).
    max_tokens:
        Max completion tokens per call (default ``700`` — enough for the JSON
        responses the agents expect).
    json_mode:
        When ``True`` (default) request a JSON object response. Safe because all
        agent prompts instruct JSON output.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 700,
        json_mode: bool = True,
        timeout: float = 60.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.timeout = timeout
        self.logger = logger or log

        self._api_key = api_key
        self._base_url = base_url
        self._client = None  # lazily constructed OpenAI() instance

        # Usage counters (accumulated across all generate() calls).
        self.total_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # ------------------------------------------------------------------
    # Lazy client construction
    # ------------------------------------------------------------------

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise LLMClientError(
                "The 'openai' package is required for OpenAIClient. "
                "Install it with: pip install openai"
            ) from exc
        kwargs = {"timeout": self.timeout}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        try:
            self._client = OpenAI(**kwargs)
        except Exception as exc:  # auth/config errors surface here
            raise LLMClientError(f"Failed to initialise OpenAI client: {exc}") from exc
        return self._client

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        client = self._ensure_client()
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            request["response_format"] = {"type": "json_object"}

        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:
            raise LLMClientError(f"OpenAI API call failed: {exc}") from exc

        self.total_calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

        content = response.choices[0].message.content
        return content or ""

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def usage_summary(self) -> dict:
        """Return accumulated call/token counts plus a list-price cost estimate."""
        in_rate, out_rate = PRICING_USD_PER_1M.get(self.model, (0.0, 0.0))
        cost = (
            self.prompt_tokens / 1_000_000 * in_rate
            + self.completion_tokens / 1_000_000 * out_rate
        )
        return {
            "model": self.model,
            "calls": self.total_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "est_cost_usd": round(cost, 4),
            "pricing_note": (
                "List price estimate only — verify current OpenAI pricing."
            ),
        }
