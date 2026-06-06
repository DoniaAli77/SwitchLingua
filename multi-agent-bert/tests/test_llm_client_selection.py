"""Tests for the --llm_client selection seam and the OpenAI client wrapper.

The OpenAI branch is exercised entirely through monkeypatching — no network
call is ever made and the `openai` package is not required. Default stays mock,
so the rest of the suite is unaffected.
"""

from __future__ import annotations

import pytest

import evaluate_pipeline
from src.llm.mock_client import MockLLMClient
from src.llm.openai_client import OpenAIClient
from src.llm.base_client import LLMClientError


# ---------------------------------------------------------------------------
# build_llm_client — selection seam
# ---------------------------------------------------------------------------

class TestBuildLLMClient:
    def test_default_is_mock(self):
        client = evaluate_pipeline.build_llm_client()
        assert isinstance(client, MockLLMClient)
        assert client.mode == "label_echo"

    def test_explicit_mock(self):
        client = evaluate_pipeline.build_llm_client(
            "mock", allowed_labels=["positive", "negative", "neutral"]
        )
        assert isinstance(client, MockLLMClient)
        assert client.allowed_labels == ["positive", "negative", "neutral"]

    def test_openai_routes_to_openai_client(self):
        client = evaluate_pipeline.build_llm_client("openai", llm_model="gpt-4o-mini")
        assert isinstance(client, OpenAIClient)
        assert client.model == "gpt-4o-mini"
        # Must not have constructed a real OpenAI SDK client yet (no network).
        assert client._client is None

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown llm_client"):
            evaluate_pipeline.build_llm_client("bogus")


# ---------------------------------------------------------------------------
# OpenAIClient — behaviour with a fully mocked SDK (no network)
# ---------------------------------------------------------------------------

class _FakeUsage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content, p, c):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(p, c)


class _FakeCompletions:
    def __init__(self, response, recorder):
        self._response = response
        self._recorder = recorder

    def create(self, **kwargs):
        self._recorder.append(kwargs)
        return self._response


class _FakeChat:
    def __init__(self, response, recorder):
        self.completions = _FakeCompletions(response, recorder)


class _FakeOpenAISDK:
    """Stand-in for an `openai.OpenAI()` instance."""

    def __init__(self, response, recorder):
        self.chat = _FakeChat(response, recorder)


class TestOpenAIClientMocked:
    def _install_fake(self, client, response, recorder):
        # Bypass _ensure_client's real `from openai import OpenAI`.
        client._client = _FakeOpenAISDK(response, recorder)

    def test_generate_returns_content_and_tracks_usage(self):
        client = OpenAIClient(model="gpt-4o-mini")
        recorder: list = []
        self._install_fake(
            client,
            _FakeResponse('{"label": "positive", "confidence": 0.8}', 120, 30),
            recorder,
        )

        out = client.generate("classify this as json")
        assert out == '{"label": "positive", "confidence": 0.8}'
        assert client.total_calls == 1
        assert client.prompt_tokens == 120
        assert client.completion_tokens == 30
        # JSON mode requested by default.
        assert recorder[0]["response_format"] == {"type": "json_object"}
        assert recorder[0]["model"] == "gpt-4o-mini"

    def test_usage_summary_cost(self):
        client = OpenAIClient(model="gpt-4o-mini")
        recorder: list = []
        self._install_fake(client, _FakeResponse("{}", 1_000_000, 1_000_000), recorder)
        client.generate("json")
        u = client.usage_summary()
        # 1M input @ $0.15 + 1M output @ $0.60 = $0.75
        assert u["est_cost_usd"] == pytest.approx(0.75)
        assert u["calls"] == 1

    def test_api_error_wrapped(self):
        client = OpenAIClient()

        class _Boom:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        raise RuntimeError("network down")

        client._client = _Boom()
        with pytest.raises(LLMClientError, match="OpenAI API call failed"):
            client.generate("json")

    def test_module_import_does_not_require_openai(self):
        # Importing the client module and constructing it must not import openai.
        c = OpenAIClient()
        assert c._client is None  # lazy — nothing constructed, nothing imported
