"""Tests for src.research_service and src.claude_client."""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import pytest

from src.claude_client import ClaudeClient, ClaudeClientError
from src.research_service import ResearchService
from src.tools import PageReaderTool, TavilySearchTool


class _ScriptedClient:
    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_message: str) -> str:
        self.calls.append((system_prompt, user_message))
        if "research analyst" in system_prompt.lower():
            return "## Findings\nA result [1]."
        return self._replies.pop(0) if self._replies else "THOUGHT: x\nACTION: finish\nINPUT: done"


def _fake_tools():
    return [
        TavilySearchTool(
            search_fn=lambda q, k, n: [{"title": "A", "url": "https://a.com", "content": "About"}],
            api_key="k",
        ),
        PageReaderTool(fetch_fn=lambda url: "Page text."),
    ]


class TestResearchService:
    def test_exposes_tool_names(self):
        service = ResearchService(_ScriptedClient([]), tools=_fake_tools())
        assert set(service.tool_names) == {"web_search", "read_page"}

    def test_exposes_max_steps(self):
        service = ResearchService(_ScriptedClient([]), tools=_fake_tools(), max_steps=4)
        assert service.max_steps == 4

    def test_research_returns_a_completed_run(self):
        client = _ScriptedClient([
            "THOUGHT: Search.\nACTION: web_search\nINPUT: query",
            "THOUGHT: Done.\nACTION: finish\nINPUT: done",
        ])
        service = ResearchService(client, tools=_fake_tools(), max_steps=4)
        run = service.research("What is X?")

        assert run.question == "What is X?"
        assert run.stopped_reason == "finished"
        assert len(run.citations) == 1

    def test_save_report_writes_markdown_with_sources(self, tmp_path):
        client = _ScriptedClient([
            "THOUGHT: Search.\nACTION: web_search\nINPUT: query",
            "THOUGHT: Done.\nACTION: finish\nINPUT: done",
        ])
        service = ResearchService(client, tools=_fake_tools(), max_steps=4)
        run = service.research("What is X?")

        out = service.save_report(run, tmp_path / "nested" / "report.md")

        assert out.exists()
        content = out.read_text()
        assert "What is X?" in content
        assert "## Sources" in content
        assert "https://a.com" in content

    def test_defaults_to_live_tools_when_none_given(self):
        # Constructing the default tools must not require network or keys —
        # only *calling* them does.
        service = ResearchService(_ScriptedClient([]))
        assert set(service.tool_names) == {"web_search", "read_page"}


# --------------------------------------------------------- claude client --


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeMessagesAPI:
    def __init__(self, reply_text="reply", raise_error=False):
        self._reply_text = reply_text
        self._raise_error = raise_error
        self.last_call_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        if self._raise_error:
            raise anthropic.APIError("boom", request=None, body=None)  # type: ignore[call-arg]
        return SimpleNamespace(content=[_FakeTextBlock(self._reply_text)])


class _FakeAnthropic:
    def __init__(self, api_key, reply_text="reply", raise_error=False):
        self.api_key = api_key
        self.messages = _FakeMessagesAPI(reply_text, raise_error)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


class TestClaudeClient:
    def test_raises_without_any_api_key(self):
        with pytest.raises(ClaudeClientError):
            ClaudeClient()

    def test_reads_key_from_environment(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        monkeypatch.setattr(anthropic, "Anthropic", lambda api_key: _FakeAnthropic(api_key))
        assert ClaudeClient()._client.api_key == "env-key"

    def test_explicit_key_wins(self, monkeypatch):
        monkeypatch.setattr(anthropic, "Anthropic", lambda api_key: _FakeAnthropic(api_key))
        assert ClaudeClient(api_key="explicit")._client.api_key == "explicit"

    def test_complete_returns_text(self, monkeypatch):
        monkeypatch.setattr(
            anthropic, "Anthropic", lambda api_key: _FakeAnthropic(api_key, "ACTION: finish")
        )
        assert ClaudeClient(api_key="k").complete("SYS", "msg") == "ACTION: finish"

    def test_complete_passes_arguments_through(self, monkeypatch):
        fake = _FakeAnthropic("k")
        monkeypatch.setattr(anthropic, "Anthropic", lambda api_key: fake)
        ClaudeClient(api_key="k", model="m", max_tokens=42).complete("SYS", "USER")
        assert fake.messages.last_call_kwargs["system"] == "SYS"
        assert fake.messages.last_call_kwargs["model"] == "m"
        assert fake.messages.last_call_kwargs["max_tokens"] == 42

    def test_api_errors_are_wrapped(self, monkeypatch):
        monkeypatch.setattr(
            anthropic, "Anthropic", lambda api_key: _FakeAnthropic(api_key, raise_error=True)
        )
        with pytest.raises(ClaudeClientError):
            ClaudeClient(api_key="k").complete("SYS", "msg")
