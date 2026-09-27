"""Tests for src.tools."""

from __future__ import annotations

import pytest

from src.tools import (
    PageReaderTool,
    SearchResult,
    TavilySearchTool,
    ToolError,
    ToolRegistry,
    _http_fetch_text,
    _tavily_http_search,
)


def _fake_search(results):
    def _search(query, api_key, max_results):
        return results
    return _search


class TestTavilySearchTool:
    def test_rejects_empty_query(self):
        tool = TavilySearchTool(search_fn=_fake_search([]), api_key="k")
        with pytest.raises(ToolError):
            tool.run("   ")

    def test_name_and_description(self):
        tool = TavilySearchTool(search_fn=_fake_search([]), api_key="k")
        assert tool.name == "web_search"
        assert "Search the web" in tool.description

    def test_returns_structured_sources(self):
        tool = TavilySearchTool(
            search_fn=_fake_search([
                {"title": "Page A", "url": "https://a.com", "content": "About A"},
                {"title": "Page B", "url": "https://b.com", "content": "About B"},
            ]),
            api_key="k",
        )
        result = tool.run("query")
        assert len(result.sources) == 2
        assert result.sources[0].url == "https://a.com"
        assert "Page A" in result.output

    def test_skips_results_without_url(self):
        tool = TavilySearchTool(
            search_fn=_fake_search([
                {"title": "No URL", "content": "x"},
                {"title": "Has URL", "url": "https://b.com", "content": "y"},
            ]),
            api_key="k",
        )
        result = tool.run("query")
        assert len(result.sources) == 1

    def test_empty_results_reported_not_crashed(self):
        tool = TavilySearchTool(search_fn=_fake_search([]), api_key="k")
        result = tool.run("query")
        assert "No results" in result.output
        assert result.sources == []

    def test_falls_back_to_url_when_title_missing(self):
        tool = TavilySearchTool(
            search_fn=_fake_search([{"url": "https://a.com", "content": "x"}]), api_key="k"
        )
        assert tool.run("q").sources[0].title == "https://a.com"

    def test_backend_failure_wrapped_in_toolerror(self):
        def _boom(query, api_key, max_results):
            raise RuntimeError("network down")

        tool = TavilySearchTool(search_fn=_boom, api_key="k")
        with pytest.raises(ToolError, match="web_search failed"):
            tool.run("query")

    def test_reads_api_key_from_environment(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "env-key")
        captured = {}

        def _capture(query, api_key, max_results):
            captured["key"] = api_key
            return []

        TavilySearchTool(search_fn=_capture).run("q")
        assert captured["key"] == "env-key"


class TestPageReaderTool:
    def test_rejects_empty_url(self):
        tool = PageReaderTool(fetch_fn=lambda url: "text")
        with pytest.raises(ToolError):
            tool.run("")

    def test_rejects_non_http_url(self):
        tool = PageReaderTool(fetch_fn=lambda url: "text")
        with pytest.raises(ToolError, match="http"):
            tool.run("ftp://example.com/file")

    def test_returns_page_text_and_source(self):
        tool = PageReaderTool(fetch_fn=lambda url: "The full page content here.")
        result = tool.run("https://example.com/page")
        assert "full page content" in result.output
        assert result.sources[0].url == "https://example.com/page"

    def test_truncates_long_pages(self):
        tool = PageReaderTool(fetch_fn=lambda url: "x" * 10_000, max_chars=100)
        result = tool.run("https://example.com")
        assert "[truncated]" in result.output
        assert len(result.output) < 200

    def test_empty_page_reported_not_crashed(self):
        tool = PageReaderTool(fetch_fn=lambda url: "   ")
        result = tool.run("https://example.com")
        assert "no readable text" in result.output.lower()

    def test_fetch_failure_wrapped_in_toolerror(self):
        def _boom(url):
            raise RuntimeError("404")

        tool = PageReaderTool(fetch_fn=_boom)
        with pytest.raises(ToolError, match="read_page failed"):
            tool.run("https://example.com")


class TestToolRegistry:
    def test_rejects_empty_tool_list(self):
        with pytest.raises(ValueError):
            ToolRegistry([])

    def test_rejects_duplicate_tool_names(self):
        a = PageReaderTool(fetch_fn=lambda url: "x")
        b = PageReaderTool(fetch_fn=lambda url: "y")
        with pytest.raises(ValueError, match="Duplicate"):
            ToolRegistry([a, b])

    def test_names_lists_registered_tools(self):
        registry = ToolRegistry([
            TavilySearchTool(search_fn=_fake_search([]), api_key="k"),
            PageReaderTool(fetch_fn=lambda url: "x"),
        ])
        assert set(registry.names) == {"web_search", "read_page"}

    def test_describe_includes_every_tool(self):
        registry = ToolRegistry([
            TavilySearchTool(search_fn=_fake_search([]), api_key="k"),
            PageReaderTool(fetch_fn=lambda url: "x"),
        ])
        described = registry.describe()
        assert "web_search" in described
        assert "read_page" in described

    def test_run_dispatches_to_correct_tool(self):
        registry = ToolRegistry([PageReaderTool(fetch_fn=lambda url: "page text")])
        result = registry.run("read_page", "https://example.com")
        assert "page text" in result.output

    def test_unknown_tool_raises_with_available_names(self):
        registry = ToolRegistry([PageReaderTool(fetch_fn=lambda url: "x")])
        with pytest.raises(ToolError, match="read_page"):
            registry.run("nonexistent_tool", "input")


class TestRealBackends:
    """The live backends aren't callable here (no network allowlist for them),
    but their guard clauses are still worth verifying."""

    def test_tavily_search_without_key_raises_clear_error(self):
        with pytest.raises(ToolError, match="TAVILY_API_KEY"):
            _tavily_http_search("query", "", 5)

    def test_tavily_search_posts_and_returns_results(self, monkeypatch):
        captured = {}

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"results": [{"title": "T", "url": "https://u.com"}]}

        def _fake_post(url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            return _FakeResponse()

        import requests

        monkeypatch.setattr(requests, "post", _fake_post)
        results = _tavily_http_search("my query", "my-key", 7)

        assert captured["url"] == "https://api.tavily.com/search"
        assert captured["json"]["query"] == "my query"
        assert captured["json"]["max_results"] == 7
        assert results[0]["url"] == "https://u.com"

    def test_http_fetch_strips_tags_and_scripts(self, monkeypatch):
        class _FakeResponse:
            text = "<html><script>bad()</script><p>Hello   world</p></html>"

            def raise_for_status(self):
                pass

        import requests

        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse())
        text = _http_fetch_text("https://example.com")
        assert "bad()" not in text
        assert "Hello world" in text
        assert "<" not in text


class TestSearchResult:
    def test_defaults_snippet_to_empty(self):
        result = SearchResult(title="T", url="https://u.com")
        assert result.snippet == ""
