"""
tools.py
========

The agent's tool layer: a small ``Tool`` protocol plus the two tools this
research agent needs — web search and page reading.

Both tools take their network backend as an injected callable rather than
importing a specific HTTP/search client at module level. That's what lets
the agent loop be tested deterministically end-to-end with fake backends,
and what makes the search provider swappable (Tavily, Brave, SerpAPI, a
local index) without touching agent logic.

Environment note, stated honestly: this sandbox's network allowlist does
not include any web-search API host, so the agent could not be run
against a live search provider here. The tool interfaces and the full
agent loop are exercised end-to-end against fake backends in the test
suite, and ``TavilySearchTool`` is wired for real use wherever outbound
access to that host is allowed — see the README.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


class ToolError(RuntimeError):
    """Raised when a tool fails to execute."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One search hit returned by a search tool."""

    title: str
    url: str
    snippet: str = ""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of one tool invocation, as the agent observes it."""

    tool_name: str
    query: str
    output: str
    sources: list[SearchResult] = field(default_factory=list)


class Tool(Protocol):
    """The minimal interface the agent depends on for any tool."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def run(self, query: str) -> ToolResult: ...


class TavilySearchTool:
    """Web search backed by the Tavily API (or any injected equivalent).

    Args:
        search_fn: A callable taking ``(query, api_key, max_results)`` and
            returning a list of ``{"title", "url", "content"}`` dicts.
            Defaults to a real Tavily HTTP call. Tests inject a fake.
        api_key: Tavily API key. Falls back to the ``TAVILY_API_KEY``
            environment variable.
        max_results: Number of results to request per search.
    """

    def __init__(
        self,
        search_fn: Callable[[str, str, int], list[dict[str, Any]]] | None = None,
        api_key: str | None = None,
        max_results: int = 5,
    ) -> None:
        self._search_fn = search_fn or _tavily_http_search
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self.max_results = max_results

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for pages relevant to a query. Input: a short search query."

    def run(self, query: str) -> ToolResult:
        """Run a web search.

        Args:
            query: The search query.

        Returns:
            A :class:`ToolResult` whose ``output`` is a numbered, readable
            digest of the hits and whose ``sources`` carries the
            structured results for citation.

        Raises:
            ToolError: If the query is empty or the backend call fails.
        """
        if not query or not query.strip():
            raise ToolError("web_search requires a non-empty query")

        try:
            raw = self._search_fn(query.strip(), self._api_key, self.max_results)
        except Exception as exc:
            raise ToolError(f"web_search failed: {exc}") from exc

        results = [
            SearchResult(
                title=str(item.get("title", "")).strip() or item.get("url", "untitled"),
                url=str(item.get("url", "")).strip(),
                snippet=str(item.get("content", "")).strip(),
            )
            for item in raw
            if item.get("url")
        ]

        if not results:
            return ToolResult(tool_name=self.name, query=query, output="No results found.")

        lines = [
            f"{i}. {r.title} — {r.url}\n   {r.snippet[:300]}"
            for i, r in enumerate(results, start=1)
        ]
        return ToolResult(
            tool_name=self.name, query=query, output="\n".join(lines), sources=results
        )


class PageReaderTool:
    """Fetches one web page and returns its readable text.

    Args:
        fetch_fn: A callable taking a URL and returning the page's text.
            Defaults to a real HTTP fetch. Tests inject a fake.
        max_chars: Truncation limit — a full page easily exceeds what's
            useful to feed back into the agent's context.
    """

    def __init__(
        self,
        fetch_fn: Callable[[str], str] | None = None,
        max_chars: int = 4000,
    ) -> None:
        self._fetch_fn = fetch_fn or _http_fetch_text
        self.max_chars = max_chars

    @property
    def name(self) -> str:
        return "read_page"

    @property
    def description(self) -> str:
        return "Read the full text of one web page. Input: a single URL."

    def run(self, query: str) -> ToolResult:
        """Fetch and return a page's text.

        Args:
            query: The URL to read.

        Returns:
            A :class:`ToolResult` whose ``output`` is the page text
            (truncated to ``max_chars``) and whose ``sources`` holds the
            page as a single citable source.

        Raises:
            ToolError: If the URL is empty/not http(s), or the fetch fails.
        """
        url = (query or "").strip()
        if not url:
            raise ToolError("read_page requires a URL")
        if not url.startswith(("http://", "https://")):
            raise ToolError(f"read_page requires an http(s) URL, got {url!r}")

        try:
            text = self._fetch_fn(url)
        except Exception as exc:
            raise ToolError(f"read_page failed for {url}: {exc}") from exc

        text = (text or "").strip()
        if not text:
            return ToolResult(tool_name=self.name, query=url, output="Page had no readable text.")

        truncated = text[: self.max_chars]
        if len(text) > self.max_chars:
            truncated += "\n...[truncated]"

        return ToolResult(
            tool_name=self.name,
            query=url,
            output=truncated,
            sources=[SearchResult(title=url, url=url, snippet=text[:200])],
        )


class ToolRegistry:
    """Holds the tools available to an agent and dispatches calls by name.

    Args:
        tools: The tools to register. Names must be unique.

    Raises:
        ValueError: If ``tools`` is empty or contains duplicate names.
    """

    def __init__(self, tools: list[Tool]) -> None:
        if not tools:
            raise ValueError("ToolRegistry requires at least one tool")
        names = [t.name for t in tools]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate tool names: {names}")
        self._tools = {t.name: t for t in tools}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        """Render the tool list for inclusion in the agent's system prompt."""
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    def run(self, tool_name: str, query: str) -> ToolResult:
        """Dispatch a call to a registered tool.

        Args:
            tool_name: The tool's registered name.
            query: The tool's input.

        Returns:
            The tool's result.

        Raises:
            ToolError: If no tool with that name is registered, or the
                tool itself fails.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ToolError(f"Unknown tool {tool_name!r}. Available: {', '.join(self._tools)}")
        return tool.run(query)


# --------------------------------------------------------------- backends --


def _tavily_http_search(query: str, api_key: str, max_results: int) -> list[dict[str, Any]]:
    """Real Tavily search call. Requires ``TAVILY_API_KEY`` and network access."""
    import requests

    if not api_key:
        raise ToolError(
            "No Tavily API key found. Set TAVILY_API_KEY to enable live web search."
        )
    response = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": max_results},
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def _http_fetch_text(url: str) -> str:
    """Real page fetch, stripped to readable text. Requires network access."""
    import re

    import requests

    response = requests.get(url, timeout=30, headers={"User-Agent": "ResearchAgent/1.0"})
    response.raise_for_status()
    html = response.text

    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()
