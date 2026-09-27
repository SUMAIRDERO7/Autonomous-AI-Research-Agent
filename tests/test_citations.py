"""Tests for src.citations."""

from __future__ import annotations

import pytest

from src.citations import CitationRegistry
from src.tools import SearchResult


def _source(url: str, title: str = "") -> SearchResult:
    return SearchResult(title=title or f"Page at {url}", url=url)


class TestAdd:
    def test_first_source_gets_number_one(self):
        registry = CitationRegistry()
        citation = registry.add(_source("https://a.com"))
        assert citation.number == 1

    def test_numbers_increment_for_new_urls(self):
        registry = CitationRegistry()
        assert registry.add(_source("https://a.com")).number == 1
        assert registry.add(_source("https://b.com")).number == 2
        assert registry.add(_source("https://c.com")).number == 3

    def test_same_url_returns_same_citation_number(self):
        registry = CitationRegistry()
        first = registry.add(_source("https://a.com", "Title One"))
        second = registry.add(_source("https://a.com", "A Different Title"))
        assert first.number == second.number
        assert len(registry) == 1

    def test_dedup_keeps_the_original_title(self):
        # Deduping by URL means the first title wins — a later, worse
        # title for the same page must not overwrite it.
        registry = CitationRegistry()
        registry.add(_source("https://a.com", "Good Title"))
        again = registry.add(_source("https://a.com", "https://a.com"))
        assert again.title == "Good Title"

    def test_numbers_stay_stable_as_more_sources_arrive(self):
        registry = CitationRegistry()
        first = registry.add(_source("https://a.com"))
        registry.add(_source("https://b.com"))
        registry.add(_source("https://c.com"))
        assert registry.add(_source("https://a.com")).number == first.number

    def test_source_without_url_raises(self):
        registry = CitationRegistry()
        with pytest.raises(ValueError):
            registry.add(SearchResult(title="No URL", url="  "))

    def test_falls_back_to_url_when_title_empty(self):
        registry = CitationRegistry()
        citation = registry.add(SearchResult(title="  ", url="https://a.com"))
        assert citation.title == "https://a.com"


class TestAddAll:
    def test_registers_every_source_with_a_url(self):
        registry = CitationRegistry()
        citations = registry.add_all([_source("https://a.com"), _source("https://b.com")])
        assert len(citations) == 2
        assert len(registry) == 2

    def test_silently_skips_sources_without_urls(self):
        registry = CitationRegistry()
        citations = registry.add_all([
            SearchResult(title="No URL", url=""),
            _source("https://b.com"),
        ])
        assert len(citations) == 1
        assert len(registry) == 1

    def test_empty_list_is_a_no_op(self):
        registry = CitationRegistry()
        assert registry.add_all([]) == []
        assert len(registry) == 0


class TestRendering:
    def test_citations_are_ordered_by_number(self):
        registry = CitationRegistry()
        registry.add_all([_source("https://c.com"), _source("https://a.com"), _source("https://b.com")])
        numbers = [c.number for c in registry.citations]
        assert numbers == [1, 2, 3]

    def test_to_markdown_lists_every_source(self):
        registry = CitationRegistry()
        registry.add(_source("https://a.com", "Alpha"))
        registry.add(_source("https://b.com", "Beta"))
        markdown = registry.to_markdown()
        assert "[1] Alpha — https://a.com" in markdown
        assert "[2] Beta — https://b.com" in markdown

    def test_to_markdown_handles_empty_registry(self):
        assert "No sources" in CitationRegistry().to_markdown()

    def test_context_block_includes_numbers_and_urls(self):
        registry = CitationRegistry()
        registry.add(_source("https://a.com", "Alpha"))
        block = registry.context_block()
        assert "[1] Alpha (https://a.com)" in block

    def test_context_block_handles_empty_registry(self):
        assert CitationRegistry().context_block() == "(no sources)"

    def test_citation_to_markdown_format(self):
        registry = CitationRegistry()
        citation = registry.add(_source("https://a.com", "Alpha"))
        assert citation.to_markdown() == "[1] Alpha — https://a.com"
