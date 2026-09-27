"""
citations.py
=============

Tracks every source the agent touches during a research run and assigns
each a stable citation number.

Deduplication is by URL, not by title: the same page routinely appears
in several searches under slightly different titles, and citing it as
[3] and [7] in one report would be wrong. The number a URL gets the
first time it is seen is the number it keeps for the rest of the run,
so citations in the report body never shift as more sources arrive.

Pure Python with no LLM or network dependency — fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.tools import SearchResult


@dataclass(frozen=True, slots=True)
class Citation:
    """One numbered source in the final report."""

    number: int
    title: str
    url: str

    def to_markdown(self) -> str:
        """Render as a markdown reference-list line."""
        return f"[{self.number}] {self.title} — {self.url}"


class CitationRegistry:
    """Assigns and remembers citation numbers for sources across a run."""

    def __init__(self) -> None:
        self._by_url: dict[str, Citation] = {}

    def add(self, source: SearchResult) -> Citation:
        """Register a source, returning its citation (existing or new).

        Args:
            source: The source to register.

        Returns:
            The source's citation. If this URL was seen before, the
            original citation — same number, same title — is returned
            unchanged, so a URL's number is stable for the whole run.

        Raises:
            ValueError: If the source has no URL.
        """
        url = source.url.strip()
        if not url:
            raise ValueError("Cannot cite a source with no URL")

        existing = self._by_url.get(url)
        if existing is not None:
            return existing

        citation = Citation(
            number=len(self._by_url) + 1,
            title=source.title.strip() or url,
            url=url,
        )
        self._by_url[url] = citation
        return citation

    def add_all(self, sources: list[SearchResult]) -> list[Citation]:
        """Register several sources at once, skipping any without a URL.

        Args:
            sources: Sources to register.

        Returns:
            The citations for every source that had a URL, in input order.
        """
        citations = []
        for source in sources:
            if source.url.strip():
                citations.append(self.add(source))
        return citations

    @property
    def citations(self) -> list[Citation]:
        """All citations, ordered by citation number."""
        return sorted(self._by_url.values(), key=lambda c: c.number)

    def __len__(self) -> int:
        return len(self._by_url)

    def to_markdown(self) -> str:
        """Render the full reference list as markdown."""
        if not self._by_url:
            return "_No sources were consulted._"
        return "\n".join(c.to_markdown() for c in self.citations)

    def context_block(self) -> str:
        """Render the sources as a numbered block for the synthesis prompt.

        The agent's final report-writing call needs to know which number
        maps to which source so it can cite them inline as [1], [2], etc.
        """
        if not self._by_url:
            return "(no sources)"
        return "\n".join(f"[{c.number}] {c.title} ({c.url})" for c in self.citations)
