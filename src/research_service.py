"""
research_service.py
====================

The orchestration layer: builds a configured :class:`ResearchAgent` and
exposes the single operation both UIs need. Contains no I/O framework
code, so the Streamlit app and the CLI share it identically.
"""

from __future__ import annotations

from pathlib import Path

from src.agent import ResearchAgent, ResearchRun
from src.claude_client import ClaudeClient
from src.tools import PageReaderTool, TavilySearchTool, Tool, ToolRegistry

DEFAULT_MAX_STEPS = 6


class ResearchService:
    """Runs research questions through an autonomous agent.

    Args:
        client: The Claude client driving the agent.
        tools: Optional explicit tool list. Defaults to live web search
            plus a page reader; tests and offline demos inject fakes.
        max_steps: Hard cap on the agent's tool-calling steps.
    """

    def __init__(
        self,
        client: ClaudeClient,
        tools: list[Tool] | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        resolved_tools = tools if tools is not None else [TavilySearchTool(), PageReaderTool()]
        self.agent = ResearchAgent(
            client=client, tools=ToolRegistry(resolved_tools), max_steps=max_steps
        )

    @property
    def tool_names(self) -> list[str]:
        """Names of the tools available to the agent."""
        return self.agent.tools.names

    @property
    def max_steps(self) -> int:
        return self.agent.max_steps

    def research(self, question: str) -> ResearchRun:
        """Research a question end-to-end.

        Args:
            question: The research question.

        Returns:
            The completed run — report, trace, sources, and stop reason.
        """
        return self.agent.research(question)

    def save_report(self, run: ResearchRun, output_path: str | Path) -> Path:
        """Write a run's report (with references) to a markdown file.

        Args:
            run: The completed run.
            output_path: Where to write the ``.md`` file.

        Returns:
            The path written to.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(run.to_markdown())
        return path
