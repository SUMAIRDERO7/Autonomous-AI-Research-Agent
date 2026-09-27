#!/usr/bin/env python3
"""
main.py
=======

Terminal client for the Autonomous Research Agent.

Usage:
    python main.py research "What are the main approaches to RAG in 2026?"
    python main.py research "..." --max-steps 8 --output report.md
    python main.py demo    # runs the full agent loop against fake tools, no keys needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.claude_client import ClaudeClient, ClaudeClientError
from src.research_service import ResearchService
from src.tools import PageReaderTool, TavilySearchTool


def _print_run(run) -> None:
    """Print a completed run's trace, report, and sources."""
    print("\n" + "=" * 70)
    print("AGENT TRACE")
    print("=" * 70)
    for step in run.steps:
        marker = "✓" if step.succeeded else "✗"
        target = f" [{step.action_input[:60]}]" if step.action_input else ""
        print(f"\n{marker} Step {step.number}: {step.action}{target}")
        print(f"   thought: {step.thought}")
        observation = step.observation.replace("\n", "\n            ")
        print(f"   result:  {observation[:400]}")

    print("\n" + "=" * 70)
    print(f"STOPPED: {run.stopped_reason}  |  {run.step_count} steps  |  {len(run.citations)} sources")
    print("=" * 70)
    print("\n" + run.to_markdown())


def cmd_research(args: argparse.Namespace) -> int:
    try:
        client = ClaudeClient()
    except ClaudeClientError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    service = ResearchService(
        client, tools=[TavilySearchTool(), PageReaderTool()], max_steps=args.max_steps
    )

    print(f"🔎 Researching: {args.question}")
    print(f"   Step budget: {args.max_steps} | Tools: {', '.join(service.tool_names)}")

    try:
        run = service.research(args.question)
    except (ClaudeClientError, ValueError, RuntimeError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    _print_run(run)

    if args.output:
        path = service.save_report(run, args.output)
        print(f"\n✅ Report saved to {path}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the full agent loop against fake tools and a scripted model.

    This exists so the agent's control flow — searching, reading,
    deduplicating sources, stopping — can be demonstrated end-to-end with
    no API keys and no network access at all.
    """
    from src.agent import ResearchAgent
    from src.tools import ToolRegistry

    def fake_search(query, api_key, max_results):
        return [
            {"title": "RAG Survey 2026", "url": "https://example.com/rag-survey",
             "content": "Overview of retrieval-augmented generation approaches."},
            {"title": "Hybrid Retrieval Methods", "url": "https://example.com/hybrid",
             "content": "Combining dense and sparse retrieval improves recall."},
        ]

    def fake_fetch(url):
        return (
            "Retrieval-augmented generation combines a retriever with a generator. "
            "Recent work focuses on hybrid dense/sparse retrieval and re-ranking. "
        ) * 10

    class ScriptedClient:
        def __init__(self):
            self.turn = 0

        def complete(self, system_prompt, user_message):
            if "research analyst" in system_prompt.lower():
                return (
                    "## Executive summary\n"
                    "Retrieval-augmented generation in 2026 centres on hybrid retrieval "
                    "and re-ranking [1][2].\n\n"
                    "## Hybrid retrieval\n"
                    "Combining dense and sparse retrieval improves recall over either alone [2]."
                )
            self.turn += 1
            if self.turn == 1:
                return "THOUGHT: I need an overview first.\nACTION: web_search\nINPUT: RAG approaches 2026"
            if self.turn == 2:
                return "THOUGHT: Read the survey in depth.\nACTION: read_page\nINPUT: https://example.com/rag-survey"
            if self.turn == 3:
                return "THOUGHT: Try that same search again.\nACTION: web_search\nINPUT: RAG approaches 2026"
            return "THOUGHT: I have enough to answer.\nACTION: finish\nINPUT: done"

    tools = ToolRegistry([
        TavilySearchTool(search_fn=fake_search, api_key="demo"),
        PageReaderTool(fetch_fn=fake_fetch),
    ])
    agent = ResearchAgent(ScriptedClient(), tools, max_steps=args.max_steps)

    print("🧪 DEMO MODE — fake tools, scripted model, no API keys or network needed.")
    print("   Watch step 3: the agent repeats a search and the loop blocks it.\n")

    run = agent.research("What are the main approaches to RAG in 2026?")
    _print_run(run)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous research agent.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("research", help="Research a question with live tools.")
    p.add_argument("question", help="The research question.")
    p.add_argument("--max-steps", type=int, default=6, help="Hard cap on tool calls.")
    p.add_argument("--output", help="Save the report to this markdown file.")

    p = sub.add_parser("demo", help="Run the agent loop offline with fake tools.")
    p.add_argument("--max-steps", type=int, default=6)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return {"research": cmd_research, "demo": cmd_demo}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
