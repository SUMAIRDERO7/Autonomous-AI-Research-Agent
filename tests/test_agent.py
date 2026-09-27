"""Tests for src.agent.

The agent loop is driven by a scripted fake client, so every branch —
step budget exhaustion, repeated actions, tool failures, malformed
replies — is exercised deterministically with no network or API key.
"""

from __future__ import annotations

import pytest

from src.agent import AgentError, ResearchAgent, parse_action
from src.tools import PageReaderTool, TavilySearchTool, ToolRegistry


class _ScriptedClient:
    """Returns queued replies in order; any synthesis call returns a report."""

    def __init__(self, replies: list[str], report: str = "## Findings\nSomething [1]."):
        self._replies = list(replies)
        self.report = report
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_message: str) -> str:
        self.calls.append((system_prompt, user_message))
        if "research analyst" in system_prompt.lower():
            return self.report
        return self._replies.pop(0) if self._replies else "THOUGHT: done\nACTION: finish\nINPUT: done"


def _search_results():
    return [
        {"title": "Page A", "url": "https://a.com", "content": "About A"},
        {"title": "Page B", "url": "https://b.com", "content": "About B"},
    ]


def _registry(fetch_text: str = "Full page text.", search_results=None, search_boom: bool = False):
    def _search(query, api_key, max_results):
        if search_boom:
            raise RuntimeError("search backend down")
        return _search_results() if search_results is None else search_results

    return ToolRegistry([
        TavilySearchTool(search_fn=_search, api_key="k"),
        PageReaderTool(fetch_fn=lambda url: fetch_text),
    ])


class TestParseAction:
    def test_parses_all_three_fields(self):
        parsed = parse_action("THOUGHT: I need data.\nACTION: web_search\nINPUT: quantum 2026")
        assert parsed.thought == "I need data."
        assert parsed.action == "web_search"
        assert parsed.action_input == "quantum 2026"

    def test_detects_finish_action(self):
        assert parse_action("THOUGHT: enough\nACTION: finish\nINPUT: done").is_finish

    def test_finish_detection_is_case_insensitive(self):
        assert parse_action("THOUGHT: x\nACTION: FINISH\nINPUT: done").is_finish

    def test_missing_thought_gets_placeholder(self):
        parsed = parse_action("ACTION: web_search\nINPUT: query")
        assert parsed.thought == "(no thought given)"

    def test_missing_action_raises(self):
        with pytest.raises(AgentError):
            parse_action("THOUGHT: I was thinking\nINPUT: something")

    def test_tolerates_surrounding_prose(self):
        reply = "Sure!\nTHOUGHT: Need info.\nACTION: web_search\nINPUT: query\nHope that helps."
        parsed = parse_action(reply)
        assert parsed.action == "web_search"


class TestAgentInit:
    def test_rejects_non_positive_max_steps(self):
        with pytest.raises(ValueError):
            ResearchAgent(_ScriptedClient([]), _registry(), max_steps=0)


class TestResearchHappyPath:
    def test_rejects_empty_question(self):
        agent = ResearchAgent(_ScriptedClient([]), _registry())
        with pytest.raises(ValueError):
            agent.research("   ")

    def test_full_search_read_finish_cycle(self):
        client = _ScriptedClient([
            "THOUGHT: Search first.\nACTION: web_search\nINPUT: quantum computing",
            "THOUGHT: Read the top hit.\nACTION: read_page\nINPUT: https://a.com",
            "THOUGHT: Enough.\nACTION: finish\nINPUT: done",
        ])
        run = ResearchAgent(client, _registry(), max_steps=6).research("What's new in quantum?")

        assert run.stopped_reason == "finished"
        assert run.step_count == 3
        assert [s.action for s in run.steps] == ["web_search", "read_page", "finish"]
        assert all(s.succeeded for s in run.steps)

    def test_sources_are_collected_and_deduplicated(self):
        client = _ScriptedClient([
            "THOUGHT: Search.\nACTION: web_search\nINPUT: query one",
            "THOUGHT: Search again.\nACTION: web_search\nINPUT: query two",
            "THOUGHT: Done.\nACTION: finish\nINPUT: done",
        ])
        run = ResearchAgent(client, _registry(), max_steps=6).research("Question?")
        # Both searches return the same two URLs — they must dedupe to 2.
        assert len(run.citations) == 2

    def test_report_includes_question_and_sources(self):
        client = _ScriptedClient(["THOUGHT: Search.\nACTION: web_search\nINPUT: q"])
        run = ResearchAgent(client, _registry(), max_steps=2).research("My Question?")
        markdown = run.to_markdown()
        assert "My Question?" in markdown
        assert "## Sources" in markdown
        assert "https://a.com" in markdown


class TestStepBudget:
    def test_loop_never_exceeds_max_steps(self):
        # The model never says finish — Python must stop it anyway.
        never_finishes = ["THOUGHT: More.\nACTION: web_search\nINPUT: query %d" % i for i in range(50)]
        client = _ScriptedClient(never_finishes)
        run = ResearchAgent(client, _registry(), max_steps=3).research("Question?")

        assert run.step_count == 3
        assert run.stopped_reason == "step_budget_exhausted"

    def test_report_is_still_written_when_budget_runs_out(self):
        client = _ScriptedClient(["THOUGHT: More.\nACTION: web_search\nINPUT: q%d" % i for i in range(10)])
        run = ResearchAgent(client, _registry(), max_steps=2).research("Question?")
        assert run.report  # work isn't discarded just because the budget ended
        assert len(run.citations) > 0

    def test_remaining_step_count_is_told_to_the_agent(self):
        client = _ScriptedClient(["THOUGHT: x\nACTION: finish\nINPUT: done"])
        ResearchAgent(client, _registry(), max_steps=5).research("Question?")
        assert "Steps remaining: 5" in client.calls[0][1]


class TestRepeatedActionGuard:
    def test_identical_action_is_blocked_not_re_run(self):
        client = _ScriptedClient([
            "THOUGHT: Search.\nACTION: web_search\nINPUT: same query",
            "THOUGHT: Search again.\nACTION: web_search\nINPUT: same query",
            "THOUGHT: Fine.\nACTION: finish\nINPUT: done",
        ])
        run = ResearchAgent(client, _registry(), max_steps=6).research("Question?")

        repeat_step = run.steps[1]
        assert repeat_step.succeeded is False
        assert "already ran" in repeat_step.observation

    def test_repeat_check_is_case_insensitive(self):
        client = _ScriptedClient([
            "THOUGHT: Search.\nACTION: web_search\nINPUT: Quantum Computing",
            "THOUGHT: Again.\nACTION: web_search\nINPUT: quantum computing",
            "THOUGHT: Fine.\nACTION: finish\nINPUT: done",
        ])
        run = ResearchAgent(client, _registry(), max_steps=6).research("Question?")
        assert "already ran" in run.steps[1].observation

    def test_different_query_is_allowed(self):
        client = _ScriptedClient([
            "THOUGHT: Search.\nACTION: web_search\nINPUT: query one",
            "THOUGHT: Different.\nACTION: web_search\nINPUT: query two",
            "THOUGHT: Fine.\nACTION: finish\nINPUT: done",
        ])
        run = ResearchAgent(client, _registry(), max_steps=6).research("Question?")
        assert run.steps[1].succeeded is True


class TestFailureResilience:
    def test_tool_failure_becomes_an_observation_not_a_crash(self):
        client = _ScriptedClient([
            "THOUGHT: Search.\nACTION: web_search\nINPUT: query",
            "THOUGHT: Give up on search.\nACTION: finish\nINPUT: done",
        ])
        run = ResearchAgent(client, _registry(search_boom=True), max_steps=4).research("Question?")

        assert run.steps[0].succeeded is False
        assert "Tool failed" in run.steps[0].observation
        assert run.stopped_reason == "finished"  # the run still completed

    def test_unknown_tool_is_reported_back_to_the_agent(self):
        client = _ScriptedClient([
            "THOUGHT: Try something.\nACTION: nonexistent_tool\nINPUT: whatever",
            "THOUGHT: Fine.\nACTION: finish\nINPUT: done",
        ])
        run = ResearchAgent(client, _registry(), max_steps=4).research("Question?")
        assert run.steps[0].succeeded is False
        assert "Unknown tool" in run.steps[0].observation

    def test_malformed_reply_is_recoverable(self):
        client = _ScriptedClient([
            "I'm just going to chat instead of following the format.",
            "THOUGHT: OK, proper format.\nACTION: web_search\nINPUT: query",
            "THOUGHT: Done.\nACTION: finish\nINPUT: done",
        ])
        run = ResearchAgent(client, _registry(), max_steps=5).research("Question?")

        assert run.steps[0].succeeded is False
        assert "format" in run.steps[0].observation.lower()
        assert run.steps[1].action == "web_search"  # recovered and continued
        assert run.stopped_reason == "finished"

    def test_no_sources_produces_an_honest_report_not_a_fabricated_one(self):
        # Every tool call fails, so nothing was gathered. The report must
        # say so rather than inventing an answer.
        client = _ScriptedClient([
            "THOUGHT: Search.\nACTION: web_search\nINPUT: query",
            "THOUGHT: Give up.\nACTION: finish\nINPUT: done",
        ])
        run = ResearchAgent(client, _registry(search_boom=True), max_steps=4).research("Question?")

        assert len(run.citations) == 0
        assert "No sources could be gathered" in run.report

    def test_synthesis_is_skipped_entirely_when_no_sources(self):
        client = _ScriptedClient(["THOUGHT: Give up.\nACTION: finish\nINPUT: done"])
        ResearchAgent(client, _registry(search_boom=True), max_steps=2).research("Question?")
        # No call should have used the synthesis system prompt.
        assert not any("research analyst" in sp.lower() for sp, _ in client.calls)


class TestSynthesisPrompt:
    def test_synthesis_receives_numbered_source_block(self):
        client = _ScriptedClient([
            "THOUGHT: Search.\nACTION: web_search\nINPUT: query",
            "THOUGHT: Done.\nACTION: finish\nINPUT: done",
        ])
        ResearchAgent(client, _registry(), max_steps=4).research("Question?")

        synthesis_call = next(c for c in client.calls if "research analyst" in c[0].lower())
        assert "[1] Page A (https://a.com)" in synthesis_call[1]

    def test_synthesis_prompt_forbids_inventing_citation_numbers(self):
        client = _ScriptedClient([
            "THOUGHT: Search.\nACTION: web_search\nINPUT: query",
            "THOUGHT: Done.\nACTION: finish\nINPUT: done",
        ])
        ResearchAgent(client, _registry(), max_steps=4).research("Question?")
        synthesis_call = next(c for c in client.calls if "research analyst" in c[0].lower())
        assert "never invent a number" in synthesis_call[0].lower()

    def test_synthesis_failure_is_wrapped_in_agenterror(self):
        from src.claude_client import ClaudeClientError

        class _FailsOnSynthesis(_ScriptedClient):
            def complete(self, system_prompt, user_message):
                if "research analyst" in system_prompt.lower():
                    raise ClaudeClientError("API down during synthesis")
                return super().complete(system_prompt, user_message)

        client = _FailsOnSynthesis([
            "THOUGHT: Search.\nACTION: web_search\nINPUT: query",
            "THOUGHT: Done.\nACTION: finish\nINPUT: done",
        ])
        with pytest.raises(AgentError, match="Could not write the final report"):
            ResearchAgent(client, _registry(), max_steps=4).research("Question?")
