"""
agent.py
========

The autonomous research loop: a ReAct-style cycle of
*think → act → observe*, repeated until the agent decides it has enough
material or hits its step budget, then a final synthesis pass that writes
a cited report.

Three things here are deliberately enforced in Python rather than left to
the model's good behaviour, because an agent that polices itself only via
its prompt will eventually not:

1. **A hard step budget.** The loop cannot exceed ``max_steps`` tool
   calls no matter what the model asks for. This is the difference
   between an agent and an unbounded bill.
2. **A repeated-action guard.** If the model asks for the exact same
   tool + query it already ran, the loop feeds back a nudge instead of
   burning a step re-fetching identical data — a very common agent
   failure mode.
3. **Tool failures don't kill the run.** A failed tool call becomes an
   observation the agent can react to and route around, not an exception
   that discards all work done so far.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.citations import CitationRegistry
from src.claude_client import ClaudeClient, ClaudeClientError
from src.tools import ToolError, ToolRegistry, ToolResult

_AGENT_SYSTEM_PROMPT = """\
You are an autonomous research agent. You answer a research question by \
searching the web, reading pages, and then synthesizing what you found.

Available tools:
{tool_descriptions}

On each turn, reply in EXACTLY this format and nothing else:

THOUGHT: <one sentence on what you still need and why>
ACTION: <tool_name>
INPUT: <the input for that tool>

When you have gathered enough material to answer the question well, \
reply instead with EXACTLY:

THOUGHT: <one sentence on why you have enough>
ACTION: finish
INPUT: done

Rules:
- Use web_search to find pages; use read_page (with a URL from a prior \
search) to read one in depth.
- Do not repeat a search or page read you have already performed.
- Prefer reading 1-2 promising pages over running many shallow searches.
- You have a limited number of steps. Finish as soon as you can answer well.
"""

_SYNTHESIS_SYSTEM_PROMPT = """\
You are a research analyst writing a final report from gathered material.

Write a clear, well-structured markdown report answering the research \
question. Requirements:
- Open with a 2-3 sentence executive summary.
- Then 2-5 short thematic sections with ## headings.
- Cite sources inline using their bracketed numbers, e.g. [1], [2]. Use \
ONLY the numbers listed in the Sources block — never invent a number.
- Base every factual claim on the gathered material. If the material \
does not settle something, say so plainly rather than filling the gap.
- Do not include a references list — it is appended automatically.
- Do not repeat the raw gathered text; synthesize it.
"""


class AgentError(RuntimeError):
    """Raised when the agent cannot complete a research run."""


@dataclass(frozen=True, slots=True)
class AgentStep:
    """One recorded think→act→observe cycle, for the run trace."""

    number: int
    thought: str
    action: str
    action_input: str
    observation: str
    succeeded: bool = True
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass(frozen=True, slots=True)
class ParsedAction:
    """The agent's parsed intent for one turn."""

    thought: str
    action: str
    action_input: str

    @property
    def is_finish(self) -> bool:
        return self.action.strip().lower() == "finish"


@dataclass
class ResearchRun:
    """The full result of a research run: the report, trace, and sources."""

    question: str
    report: str
    steps: list[AgentStep]
    citations: CitationRegistry
    stopped_reason: str  # "finished" | "step_budget_exhausted"

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_markdown(self) -> str:
        """Render the report plus its auto-appended reference list."""
        return f"# {self.question}\n\n{self.report}\n\n## Sources\n\n{self.citations.to_markdown()}"


def parse_action(reply: str) -> ParsedAction:
    """Parse the agent's THOUGHT/ACTION/INPUT reply.

    Args:
        reply: The model's raw reply text.

    Returns:
        The parsed action.

    Raises:
        AgentError: If no ACTION line can be found — without one there is
            nothing the loop can execute.
    """
    def _field(name: str) -> str:
        match = re.search(rf"^{name}:\s*(.+?)\s*$", reply, re.MULTILINE | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    action = _field("ACTION")
    if not action:
        raise AgentError(f"Agent reply contained no ACTION line:\n{reply[:300]}")

    return ParsedAction(
        thought=_field("THOUGHT") or "(no thought given)",
        action=action,
        action_input=_field("INPUT"),
    )


class ResearchAgent:
    """Runs an autonomous, tool-using research loop over a question.

    Args:
        client: The Claude client driving the agent's reasoning.
        tools: The tools the agent may call.
        max_steps: Hard cap on tool-calling steps. Enforced in Python —
            the model cannot exceed it by asking to.

    Raises:
        ValueError: If ``max_steps`` is not positive.
    """

    def __init__(self, client: ClaudeClient, tools: ToolRegistry, max_steps: int = 6) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.client = client
        self.tools = tools
        self.max_steps = max_steps

    def research(self, question: str) -> ResearchRun:
        """Research a question and return a cited report plus the full trace.

        Args:
            question: The research question.

        Returns:
            The completed run.

        Raises:
            ValueError: If ``question`` is empty.
            ClaudeClientError: If the LLM is unreachable.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        question = question.strip()

        registry = CitationRegistry()
        steps: list[AgentStep] = []
        transcript: list[str] = [f"Research question: {question}"]
        performed: set[tuple[str, str]] = set()
        stopped_reason = "step_budget_exhausted"

        system_prompt = _AGENT_SYSTEM_PROMPT.format(tool_descriptions=self.tools.describe())

        for step_number in range(1, self.max_steps + 1):
            remaining = self.max_steps - step_number + 1
            user_message = (
                "\n\n".join(transcript)
                + f"\n\nSteps remaining: {remaining}. What is your next action?"
            )
            reply = self.client.complete(system_prompt, user_message)

            try:
                parsed = parse_action(reply)
            except AgentError:
                # A malformed reply is recoverable: tell the agent the
                # required format and let it retry on the next step,
                # rather than aborting a run that may already have
                # gathered useful material.
                steps.append(
                    AgentStep(
                        number=step_number,
                        thought="(unparseable reply)",
                        action="(none)",
                        action_input="",
                        observation="Reply did not follow the THOUGHT/ACTION/INPUT format.",
                        succeeded=False,
                    )
                )
                transcript.append(
                    "OBSERVATION: Your reply was not in the required format. "
                    "Reply with exactly THOUGHT:, ACTION:, and INPUT: lines."
                )
                continue

            if parsed.is_finish:
                steps.append(
                    AgentStep(
                        number=step_number,
                        thought=parsed.thought,
                        action="finish",
                        action_input="",
                        observation="Agent decided it had gathered enough material.",
                    )
                )
                stopped_reason = "finished"
                break

            signature = (parsed.action, parsed.action_input.strip().lower())
            if signature in performed:
                observation = (
                    f"You already ran {parsed.action} with that exact input. "
                    "Try a different query, read a different page, or finish."
                )
                steps.append(
                    AgentStep(
                        number=step_number,
                        thought=parsed.thought,
                        action=parsed.action,
                        action_input=parsed.action_input,
                        observation=observation,
                        succeeded=False,
                    )
                )
                transcript.append(
                    f"THOUGHT: {parsed.thought}\nACTION: {parsed.action}\n"
                    f"INPUT: {parsed.action_input}\nOBSERVATION: {observation}"
                )
                continue

            performed.add(signature)

            try:
                result: ToolResult = self.tools.run(parsed.action, parsed.action_input)
                registry.add_all(result.sources)
                observation = result.output
                succeeded = True
            except ToolError as exc:
                # A tool failure is an observation, not a crash — the
                # agent can route around a dead link or bad query.
                observation = f"Tool failed: {exc}"
                succeeded = False

            steps.append(
                AgentStep(
                    number=step_number,
                    thought=parsed.thought,
                    action=parsed.action,
                    action_input=parsed.action_input,
                    observation=observation,
                    succeeded=succeeded,
                )
            )
            transcript.append(
                f"THOUGHT: {parsed.thought}\nACTION: {parsed.action}\n"
                f"INPUT: {parsed.action_input}\nOBSERVATION: {observation}"
            )

        report = self._synthesize(question, transcript, registry)
        return ResearchRun(
            question=question,
            report=report,
            steps=steps,
            citations=registry,
            stopped_reason=stopped_reason,
        )

    def _synthesize(
        self, question: str, transcript: list[str], registry: CitationRegistry
    ) -> str:
        """Write the final report from everything gathered."""
        if len(registry) == 0:
            return (
                "No sources could be gathered for this question, so no "
                "evidence-based report can be written. This usually means the "
                "search tool failed or returned nothing — check the run trace "
                "for the specific tool errors."
            )

        user_message = (
            f"Research question: {question}\n\n"
            f"Sources (cite by these numbers only):\n{registry.context_block()}\n\n"
            f"Gathered material:\n" + "\n\n".join(transcript)
        )
        try:
            return self.client.complete(_SYNTHESIS_SYSTEM_PROMPT, user_message).strip()
        except ClaudeClientError as exc:
            raise AgentError(f"Could not write the final report: {exc}") from exc
