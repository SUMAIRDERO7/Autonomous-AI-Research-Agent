"""
app.py
======

Streamlit UI for the Autonomous AI Research Agent.

Layout is an **agent trace timeline** — the agent's think→act→observe
steps stream down the page as numbered timeline entries, with the final
report and sources appearing beneath once the run completes. This is a
new shape versus every prior project (Day 41 chat thread, Day 42 document
library, Day 43 video theater, Day 44 mode-tabbed playground, Day 45
audio lab, Day 46 step wizard).

The choice is deliberate: the single most important thing about an
autonomous agent is *what it actually did* — which searches it ran,
which pages it read, where a tool failed, why it stopped. Hiding that
behind a spinner and showing only the final report would make the agent
unauditable, which is exactly the criticism agentic systems deserve when
they're built that way.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.claude_client import ClaudeClient, ClaudeClientError
from src.research_service import ResearchService
from src.tools import PageReaderTool, TavilySearchTool

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --navy: #0A2540; --royal: #2563EB; --sky: #38BDF8;
        --success: #10B981; --warn: #F59E0B; --danger: #EF4444;
        --bg: #F8FAFC; --text-primary: #111827; --text-secondary: #4B5563;
    }
    .stApp { background: var(--bg); }
    .app-title { font-size: 1.6rem; font-weight: 800; color: var(--navy); margin-bottom: 0; }
    .app-subtitle { color: var(--text-secondary); font-size: 0.92rem; margin: 0.15rem 0 1rem; }

    .trace-step {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 3px solid var(--royal);
        border-radius: 0 10px 10px 0; padding: 0.7rem 0.9rem; margin-bottom: 0.6rem;
    }
    .trace-step.failed { border-left-color: var(--warn); }
    .trace-step.finish { border-left-color: var(--success); }
    .trace-num {
        display: inline-block; background: var(--royal); color: #fff; border-radius: 999px;
        width: 22px; height: 22px; text-align: center; line-height: 22px;
        font-size: 0.72rem; font-weight: 700; margin-right: 0.45rem;
    }
    .trace-step.failed .trace-num { background: var(--warn); }
    .trace-step.finish .trace-num { background: var(--success); }
    .trace-action {
        font-family: ui-monospace, monospace; font-size: 0.78rem;
        color: var(--navy); font-weight: 700;
    }
    .trace-thought { color: var(--text-secondary); font-size: 0.85rem; margin: 0.35rem 0 0.3rem; }
    .trace-obs {
        background: #F8FAFC; border-radius: 6px; padding: 0.45rem 0.6rem;
        font-size: 0.76rem; color: var(--text-primary); max-height: 130px;
        overflow-y: auto; white-space: pre-wrap; line-height: 1.45;
    }

    .stat-card {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
        padding: 0.8rem; text-align: center; margin-bottom: 0.5rem;
    }
    .stat-value { font-size: 1.7rem; font-weight: 800; color: var(--royal); line-height: 1; }
    .stat-label {
        font-size: 0.68rem; color: var(--text-secondary);
        text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.3rem;
    }
    .empty-state { text-align: center; padding: 2.5rem 1rem; color: var(--text-secondary); }
    .empty-state h4 { color: var(--navy); margin-bottom: 0.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="app-title">🔎 Autonomous Research Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Ask a question — the agent searches, reads, and writes a '
    "cited report. Every step it takes is shown below.</p>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ setup --

with st.sidebar:
    st.subheader("Agent settings")
    max_steps = st.slider("Step budget", 2, 12, 6, help="Hard cap on tool calls, enforced in code.")
    st.caption("The agent physically cannot exceed this, regardless of what the model asks for.")

    st.divider()
    has_tavily = bool(__import__("os").environ.get("TAVILY_API_KEY"))
    if has_tavily:
        st.success("Live web search enabled", icon="✅")
    else:
        st.warning(
            "No `TAVILY_API_KEY` set — live web search will fail. "
            "The agent will still run and report the tool errors honestly.",
            icon="⚠️",
        )


@st.cache_resource
def get_client() -> ClaudeClient | None:
    try:
        return ClaudeClient()
    except ClaudeClientError:
        return None


client = get_client()

if client is None:
    st.markdown(
        """
        <div class="empty-state">
            <h4>No API key configured</h4>
            <p>Set <code>ANTHROPIC_API_KEY</code> in your environment and restart the app.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

service = ResearchService(
    client, tools=[TavilySearchTool(), PageReaderTool()], max_steps=max_steps
)

# ------------------------------------------------------------------- run --

question = st.text_input(
    "Research question",
    placeholder="e.g. What are the main approaches to retrieval-augmented generation in 2026?",
)

if st.button("🚀 Run research", type="primary") and question.strip():
    with st.spinner("Agent is researching… (this takes a minute)"):
        try:
            st.session_state.run = service.research(question)
        except ClaudeClientError as exc:
            st.error(f"Couldn't reach Claude: {exc}", icon="🚨")
        except Exception as exc:  # agent-level failure, surfaced not swallowed
            st.error(f"Research run failed: {exc}", icon="🚨")

run = st.session_state.get("run")

if run is None:
    st.markdown(
        '<div class="empty-state"><h4>No run yet</h4>'
        "<p>Enter a research question above to start the agent.</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ------------------------------------------------------------ run summary --

c1, c2, c3 = st.columns(3)
c1.markdown(
    f'<div class="stat-card"><div class="stat-value">{run.step_count}</div>'
    f'<div class="stat-label">steps taken</div></div>',
    unsafe_allow_html=True,
)
c2.markdown(
    f'<div class="stat-card"><div class="stat-value">{len(run.citations)}</div>'
    f'<div class="stat-label">sources cited</div></div>',
    unsafe_allow_html=True,
)
stop_label = "finished on its own" if run.stopped_reason == "finished" else "hit step budget"
c3.markdown(
    f'<div class="stat-card"><div class="stat-value" style="font-size:1rem;padding-top:0.4rem">'
    f'{stop_label}</div><div class="stat-label">why it stopped</div></div>',
    unsafe_allow_html=True,
)

trace_col, report_col = st.columns([1, 1.3])

with trace_col:
    st.markdown("### Agent trace")
    for step in run.steps:
        css = "finish" if step.action == "finish" else ("" if step.succeeded else "failed")
        st.markdown(
            f'<div class="trace-step {css}">'
            f'<span class="trace-num">{step.number}</span>'
            f'<span class="trace-action">{step.action}'
            f'{f"({step.action_input[:60]})" if step.action_input else ""}</span>'
            f'<div class="trace-thought">{step.thought}</div>'
            f'<div class="trace-obs">{step.observation[:600]}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

with report_col:
    st.markdown("### Report")
    st.markdown(run.report)

    if len(run.citations) > 0:
        st.markdown("#### Sources")
        for citation in run.citations.citations:
            st.markdown(f"[{citation.number}] [{citation.title}]({citation.url})")

    st.download_button(
        "⬇️ Download report (markdown)",
        data=run.to_markdown(),
        file_name="research_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
