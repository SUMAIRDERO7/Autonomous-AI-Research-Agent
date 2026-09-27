# 🔎 Autonomous AI Research Agent

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-76%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Agent](https://img.shields.io/badge/pattern-ReAct-9333EA)
![LLM](https://img.shields.io/badge/LLM-Claude%20API-8A63D2)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Give it a research question. It decides what to search, reads the pages
worth reading, notices when it's going in circles, stops when it has
enough — and writes a report where every claim is cited back to a real
source it actually fetched.

---

## Table of Contents

- [Why This Project](#why-this-project)
- [What Makes This an *Agent*](#what-makes-this-an-agent)
- [Three Guarantees Enforced in Python, Not in the Prompt](#three-guarantees-enforced-in-python-not-in-the-prompt)
- [Architecture](#architecture)
- [Environment Constraint — Stated Honestly](#environment-constraint--stated-honestly)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Real Demo Output](#real-demo-output)
- [Testing](#testing)
- [Deployment](#deployment)
- [Version 2.0 Roadmap](#version-20-roadmap)
- [GitHub Topics](#github-topics)
- [Resume Bullet](#resume-bullet)
- [Portfolio Description](#portfolio-description)
- [LinkedIn Post Draft](#linkedin-post-draft)

---

## Why This Project

"Agent" is the most over-claimed word in AI right now. Most things
labelled agents are a single LLM call wearing a costume. A real agent
has to do four things a chatbot doesn't: **choose** its own actions,
**observe** the results, **adapt** when something fails, and **know when
to stop**.

This project implements all four in a ReAct (Reason + Act) loop, and —
more importantly — implements the *safety rails* around them in Python
rather than politely asking the model to behave. An agent that polices
itself only through its prompt will eventually not.

## What Makes This an *Agent*

Each turn, the model produces a structured `THOUGHT / ACTION / INPUT`
reply. The loop parses it, executes the chosen tool, feeds the result
back as an observation, and repeats:

```
Question → [THINK → ACT → OBSERVE] × N → SYNTHESIZE → cited report
```

The agent picks its own search queries, decides which URL is worth
reading in full, routes around dead tools, and chooses when it has
enough material. Nothing in that sequence is hardcoded.

## Three Guarantees Enforced in Python, Not in the Prompt

**1. A hard step budget.**
`max_steps` is a loop bound. The model can ask for a hundred more
searches; it gets zero. This is the difference between an agent and an
unbounded API bill. The remaining step count is also fed back to the
model each turn so it can prioritize — but the cap holds regardless of
whether it listens.

**2. A repeated-action guard.**
If the model asks for a tool + input it has already run, the loop
returns a nudge instead of burning a step re-fetching identical data.
Looping on the same query is one of the most common real agent failure
modes, and it's a control-flow problem, so it gets a control-flow fix.

**3. Tool failures are observations, not crashes.**
A dead link or a failed search becomes text the agent can react to and
route around. A run that has already gathered five good sources is not
thrown away because the sixth fetch 404'd.

Two smaller ones worth noting:

- **Malformed replies are recoverable.** If the model ignores the output
  format, the loop tells it the required format and continues, rather
  than aborting a run mid-flight.
- **No sources → no report.** If every tool call failed, the agent says
  so plainly instead of inventing an answer from the model's training
  data. That's the whole point of a *research* agent — the synthesis
  call is skipped entirely rather than being allowed to hallucinate.

## Architecture

```
                         ResearchService (facade)
                                   |
                            ResearchAgent
              ReAct loop: think -> act -> observe -> repeat
              + step budget + repeat guard + failure handling
                    |                            |
        +-----------+----------+          CitationRegistry
        |                      |          dedupes by URL,
   ToolRegistry          claude_client     stable numbering
   dispatch by name      Anthropic API     across the run
        |
   +----+--------------------+
   |                         |
TavilySearchTool       PageReaderTool
injected search_fn     injected fetch_fn
(swappable provider)   (HTML -> text)

   UI layer (either one, same service underneath):
   +------------+   +------------+
   |  app.py     |   |  main.py    |
   |  Streamlit  |   |  Terminal   |
   |  trace      |   |  CLI +      |
   |  timeline   |   |  demo mode  |
   +------------+   +------------+
```

Both tools take their network backend as an **injected callable** rather
than importing an HTTP client at module level. That's what makes the
entire agent loop testable deterministically, and what makes the search
provider swappable (Tavily, Brave, SerpAPI, a local index) without
touching any agent logic.

## Environment Constraint — Stated Honestly

This sandbox's network allowlist does not include any web-search API
host, so **the agent could not be run against a live search provider
here.** What that means precisely:

- The full agent loop — tool selection, execution, observation,
  repeat-detection, budget enforcement, citation collection, synthesis —
  **was run end-to-end** and is verified by 76 tests plus a working
  `demo` command (output below).
- `TavilySearchTool` and `PageReaderTool` are wired for real use and
  their HTTP backends are unit-tested with mocked transports, but no
  live search was executed in this environment.
- Running `python main.py research "..."` with `ANTHROPIC_API_KEY` and
  `TAVILY_API_KEY` set, on a machine with outbound access, exercises the
  same code path against real sources.

This is the same kind of documented trade-off as Day 42's local embedder
and Day 43's missing PyTorch — a real constraint, named plainly, rather
than a claim that something was verified when it wasn't.

## Folder Structure

```
day47-research-agent/
├── src/
│   ├── agent.py             # the ReAct loop + budget/repeat/failure guards
│   ├── tools.py             # Tool protocol, search + page reader, registry
│   ├── citations.py         # URL-deduplicated, stable source numbering
│   ├── claude_client.py     # Anthropic API wrapper
│   └── research_service.py  # facade
├── tests/                   # 76 tests, 100% coverage
│   ├── test_agent.py
│   ├── test_tools.py
│   ├── test_citations.py
│   └── test_research_service.py
├── app.py                   # Streamlit agent-trace timeline UI
├── main.py                  # CLI: `research` (live) and `demo` (offline)
├── requirements.txt
├── pytest.ini
├── .gitignore
├── GUIDE.txt                # Roman Urdu walkthrough
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/autonomous-research-agent.git
cd autonomous-research-agent
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."   # required for live runs
export TAVILY_API_KEY="tvly-..."        # required for live web search
```

## Usage

**Offline demo — no keys, no network, runs the real agent loop:**
```bash
python main.py demo
```

**Live research:**
```bash
python main.py research "What are the main approaches to RAG in 2026?"
python main.py research "..." --max-steps 8 --output output/report.md
```

**Streamlit UI:**
```bash
streamlit run app.py
```
The agent's every step renders as a timeline — which search it ran, which
page it read, where a tool failed, why it stopped — with the cited report
alongside. Hiding that behind a spinner would make the agent unauditable,
which is exactly the fair criticism of most agent demos.

## Real Demo Output

Actual output from `python main.py demo` (scripted model + fake tools, so
the control flow is deterministic and reproducible):

```
✓ Step 1: web_search [RAG approaches 2026]
   thought: I need an overview first.
   result:  1. RAG Survey 2026 — https://example.com/rag-survey
            2. Hybrid Retrieval Methods — https://example.com/hybrid

✓ Step 2: read_page [https://example.com/rag-survey]
   thought: Read the survey in depth.
   result:  Retrieval-augmented generation combines a retriever with a generator...

✗ Step 3: web_search [RAG approaches 2026]
   thought: Try that same search again.
   result:  You already ran web_search with that exact input. Try a different
            query, read a different page, or finish.

✓ Step 4: finish
   thought: I have enough to answer.

STOPPED: finished  |  4 steps  |  2 sources
```

Step 3 is the repeat guard doing its job — the agent tried to re-run an
identical search and the loop refused to spend a step on it.

## Testing

```bash
pytest
pytest --cov=src --cov-report=term-missing
```

**76 tests, 100% statement coverage on `src/`.**

```
Name                      Stmts   Miss  Cover
------------------------------------------------------
src/agent.py                105      0   100%
src/citations.py             42      0   100%
src/claude_client.py         24      0   100%
src/research_service.py      23      0   100%
src/tools.py                102      0   100%
------------------------------------------------------
TOTAL                       296      0   100%
```

The agent tests are the interesting ones. A scripted fake client drives
the loop down every branch deterministically: budget exhaustion with a
model that never stops, an agent repeating itself, a search backend that
throws, an unknown tool name, a reply that ignores the output format,
and a run where every tool fails and the report must admit it gathered
nothing. None of that needs an API key or network access.

## Deployment

**Streamlit Community Cloud (recommended, free):**
1. Push to GitHub, point [share.streamlit.io](https://share.streamlit.io) at `app.py`.
2. Add both `ANTHROPIC_API_KEY` and `TAVILY_API_KEY` in Secrets.
3. Consider lowering the default step budget — each step is an LLM call
   plus a network fetch, and free-tier resources are modest.

**Render / Hugging Face Spaces:** standard Streamlit deployment with the
same two environment variables.

## Version 2.0 Roadmap

*(Documented as future work only — not implemented now, per project policy.)*

- Parallel tool execution — fan out several independent searches in one step instead of strictly serially
- A reflection step: let the agent critique its own draft report and run targeted follow-up searches to fill the gaps it identifies
- Source quality weighting — prefer primary sources, papers, and official docs over content-farm results
- Persistent research memory across sessions, so a follow-up question reuses what was already gathered
- Streaming the trace live into the UI as each step completes, rather than after the whole run
- A cost meter showing tokens and API spend per run, since step budget is really a proxy for cost

## GitHub Topics

`python` `ai-agent` `react-agent` `agentic-ai` `claude-api` `anthropic` `llm` `tool-use` `streamlit` `web-research` `pytest` `citations`

## Resume Bullet

> Built an autonomous ReAct research agent that selects its own tools, reads web sources, and writes cited reports — with a hard step budget, repeated-action detection, and tool-failure recovery all enforced in application code rather than prompt instructions; achieved 100% test coverage across 76 tests by driving the full loop with a scripted fake LLM and injected tool backends, requiring no network or API keys.

## Portfolio Description

**Autonomous AI Research Agent** implements a genuine ReAct loop — the
model chooses each action, observes the result, and decides when to stop
— wrapped in safety rails that live in Python rather than in a prompt: a
hard step budget the model cannot exceed, a repeated-action guard that
catches the classic agent-looping failure, and tool-failure handling that
turns a dead link into an observation instead of a crashed run. If every
source fetch fails, it says so rather than hallucinating an answer.
Built with injected tool backends so the entire loop is deterministically
testable without network access, and paired with a trace-timeline UI that
shows exactly what the agent did — because an agent you can't audit isn't
one you should trust.

## LinkedIn Post Draft

> Day 47/60 of my AI Portfolio Challenge: an autonomous research agent — and the boring parts are the point. 🔎
>
> The ReAct loop itself (think → act → observe → repeat) is maybe 40 lines. What actually makes it trustworthy is what surrounds it, and I put all of it in Python rather than in the prompt:
>
> → A hard step budget. The model can ask for 50 more searches; it gets zero. That's the difference between an agent and an unbounded bill.
> → A repeated-action guard. Agents love to re-run the same search forever. That's a control-flow bug, so it gets a control-flow fix.
> → Tool failures become observations, not crashes. A 404 on source #6 doesn't throw away the five good ones.
> → If every fetch fails, it reports that it found nothing — instead of quietly answering from training data. For a *research* agent, that one matters most.
>
> 76 tests, 100% coverage, and the whole loop is driven by a scripted fake LLM with injected tool backends — so every failure branch is tested with no API key and no network.
>
> #AgenticAI #AIEngineering #Python #LLM #ReAct
# Autonomous-AI-Research-Agent
