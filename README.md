# 🔎 Autonomous AI Research Agent

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python\&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-76%20Passing-brightgreen)
![Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit\&logoColor=white)
![Agent](https://img.shields.io/badge/Pattern-ReAct-9333EA)
![LLM](https://img.shields.io/badge/LLM-Claude%20API-8A63D2)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An autonomous AI research agent that receives a research question, determines what information it needs, searches for relevant sources, reads selected pages, handles failures and repeated actions, and produces a cited research report based on the sources it actually retrieved.

The system is built around a **ReAct (Reason + Act)** loop with control mechanisms implemented directly in Python for predictable execution, testing, and resource management.

---

## Overview

Unlike a simple single-prompt LLM workflow, this project implements an iterative research process:

```text
Research Question
       ↓
   Think → Act → Observe
       ↓
   Think → Act → Observe
       ↓
      ...
       ↓
   Synthesize
       ↓
 Cited Research Report
```

During a run, the agent can:

* Generate its own search queries
* Select pages to read
* Process tool results as observations
* Recover from tool failures
* Detect repeated actions
* Operate within a fixed step budget
* Decide when enough information has been gathered
* Produce citations based on retrieved sources

---

## What Makes This an Agent?

The core workflow follows the ReAct pattern:

1. **Think** — determine the next research action.
2. **Act** — execute a registered tool.
3. **Observe** — receive and process the tool result.
4. **Repeat** — continue until the agent finishes or reaches its limits.
5. **Synthesize** — generate the final cited report from gathered sources.

```text
Question
   │
   ▼
┌─────────────┐
│    Think    │
└──────┬──────┘
       ▼
┌─────────────┐
│     Act     │
└──────┬──────┘
       ▼
┌─────────────┐
│   Observe   │
└──────┬──────┘
       │
       ├──────► Continue
       │
       ▼
┌─────────────┐
│  Synthesize │
└──────┬──────┘
       ▼
  Cited Report
```

The sequence of searches and page reads is not hardcoded. The model determines the next action based on the current research state and observations.

---

## Safety & Control Mechanisms

Important agent controls are enforced in Python rather than relying exclusively on model instructions.

### 1. Hard Step Budget

Every research run has a maximum number of steps.

```python
max_steps
```

The model cannot exceed this limit, even if it continues requesting additional actions.

The remaining step count is also provided to the model so it can prioritize its remaining research actions.

---

### 2. Repeated-Action Guard

The agent tracks previously executed tool actions.

If the model requests the same tool with the same input again, the system detects the duplicate and returns an observation instead of unnecessarily repeating the operation.

Example:

```text
Step 1:
web_search("RAG approaches 2026")

Step 3:
web_search("RAG approaches 2026")

→ Repeated action detected.
```

This prevents the agent from wasting execution steps repeatedly fetching identical information.

---

### 3. Tool Failure Recovery

Tool failures are treated as observations instead of terminating the entire research process.

For example:

```text
Tool failed:
Page could not be retrieved.
```

The agent can then choose another action based on that observation.

This allows a research run to continue even when an individual search or page retrieval fails.

---

### 4. Malformed Response Recovery

The agent expects a structured response containing:

```text
THOUGHT
ACTION
INPUT
```

If the model produces an invalid response format, the system provides corrective feedback and continues the loop rather than immediately terminating the run.

---

### 5. No Sources → No Report

The system does not synthesize a research report when no usable sources were successfully gathered.

If all tool calls fail, the run reports that no sources were collected instead of generating a research answer from unsupported information.

---

## Architecture

```text
                         ResearchService
                              │
                              ▼
                       ┌───────────────┐
                       │ ResearchAgent │
                       └───────┬───────┘
                               │
                    ReAct Research Loop
                 Think → Act → Observe
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
       ┌─────────────┐                  ┌───────────────┐
       │ ToolRegistry│                  │ Claude Client │
       └──────┬──────┘                  └───────────────┘
              │
       ┌──────┴──────────────┐
       │                     │
       ▼                     ▼
┌───────────────┐    ┌────────────────┐
│ Tavily Search │    │ Page Reader    │
│     Tool      │    │     Tool       │
└───────────────┘    └────────────────┘
       │                     │
       ▼                     ▼
   Search API             HTML → Text

              ┌───────────────────────┐
              │  Citation Registry    │
              │                       │
              │ URL deduplication     │
              │ Stable numbering      │
              └───────────────────────┘

                    User Interfaces
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Streamlit UI             CLI
          app.py                main.py
```

### Main Components

| Component          | Responsibility                                     |
| ------------------ | -------------------------------------------------- |
| `ResearchAgent`    | Runs the ReAct loop and execution controls         |
| `ToolRegistry`     | Registers and dispatches tools                     |
| `TavilySearchTool` | Performs web searches                              |
| `PageReaderTool`   | Retrieves and converts pages to text               |
| `CitationRegistry` | Deduplicates URLs and maintains citation numbering |
| `Claude Client`    | Handles Anthropic API communication                |
| `ResearchService`  | Provides the main research service interface       |
| `app.py`           | Streamlit interface with agent execution timeline  |
| `main.py`          | CLI interface and offline demo                     |

---

## Testable Tool Architecture

The search and page-reading tools receive their network backend through injected callables rather than directly coupling the agent to a specific HTTP implementation.

This provides two important benefits:

* The agent can be tested deterministically without live network requests.
* The search provider can be replaced without changing the core agent logic.

The architecture can therefore support a different search backend while keeping the research loop independent from the provider implementation.

---

## Environment Constraint

The development environment used for this project did not provide access to the required live web-search API host.

Therefore:

* The complete agent control flow was tested end-to-end.
* The ReAct loop was tested with deterministic fake tools and a scripted model.
* Step-budget enforcement was tested.
* Repeated-action detection was tested.
* Tool-failure handling was tested.
* Citation collection was tested.
* Report generation conditions were tested.
* Tavily and page-reader HTTP behavior was tested using mocked transports.
* A working offline `demo` command is included.

Live web research was not executed in this environment.

For live research, configure the required API keys on a machine with outbound network access.

---

## Project Structure

```text
day47-research-agent/
│
├── src/
│   ├── agent.py
│   ├── tools.py
│   ├── citations.py
│   ├── claude_client.py
│   └── research_service.py
│
├── tests/
│   ├── test_agent.py
│   ├── test_tools.py
│   ├── test_citations.py
│   └── test_research_service.py
│
├── app.py
├── main.py
├── requirements.txt
├── pytest.ini
├── .gitignore
├── GUIDE.txt
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/autonomous-research-agent.git
cd autonomous-research-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API Keys

For live research, configure:

```bash
ANTHROPIC_API_KEY="your-anthropic-api-key"
TAVILY_API_KEY="your-tavily-api-key"
```

---

## Usage

### Offline Demo

The project includes an offline demo that does not require API keys or network access.

```bash
python main.py demo
```

This executes the actual agent control flow using scripted model responses and fake tools.

---

### Live Research

Run a research question with the live tools:

```bash
python main.py research "What are the main approaches to RAG in 2026?"
```

Specify a custom step limit:

```bash
python main.py research "What are the main approaches to RAG in 2026?" --max-steps 8
```

Save the report:

```bash
python main.py research "What are the main approaches to RAG in 2026?" --max-steps 8 --output output/report.md
```

---

## Streamlit Interface

Launch the Streamlit application:

```bash
streamlit run app.py
```

The interface provides a visible research timeline showing the agent's execution process, including:

* Search actions
* Page-reading actions
* Tool failures
* Agent steps
* Research progress
* Final cited report

The execution trace makes the agent's behavior observable instead of hiding the entire process behind a loading indicator.

---

## Demo Output

The following is the deterministic output from:

```bash
python main.py demo
```

```text
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

The third step demonstrates the repeated-action guard preventing an identical search from being executed again.

---

## Testing

Run the test suite:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

Current project results:

```text
76 tests passing
100% statement coverage on src/
```

### Coverage

```text
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

The tests cover important agent-control scenarios, including:

* Step-budget exhaustion
* Repeated actions
* Search failures
* Unknown tools
* Malformed model responses
* Citation handling
* Successful research flows
* Runs where no usable sources are collected

---

## Deployment

### Streamlit Community Cloud

The Streamlit application can be deployed using Streamlit Community Cloud.

Deployment requirements:

1. Push the project to GitHub.
2. Select `app.py` as the application entry point.
3. Configure:

   * `ANTHROPIC_API_KEY`
   * `TAVILY_API_KEY`
4. Deploy the application.

For constrained environments, the default step budget can be reduced because each research step may involve an LLM call and network operation.

### Other Deployment Options

The same Streamlit application can also be deployed using:

* Render
* Hugging Face Spaces

---

## Version 2.0 Roadmap

The following items are planned future work and are **not currently implemented**.

* Parallel tool execution
* Fan-out searches for independent research queries
* Additional research capabilities

---
---

## License

This project is licensed under the MIT License.
