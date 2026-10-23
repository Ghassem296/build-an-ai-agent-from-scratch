# Build an AI Agent from Scratch

[![Tests](https://github.com/Ghassem296/build-an-ai-agent-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/Ghassem296/build-an-ai-agent-from-scratch/actions/workflows/tests.yml)

An educational AI agent built on **[LiteLLM](https://docs.litellm.ai/)** and the
**[Model Context Protocol](https://modelcontextprotocol.io/)** (MCP).

The agent talks to an LLM, discovers the tools that an MCP server exposes, calls
them when they help, feeds the results back to the model, and returns a grounded
final answer. Everything is plain Python — no agent framework is involved, so the
control loop stays readable.

Built while following the book *Build an AI Agent from Scratch* (Manning, by
Jungjun Hur & Younghee Song).

---

## Features

- **Tool calling over MCP** — tools live in a separate stdio MCP server and are
  discovered at runtime; nothing is hard-coded in the agent.
- **Tool allowlist** — only tools advertised by the MCP server may be executed.
  Anything else is rejected with an error `ToolResult` instead of being run.
- **Argument validation** — every tool call is validated against the tool's JSON
  schema before it reaches the server.
- **Turn-aware history truncation** — long conversations are trimmed at turn
  boundaries, so an `assistant` tool call is never separated from its
  `tool` result. The most recent turn is always preserved.
- **Bounded tool output** — tool results are capped at 10 000 characters so one
  noisy search cannot blow up the context window.
- **Truncation is surfaced** &mdash; when the model stops because it hit its
  output limit (`finish_reason == "length"`), the answer is annotated instead of
  being passed off as complete.
- **Character budget for history** &mdash; `AGENT_MAX_CONTEXT_CHARS` drops the
  oldest turns (never the system prompt or the newest turn) before a request is
  sent.
- **Timeouts and retries** — each tool call is bounded by `TOOL_TIMEOUT`; web
  search retries 3 times with exponential backoff; LLM calls use LiteLLM's
  `num_retries`.
- **Errors never leak internals** — raw exceptions are logged, while the model
  (and the user) only sees a safe, generic message.
- **Prompt-injection aware system prompt** — tool output is treated as untrusted
  data.
- **Minimal MCP environment** — the subprocess receives an explicit allowlist of
  environment variables, so API keys and tokens never reach it.
- **83 tests, no network required** — the LLM and the MCP session are mocked.
- **CI on Linux and Windows** — Python 3.11, 3.12 and 3.13.

---

## Requirements

- Python **3.11+**
- [`uv`](https://docs.astral.sh/uv/) (recommended) or plain `pip`
- An LLM API key for any provider LiteLLM supports

---

## Installation

```bash
git clone https://github.com/Ghassem296/build-an-ai-agent-from-scratch.git
cd build-an-ai-agent-from-scratch

# with uv (recommended; creates .venv and installs the exact locked versions)
uv sync

# or with pip — install the modules this project imports directly
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install "litellm>=1.83.0" "jsonschema>=4.26.0" "python-dotenv>=1.2.4" \
            "mcp[cli]>=1.0.0,<2.0.0" "ddgs>=9.16.0"
```

Next, create your local environment file:

```bash
cp .env.example .env             # Windows: copy .env.example .env
```

and put your key in it (`LLM_API_KEY=...`).

---

## Configuration

All settings are read from the environment (`.env` is loaded automatically).

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | `openai/gemini-2.5-flash` | Model string in LiteLLM format. The `openai/` prefix means "OpenAI-compatible endpoint". |
| `LLM_API_KEY` | — | **Required.** Provider key. |
| `LLM_API_BASE` | — | Optional OpenAI-compatible base URL (proxy, gateway, self-hosted). Leave empty for official APIs. |
| `XAI_1_API_KEY` | — | Deprecated alias for `LLM_API_KEY`. |
| `LLM_TEMPERATURE` | `0` | 0–2. |
| `LLM_TIMEOUT` | `60` | Seconds per LLM request. |
| `AGENT_MAX_STEPS` | `10` | Think-act cycles before the agent is forced to answer. |
| `AGENT_MAX_CONTEXT_CHARS` | `120000` | Characters of history kept; older turns are dropped beyond it. |
| `TOOL_TIMEOUT` | `30` | Seconds per tool call. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |

> **Note on `LLM_API_BASE`.** The example file points at a third-party
> OpenAI-compatible gateway (`https://1xai.ir/v1`). Remove it if you call a
> provider's official endpoint directly.

---

## Usage

Single prompt:

```bash
uv run python -m app.main "What is the Model Context Protocol?"
```

Interactive mode (keeps one conversation across turns):

```bash
uv run python -m app.main
# then type your questions; 'exit' or 'quit' ends the session
```

Chapter-by-chapter demo (same agent, fixed prompt, prints the discovered tools):

```bash
uv run python mcp_agent.py
```

Quick connectivity checks live in `examples/`: `api_connection_check.py`,
`web_search_check.py`, `mcp_client.py`, `tool_schema_example.py` and
`legacy_agent_demo.py` (the pre-refactor single-file loop, kept for comparison).

---

## How it works

```
prompt
  │
  ▼
Agent.run()
  │  seed system prompt ─────────────┐
  │                                  │ ExecutionContext
  ├─► step(): truncate → LLM ────────┤ (event history:
  │        │                         │  system / user /
  │        ├── no tool calls ──► final answer
  │        │                         │  assistant / tool)
  │        └── tool calls ──► act(): validate → MCP call_tool
  │                                └── ToolResult back into context
  └─ max_steps reached ──► forced final answer (tools disabled)
```

| Module | Responsibility |
|---|---|
| `app/main.py` | CLI, logging setup, single-prompt and interactive modes. |
| `app/agent.py` | The think-act loop, tool allowlist, schema validation, tool execution. |
| `app/llm_client.py` | LiteLLM wrapper: builds OpenAI-style messages from events, parses tool calls and usage. |
| `app/execution_context.py` | Append-only event log plus turn-aware truncation. |
| `app/config.py` | Environment parsing and validation (`get_settings()`). |
| `app/mcp_runtime.py` | Minimal subprocess environment and the initialized MCP session. |
| `tools/mcp_server.py` | FastMCP server exposing the `search_web` tool (DDGS). |

---

## Testing

```bash
uv run python -m unittest discover -s tests -v
# or, if you prefer pytest:
uv run pytest
```

The suite is fully offline: the LLM client and the MCP session are replaced with
mocks, so tests never spend tokens or hit the network.

---

## Project layout

```
app/                agent, LLM client, context, config, MCP runtime, CLI
tools/mcp_server.py the MCP server (web search tool)
tests/              unittest suites for every module
examples/           small hand-run scripts used while writing the book
uv.lock             pinned dependency graph (commit it)
```

---

## Limitations

- The web-search tool scrapes DuckDuckGo through `ddgs`; it can be rate-limited.
- Agent state is in memory only — there is no persistence, resumability or
  streaming.
- The history budget is measured in characters, not tokens, so it is an
  approximation; a tokenizer-aware limit would be more precise.
- Tool output is treated as untrusted text, but prompt injection is mitigated
  rather than prevented.

---

## License

[MIT](LICENSE)
