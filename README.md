Build an AI Agent Control Loop from Scratch
An educational agent on LiteLLM and the
Model Context Protocol (MCP) — a
hand-written think-act loop, with production-grade libraries for the rest.

The agent talks to an LLM, discovers the tools that an MCP server exposes, calls
them when they help, feeds the results back to the model, and returns a grounded
final answer. The think-act loop is hand-written in plain Python — no agent
framework sits between you and the control flow. The LLM and tool layers are
delegated to LiteLLM and the MCP SDK, which is exactly what a production agent
would do.

Note: This is an unofficial companion implementation inspired by the book
Build an AI Agent from Scratch (Manning, by Jungjun Hur & Younghee Song).
The official book repository is separate. This project is not affiliated with
the authors or the publisher.

Quick start
git clone https://github.com/Ghassem296/build-an-ai-agent-from-scratch.git
cd build-an-ai-agent-from-scratch

uv sync
cp .env.example .env # then set LLM_MODEL and LLM_API_KEY

uv run python -m app.main "What is the Model Context Protocol?"

For a local Ollama server, set LLM_MODEL=ollama_chat/<model> and
LLM_API_BASE=http://127.0.0.1:11434, plus a larger LLM_TIMEOUT.

What is hand-written vs. what comes from libraries
This project is from scratch only at the control-loop level. The table below
makes the boundary explicit, so you know where to look when learning.

Layer	Hand-written?	Library
Think-act loop (app/agent.py)	Yes	—
Turn-aware history truncation (app/execution_context.py)	Yes	—
Tool allowlist & schema validation	Yes	jsonschema
Event log / ExecutionContext	Yes	—
Error handling and safe messages	Yes	—
LLM request/response parsing (app/llm_client.py)	No — thin wrapper	litellm
MCP session & tool discovery (app/mcp_runtime.py)	No — thin wrapper	mcp[cli]
MCP server (tools/mcp_server.py)	No — built on FastMCP	mcp[cli]
Web search tool	No — wrapper	ddgs
Environment loading	No — wrapper	python-dotenv
Features
Tool calling over MCP — tools live in a separate stdio MCP server and are
discovered at runtime; nothing is hard-coded in the agent.

Tool allowlist — only tools advertised by the MCP server may be executed.
Anything else is rejected with an error ToolResult instead of being run.

Argument validation — every tool call is validated against the tool's JSON
schema before it reaches the server.

Turn-aware history truncation — long conversations are trimmed at turn
boundaries, so an assistant tool call is never separated from its
tool result. The most recent turn is always preserved.

Bounded tool output — tool results are capped at 10 000 characters so one
noisy search cannot blow up the context window.

Truncation is surfaced — when the model stops because it hit its output
limit (finish_reason == "length"), the answer is annotated instead of being
passed off as complete.

Character budget for history — AGENT_MAX_CONTEXT_CHARS drops the oldest
turns (never the system prompt or the newest turn) before a request is sent.

Timeouts and retries — each tool call is bounded by TOOL_TIMEOUT; web
search retries 3 times with exponential backoff; LLM calls use LiteLLM's
num_retries.

Errors never leak internals — raw exceptions are logged, while the model
(and the user) only sees a safe, generic message.

Prompt-injection aware system prompt — tool output is treated as untrusted
data.

Minimal MCP environment — the subprocess receives an explicit allowlist of
environment variables, so API keys and tokens never reach it.

Offline automated tests — the LLM and MCP session are mocked, so the suite
does not require API calls.

CI matrix — Python 3.11, 3.12 and 3.13 on Ubuntu; Python 3.13 on Windows.

Requirements
Python 3.11+

uv (recommended) or plain pip

An LLM API key for any provider LiteLLM supports

Installation
git clone https://github.com/Ghassem296/build-an-ai-agent-from-scratch.git
cd build-an-ai-agent-from-scratch

with uv (recommended; creates .venv and installs the exact locked versions)
uv sync

or with pip — install the modules this project imports directly
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install "litellm>=1.83.0" "jsonschema>=4.26.0" "python-dotenv>=1.2.4"
"mcp[cli]>=1.0.0,<2.0.0" "ddgs>=9.16.0"

Next, create your local environment file:

cp .env.example .env # Windows: copy .env.example .env

and set LLM_MODEL and LLM_API_KEY in it.

Configuration
All settings are read from the environment (.env is loaded automatically).

Variable	Default	Meaning
LLM_MODEL	—	Required. LiteLLM model identifier, e.g. ollama_chat/qwen2.5:3b or gemini/gemini-2.5-flash. Read only from .env; there is no default in the code.
LLM_API_KEY	—	Required. Provider key. For a local Ollama server any non-empty value works (for example ollama).
LLM_API_BASE	—	Optional OpenAI-compatible base URL (proxy, gateway, self-hosted). Leave empty for official APIs.
XAI_1_API_KEY	—	Deprecated alias for LLM_API_KEY.
LLM_TEMPERATURE	0	0–2.
LLM_TIMEOUT	60	Seconds per LLM request. Use 180 or more for a local model, whose first request has to load it.
SYSTEM_PROMPT_PROFILE	default	default (general assistant) or finance (search-first rules for news, markets and live prices).
AGENT_MAX_STEPS	10	Think-act cycles before the agent is forced to answer.
AGENT_MAX_CONTEXT_CHARS	120000	Characters of history kept; older turns are dropped beyond it.
TOOL_TIMEOUT	30	Seconds per tool call.
LOG_LEVEL	INFO	DEBUG, INFO, WARNING, ERROR, CRITICAL.
Note on LLM_MODEL and LLM_API_BASE. LLM_MODEL is read only from
.env (or the process environment); the program stops with a clear error if
it is missing. Leave LLM_API_BASE empty for official cloud APIs. For a local
Ollama server use LLM_MODEL=ollama_chat/<model> and
LLM_API_BASE=http://127.0.0.1:11434 (127.0.0.1 rather than localhost,
which can add a connection delay on Windows), plus a larger LLM_TIMEOUT.

Usage
Single prompt:

uv run python -m app.main "What is the Model Context Protocol?"

Interactive mode (keeps one conversation across turns):

uv run python -m app.main

then type your questions; 'q', 'exit' or 'quit' ends the session
Chapter-by-chapter demo (same agent, fixed prompt, prints the discovered tools):

uv run python mcp_agent.py

Quick connectivity checks live in examples/: api_connection_check.py
(uses the same LLM_MODEL, LLM_API_KEY, LLM_API_BASE, LLM_TEMPERATURE
and LLM_TIMEOUT settings as the main app), web_search_check.py,
mcp_client.py, tool_schema_example.py and legacy_agent_demo.py (the
pre-refactor single-file loop, kept for comparison).

GAIA evaluation (book section 4.8) reads problems from a local JSON/JSONL file
with the keys task_id, Question and Final answer. A real run spends API
tokens, so start small:

uv run python -m examples.gaia_agent_eval problems.jsonl --limit 2
uv run python -m examples.gaia_agent_eval problems.jsonl --no-tools # baseline

How it works
prompt
│
▼
Agent.run()
│ seed system prompt ─────────────┐
│ │ ExecutionContext
├─► step(): truncate → LLM ────────┤ (event history:
│ │ │ system / user /
│ ├── no tool calls ──► final answer
│ │ │ assistant / tool)
│ └── tool calls ──► act(): validate → MCP call_tool
│ └── ToolResult back into context
└─ max_steps reached ──► forced final answer (tools disabled)

Module	Responsibility
app/main.py	CLI, logging setup, single-prompt and interactive modes.
app/agent.py	The think-act loop, tool allowlist, schema validation, tool execution.
app/llm_client.py	LiteLLM wrapper: builds OpenAI-style messages from events, parses tool calls and usage.
app/execution_context.py	Append-only event log plus turn-aware truncation.
app/config.py	Environment parsing and validation (get_settings()).
app/mcp_runtime.py	Minimal subprocess environment and the initialized MCP session.
tools/mcp_server.py	FastMCP server exposing the search_web tool (DDGS).
Design decisions
Why LiteLLM instead of raw HTTP? Provider-agnostic interface, built-in
retries, and a stable OpenAI-style message format. Re-implementing this would
add no educational value and would make the code brittle.

Why MCP instead of a custom tool protocol? MCP is an open standard; tools
are discovered at runtime instead of being hard-coded. It also gives the agent
a clean trust boundary (allowlist + schema validation).

Why a hand-written loop? So the think-act cycle, truncation, validation,
and error handling stay visible, debuggable, and easy to reason about. This is
the part worth learning.

Why character budget instead of tokens? Simplicity. A token-aware budget
would be more precise but requires a tokenizer per model; this is a known
limitation, listed below.

Testing
uv run python -m unittest discover -s tests -v

