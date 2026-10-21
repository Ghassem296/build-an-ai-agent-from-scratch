# Build an AI Agent from Scratch

An AI Agent built with **LiteLLM** and **MCP (Model Context Protocol)**. The agent can talk to an LLM, discover tools exposed by an MCP server, and execute them autonomously.

Built following the book *Build an AI Agent from Scratch* (Manning, by Jungjun Hur & Younghee Song).

---

## ✨ Features

- **Tool calling** — the agent can call tools (such as web search) exposed through MCP.
- **Context-window management** — the conversation history is automatically truncated to prevent `context_length_exceeded` errors. The most recent turn is always preserved.
- **Tool allowlist** — only tools registered by the MCP server can be executed. Unknown tools are rejected.
- **Argument validation** — tool arguments are validated against the tool's JSON schema before execution.
- **Robust error handling** — network errors, tool timeouts, and LLM failures are handled gracefully. Raw exception details are logged but never leaked to the end user.
- **Retry logic** — LLM requests retry on transient failures (`num_retries`), and the web search tool retries up to 3 times with exponential backoff.
- **Safe MCP environment** — the MCP subprocess receives only a whitelisted subset of environment variables.
- **Highly testable** — 72 automated tests using `unittest`.
- **Modular architecture** — clean separation between `Agent`, `LlmClient`, `ExecutionContext`, and `Config`.
- **CI/CD** — GitHub Actions runs the test suite on Python 3.11 and 3.12.

---

## 📋 Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/) (optional but recommended)

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ghassem296/build-an-ai-agent-from-scratch.git
cd build-an-ai-agent-from-scratch