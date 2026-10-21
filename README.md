# Build an AI Agent from Scratch

An AI Agent built with **LiteLLM** and **MCP (Model Context Protocol)**. The agent can talk to an LLM, discover tools exposed by an MCP server, and execute them autonomously.

---

## ✨ Features

- **Tool calling** — the agent can call tools (such as web search) exposed through MCP.
- **Context-window management** — the conversation history is automatically truncated to prevent `context_length_exceeded` errors.
- **Robust error handling** — network errors, tool timeouts, and LLM failures are handled gracefully, and sensitive details are never leaked to the end user.
- **Highly testable** — 62 automated tests using `unittest`.
- **Modular architecture** — clean separation between `Agent`, `LlmClient`, `ExecutionContext`, and `Config`.

---

## 📋 Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/) (optional but recommended)

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd build-an-ai-agent-from-scratch