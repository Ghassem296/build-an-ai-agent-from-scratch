# Build an AI Agent Control Loop from Scratch

[![Tests](https://github.com/Ghassem296/build-an-ai-agent-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/Ghassem296/build-an-ai-agent-from-scratch/actions/workflows/tests.yml)

An educational AI agent built with **[LiteLLM](https://docs.litellm.ai/)** and the **[Model Context Protocol](https://modelcontextprotocol.io/)** (MCP). The project features a hand-written think-act loop, with production-grade libraries handling the LLM and tool communication layers.

The agent sends requests to an LLM, discovers the tools exposed by an MCP server, calls tools when useful, feeds their results back to the model, and returns a final answer. The control loop is implemented in plain Python, without an agent framework.

> **Note:** This is an unofficial companion implementation inspired by *Build an AI Agent from Scratch* (Manning, by Jungjun Hur and Younghee Song). The official book repository is separate. This project is not affiliated with the authors or the publisher.

---

## Quick Start

```bash
git clone https://github.com/Ghassem296/build-an-ai-agent-from-scratch.git
cd build-an-ai-agent-from-scratch

uv sync
```

Create your local environment file:

```bash
# macOS / Linux
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Configure `LLM_MODEL` and `LLM_API_KEY` in `.env`, then run:

```bash
uv run python -m app.main "What is the Model Context Protocol?"
```

For a local Ollama server, configure the appropriate model name and base URL:

```dotenv
LLM_MODEL=ollama_chat/qwen2.5:3b
LLM_API_KEY=ollama
LLM_API_BASE=http://127.0.0.1:11434
LLM_TIMEOUT=300
```

Make sure Ollama is installed, running, and has the selected model available locally.

---

## Features

- **Hand-written agent loop:** The think-act cycle is implemented directly in Python.
- **MCP tool discovery:** Tools are discovered from a separate MCP server at runtime.
- **Tool allowlist:** Only tools advertised by the MCP server can be executed.
- **Argument validation:** Tool arguments are checked against their JSON schemas before execution.
- **Web search:** The `search_web` tool searches the web using DuckDuckGo through `ddgs`.
- **Semantic document search:** The `search_documents_tool` retrieves relevant knowledge-base passages using sentence embeddings and cosine similarity.
- **Overlapping text chunks:** Documents are split into overlapping chunks to improve retrieval of relevant passages.
- **Lazy embedding-model loading:** The sentence-transformer model is loaded when document search is first used.
- **Turn-aware history truncation:** Conversation history is trimmed at turn boundaries to preserve assistant tool calls and their corresponding results.
- **Bounded tool output:** Tool results are limited to prevent excessively large outputs from overwhelming the context.
- **Timeouts and retries:** Tool execution and LLM requests use configurable timeouts and retry behavior.
- **Safer error handling:** Internal exceptions are logged while users receive safe error messages.
- **Prompt-injection awareness:** Tool output is treated as untrusted data.
- **Minimal MCP environment:** The MCP subprocess receives an explicit environment-variable allowlist.
- **Offline automated tests:** Tests mock LLM and MCP interactions and do not require live API calls.
- **Continuous integration:** The test workflow covers Python 3.11, 3.12, and 3.13 on Ubuntu, plus Python 3.13 on Windows.

---

## What Is Hand-Written vs. Provided by Libraries?

The project is "from scratch" at the control-loop level. The surrounding infrastructure uses existing libraries.

| Layer | Implementation |
|---|---|
| Think-act loop (`app/agent.py`) | Hand-written Python |
| Turn-aware history (`app/execution_context.py`) | Hand-written Python |
| Tool allowlist and execution flow | Hand-written Python |
| Tool argument validation | Hand-written logic using `jsonschema` |
| LLM requests and response parsing (`app/llm_client.py`) | LiteLLM wrapper |
| MCP sessions and tool discovery (`app/mcp_runtime.py`) | MCP SDK |
| MCP server (`tools/mcp_server.py`) | FastMCP |
| Web search | `ddgs` |
| Semantic document search (`tools/document_search.py`) | Sentence Transformers and NumPy, with hand-written chunking and ranking logic |
| Environment loading | `python-dotenv` |

---

## Requirements

- Python **3.11 or newer**
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- An API key for your chosen LLM provider, or a local Ollama installation
- Internet access for downloading dependencies and the sentence-transformer model on its first use

Semantic document search uses `sentence-transformers/all-MiniLM-L6-v2`. The model is loaded lazily when document search is first invoked and may need to be downloaded from the model hub. The initial search can therefore take longer than subsequent searches.

The project supports CPU-based embeddings; a CUDA-enabled GPU is not required.

---

## Installation

### Recommended: uv

```bash
uv sync
```

This creates the virtual environment and installs dependencies according to the project configuration and lock file.

### Alternative: pip

Create and activate a virtual environment.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the project dependencies declared in `pyproject.toml`:

```bash
pip install -e .
```

If your chosen provider requires optional native SDKs, install the provider extras if they are configured in the project:

```bash
uv sync --extra providers
```

Create `.env` from `.env.example` and configure your LLM settings before running the agent.

---

## Configuration

The application loads configuration from environment variables, including values in `.env`.

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | Required | LiteLLM model identifier, such as `ollama_chat/qwen2.5:3b` or a supported cloud-provider model. |
| `LLM_API_KEY` | Required | Provider API key. For local Ollama, a non-empty placeholder such as `ollama` may be sufficient. |
| `LLM_API_BASE` | Empty | Optional API base URL for a proxy, gateway, or local server. |
| `XAI_1_API_KEY` | Deprecated alias | Legacy alias for `LLM_API_KEY`. |
| `LLM_TEMPERATURE` | `0` | Sampling temperature, normally between 0 and 2. |
| `LLM_TIMEOUT` | `60` | Timeout in seconds for each LLM request. Local models may need a larger value. |
| `SYSTEM_PROMPT_PROFILE` | `default` | Selects the `default` or `finance` system-prompt profile. |
| `AGENT_MAX_STEPS` | `10` | Maximum think-act cycles before the agent is asked to provide a final answer. |
| `AGENT_MAX_CONTEXT_CHARS` | `120000` | Approximate maximum number of characters retained in conversation history. |
| `TOOL_TIMEOUT` | `30` | Timeout in seconds for tool execution. |
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |

`LLM_MODEL` must be configured. For official cloud APIs, leave `LLM_API_BASE` empty unless your provider requires a custom endpoint.

For Ollama, use a model identifier supported by your LiteLLM installation and an appropriate local base URL, such as `http://127.0.0.1:11434`.

---

## Usage

### Single prompt

```bash
uv run python -m app.main "What is the Model Context Protocol?"
```

### Interactive mode

```bash
uv run python -m app.main
```

Enter questions in the interactive session. Type `q`, `exit`, or `quit` to finish.

### Chapter-by-chapter demo

```bash
uv run python mcp_agent.py
```

This runs the demonstration agent and prints the discovered tools.

### Example scripts

The `examples/` directory contains helper scripts for testing connectivity and exploring the project:

- `api_connection_check.py`
- `web_search_check.py`
- `mcp_client.py`
- `tool_schema_example.py`
- `legacy_agent_demo.py`

### GAIA evaluation

The GAIA evaluation example reads tasks from a local JSON or JSONL file containing the expected task fields.

```bash
uv run python -m examples.gaia_agent_eval problems.jsonl --limit 2
```

To run a baseline without tools:

```bash
uv run python -m examples.gaia_agent_eval problems.jsonl --no-tools
```

A real evaluation can make LLM requests and consume API credits. Start with a small number of tasks.

---

## Semantic Document Search

The project includes a small sample knowledge base for demonstrating retrieval-augmented workflows.

The document-search implementation is in `tools/document_search.py`. It uses:

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Chunking:** Text is split into chunks of 500 characters with an overlap of 80 characters.
- **Similarity:** Normalized embeddings are compared using cosine similarity, implemented as a dot product.
- **Retrieval:** The search function returns the most relevant chunks, including their identifiers, text, and similarity scores.
- **MCP integration:** The agent can access document search through `search_documents_tool`.

The current knowledge base contains fixed example documents covering topics such as password changes and resets, email updates, and billing.

### Example questions

Try asking the agent questions such as:

```text
How do I reset my password?
```

```text
How can I update my email address?
```

```text
Where can I find information about billing?
```

When the model decides document retrieval is useful, it can call `search_documents_tool` and use the returned passages to formulate its answer.

Tool selection depends on the LLM and the current prompt; asking a relevant question does not guarantee that the model will always select the document-search tool.

### Current scope

This is an educational implementation. The knowledge base is defined in code and does not currently ingest arbitrary PDF, TXT, or Markdown files. It does not use a vector database or a persistent embedding index.

---

## How It Works

```text
User prompt
    |
    v
Agent.run()
    |
    v
Step: prepare context and call LLM
    |
    +---- No tool calls ------> Final answer
    |
    +---- Tool calls
              |
              v
       Validate tool name
              |
              v
       Validate arguments
              |
              v
       Call MCP tool
              |
              v
       Append tool result
              |
              v
       Continue agent loop
```

The agent repeats the think-act cycle until the model produces a final answer or the configured step limit is reached. When the step limit is reached, the agent is asked to answer without further tool calls.

---

## Project Structure

```text
app/
    agent.py               # Hand-written think-act loop
    config.py              # Environment parsing and validation
    execution_context.py   # Event history and turn-aware truncation
    llm_client.py          # LiteLLM wrapper
    main.py                # CLI and interactive entry point
    mcp_runtime.py         # MCP session and subprocess setup

tools/
    mcp_server.py          # MCP tools, including web and document search
    document_search.py     # Chunking, embeddings, and semantic retrieval

tests/
    test_agent.py          # Agent behavior tests
    test_document_search.py # Semantic search tests
    ...                    # Tests for other modules

examples/                  # Helper scripts and demonstrations
uv.lock                    # Locked dependency graph
```

---

## Testing

Run the complete test suite:

```bash
uv run pytest tests/ -q
```

To run only the semantic document-search tests:

```bash
uv run pytest tests/test_document_search.py -q
```

To run the agent tests:

```bash
uv run pytest tests/test_agent.py -q
```

The automated tests mock external LLM and MCP interactions, so they do not need live API calls. The semantic-search tests may initialize the embedding model, depending on the test setup and available cache.

---

## Design Decisions

### Why LiteLLM instead of raw HTTP?

LiteLLM provides a common interface for different model providers, along with useful request handling and retry capabilities. Reimplementing each provider's API would add complexity without improving the educational value of the control loop.

### Why MCP instead of a custom tool protocol?

MCP provides a standard way for the agent to discover and call tools. The agent can work with tools exposed by the server without hard-coding each tool into its main loop.

### Why a hand-written loop?

The main learning goal is to make the think-act cycle, context management, argument validation, and error handling explicit and easy to inspect.

### Why character-based context limits?

Character counts are simple to implement and understand, but they only approximate token usage. A tokenizer-aware budget would be more precise.

### Why semantic document search?

Semantic retrieval demonstrates how text embeddings can help an agent find relevant knowledge even when a user's wording differs from the wording in the source documents.

---

## Limitations

- The web-search tool relies on DuckDuckGo through `ddgs` and may be rate-limited.
- The document-search knowledge base contains a small set of fixed example documents.
- External PDF, TXT, and Markdown ingestion is not implemented.
- Semantic search downloads a pretrained embedding model when it is not already cached.
- The first document-search request may be slower while the model is loading.
- There is no vector database or persistent embedding index.
- Conversation state is held in memory; persistence, resumability, and streaming are not implemented.
- Context length is measured in characters rather than tokens.
- Tool output is treated as untrusted text, but prompt injection is mitigated rather than completely prevented.

---

## License

See the repository for license information.
