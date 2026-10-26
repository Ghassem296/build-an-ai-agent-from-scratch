# Contributing

Thanks for your interest! This project follows the book *Build an AI Agent from
Scratch*, so readability matters more than cleverness.

## Setup

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env     # Windows: copy .env.example .env
```

Edit `.env` and set `LLM_MODEL` (and `LLM_API_KEY` / `LLM_API_BASE` if needed).

## Checks to run before opening a PR

These are the same checks CI runs:

```bash
uv run python -m compileall -q app tools examples tests mcp_agent.py
uvx ruff check --select E9,F63,F7,F82 .
uv run python -m unittest discover -s tests -v
```

The test suite is fully offline: the LLM and the MCP session are mocked, so no
API key or network access is needed.

## Guidelines

- Add or update tests for every behaviour change. Test files must be named
  `test_*.py` so `unittest discover` picks them up.
- Keep tool output untrusted: never feed it back to the model as instructions.
- Do not leak raw exceptions to the model or the user; log them and return a
  generic message.
- Save files as UTF-8 **without** BOM (the `.editorconfig` handles this in most
  editors).
- Keep PRs small and focused, with a short description of what and why.

## Reporting security issues

See [SECURITY.md](SECURITY.md).
