# Security Policy

This is an educational project, but it executes tools on behalf of an LLM, so
security reports are welcome.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Use GitHub's
private reporting instead: open the repository's **Security** tab and choose
**Report a vulnerability**.

Include what you found, how to reproduce it, and the affected version or
commit. You can expect an initial reply within a few days.

## Scope

Examples of issues that are in scope:

- Bypassing the tool allowlist or the JSON-schema argument validation.
- Secrets (API keys, tokens) reaching the MCP subprocess, logs, or model output.
- Internal exception details leaking to the model or the user.
- Prompt-injection paths where tool output is treated as trusted instructions.

## Supported versions

Only the latest commit on `main` is supported.

## Handling secrets

Never commit real API keys. Copy `.env.example` to `.env` (which is ignored by
git) and put credentials there. If you accidentally committed a key, revoke it
at the provider first, then remove it from the history.
