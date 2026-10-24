import asyncio
from typing import Any

from litellm import acompletion

from app.config import get_settings, require_api_key


TEST_PROMPT = "Reply with exactly: API connection works."


async def main() -> None:
    """Check connectivity using the same provider settings as the app."""
    try:
        settings = get_settings()
        require_api_key(settings)
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")
        return

    endpoint = settings.llm_api_base or "provider default"
    print("Testing API connection...")
    print(f"Model: {settings.llm_model}")
    print(f"Endpoint: {endpoint}")

    request: dict[str, Any] = {
        "model": settings.llm_model,
        "api_key": settings.llm_api_key,
        "temperature": settings.llm_temperature,
        "messages": [
            {
                "role": "user",
                "content": TEST_PROMPT,
            }
        ],
        "timeout": settings.llm_timeout,
        "num_retries": 0,
    }

    # Omit api_base for official provider endpoints; LiteLLM selects them
    # from the model identifier. This also avoids overriding a provider with
    # a stale, hard-coded endpoint.
    if settings.llm_api_base:
        request["api_base"] = settings.llm_api_base

    try:
        response = await acompletion(**request)
        print("\nAPI request succeeded.")

        message = response.choices[0].message
        print("Model response:", message.content)

        if response.usage:
            print("Input tokens:", response.usage.prompt_tokens)
            print("Output tokens:", response.usage.completion_tokens)

    except Exception as error:
        print("\nAPI request failed.")
        print("Error type:", type(error).__name__)
        print("Error details:", str(error))


if __name__ == "__main__":
    asyncio.run(main())
