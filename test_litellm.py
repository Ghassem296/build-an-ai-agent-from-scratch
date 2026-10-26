
import asyncio
from litellm import acompletion


async def main():
    response = await acompletion(
        model="ollama_chat/qwen2.5:3b",
        messages=[
            {"role": "user", "content": "Reply with exactly: LiteLLM works"}
        ],
        api_base="http://127.0.0.1:11434",
        api_key="ollama",
        timeout=180,
        extra_body={"keep_alive": "10m"},
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
