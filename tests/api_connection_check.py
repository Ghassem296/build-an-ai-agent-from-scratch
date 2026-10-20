import asyncio
import os

from dotenv import load_dotenv
from litellm import acompletion


load_dotenv()

MODEL_NAME = "openai/gemini-2.5-flash"
API_BASE = "https://1xai.ir/v1"
API_KEY = os.getenv("XAI_1_API_KEY")


async def main():
    if not API_KEY:
        print("ERROR: XAI_1_API_KEY was not found in .env")
        return

    print("Testing API connection...")
    print(f"Model: {MODEL_NAME}")
    print(f"Endpoint: {API_BASE}")

    try:
        response = await acompletion(
            model=MODEL_NAME,
            api_base=API_BASE,
            api_key=API_KEY,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly: API connection works.",
                }
            ],
            timeout=30,
            num_retries=0,
        )

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