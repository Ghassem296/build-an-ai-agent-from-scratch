"""Manual integration check for structured output (section 4.7).

Run:
    uv run python examples/structured_output_check.py

Requires:
    - LLM_API_KEY in .env
    - MCP server available (tools/mcp_server.py)
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Literal

# --- Fix: add project root to sys.path so `import app` works ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ---------------------------------------------------------------

from pydantic import BaseModel

from app.agent import Agent
from app.config import get_settings
from app.llm_client import LlmClient
from app.mcp_runtime import mcp_session  # ← نام واقعی از mcp_runtime.py

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ----- Schema -----
class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrases: list[str]


async def main() -> None:
    settings = get_settings()
    logger.info("Model: %s", settings.llm_model)
    logger.info("API base: %s", settings.llm_api_base or "(default)")

    llm = LlmClient(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        api_base=settings.llm_api_base or None,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
    )

    async with mcp_session(_PROJECT_ROOT) as session:
        agent = Agent(
            llm_client=llm,
            mcp_session=session,
            output_type=SentimentAnalysis,
            system_prompt=(
                "You are a sentiment analyst. "
                "You MUST call the `final_answer` tool with a structured "
                "result matching the required schema. "
                "Do NOT reply with plain text."
            ),
            max_history_events=20,
            max_context_chars=settings.agent_max_context_chars,
            tool_timeout=settings.tool_timeout,
        )

        logger.info("Running agent with structured output...")
        result = await agent.run(
            "Analyze this review: "
            "'This laptop is amazing! The battery lasts forever, "
            "and the screen is gorgeous. Highly recommend.'"
        )

        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"Type:         {type(result).__name__}")
        print(f"Sentiment:    {result.sentiment}")
        print(f"Confidence:   {result.confidence}")
        print(f"Key phrases:  {result.key_phrases}")
        print("=" * 60)

        # ----- Assertions -----
        assert isinstance(result, SentimentAnalysis), (
            f"Expected SentimentAnalysis, got {type(result).__name__}"
        )
        assert result.sentiment in ("positive", "negative", "neutral")
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.key_phrases, list)
        assert len(result.key_phrases) > 0

        print("\n✅ Integration check passed!\n")


if __name__ == "__main__":
    asyncio.run(main())