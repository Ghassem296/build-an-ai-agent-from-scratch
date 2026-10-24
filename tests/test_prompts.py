import re
import unittest
from unittest.mock import MagicMock, patch

from app import main
from app.agent import (
    DEFAULT_SYSTEM_PROMPT,
    FINANCE_SYSTEM_PROMPT,
    SYSTEM_PROMPTS,
)
from app.config import VALID_SYSTEM_PROMPT_PROFILES, Settings


class TestSystemPrompts(unittest.TestCase):
    def test_profile_names_match_config(self):
        self.assertEqual(
            set(SYSTEM_PROMPTS),
            set(VALID_SYSTEM_PROMPT_PROFILES),
        )

    def test_prompts_are_well_formed(self):
        for name, prompt in SYSTEM_PROMPTS.items():
            with self.subTest(profile=name):
                self.assertEqual(prompt, prompt.strip())
                # Two sentences glued together ("...timestamps.You are...")
                # mean a missing space or newline between string pieces.
                self.assertIsNone(re.search(r"[.!?][A-Z]", prompt))
                self.assertNotIn("  ", prompt)

    def test_default_prompt_is_general(self):
        self.assertNotIn("XAU/USD", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("untrusted", DEFAULT_SYSTEM_PROMPT)

    def test_finance_prompt_content(self):
        self.assertIn("XAU/USD", FINANCE_SYSTEM_PROMPT)
        self.assertIn("You MUST search the web", FINANCE_SYSTEM_PROMPT)
        self.assertIn("untrusted", FINANCE_SYSTEM_PROMPT)
        # The finance guidance used to be pasted in twice.
        self.assertEqual(FINANCE_SYSTEM_PROMPT.count("Never invent"), 1)
        self.assertEqual(
            FINANCE_SYSTEM_PROMPT.count("For questions about"), 1
        )

    def test_build_agent_uses_selected_profile(self):
        for profile in VALID_SYSTEM_PROMPT_PROFILES:
            with self.subTest(profile=profile):
                env = {
                    "LLM_MODEL": "test-model",
                    "SYSTEM_PROMPT_PROFILE": profile,
                }

                with patch.dict("os.environ", env, clear=True):
                    settings = Settings.from_env()

                with patch(
                    "app.main.get_settings", return_value=settings
                ), patch(
                    "app.main.create_llm_client",
                    return_value=MagicMock(),
                ):
                    agent = main.build_agent(session=object())

                self.assertEqual(
                    agent.system_prompt, SYSTEM_PROMPTS[profile]
                )


if __name__ == "__main__":
    unittest.main()
