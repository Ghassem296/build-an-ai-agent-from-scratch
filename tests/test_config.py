import unittest
from unittest.mock import patch

from app.config import Settings, require_api_key


# LLM_MODEL has no default in code, so every test that builds Settings must
# provide it, exactly like a real run gets it from ``.env``.
BASE_ENV = {"LLM_MODEL": "test-model"}


class TestSettings(unittest.TestCase):
    @patch.dict("os.environ", BASE_ENV, clear=True)
    def test_default_values(self):
        settings = Settings.from_env()

        self.assertEqual(settings.llm_model, "test-model")
        self.assertEqual(settings.llm_temperature, 0.0)
        self.assertEqual(settings.llm_timeout, 60.0)
        self.assertEqual(settings.tool_timeout, 30.0)
        self.assertEqual(settings.agent_max_steps, 10)
        self.assertEqual(settings.agent_max_context_chars, 120_000)
        self.assertEqual(settings.log_level, "INFO")
        self.assertIsNone(settings.llm_api_key)
        self.assertIsNone(settings.llm_api_base)

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "LLM_MODEL is not set"):
            Settings.from_env()

    @patch.dict("os.environ", {"LLM_MODEL": "  test-model  "}, clear=True)
    def test_model_is_stripped(self):
        settings = Settings.from_env()
        self.assertEqual(settings.llm_model, "test-model")

    @patch.dict("os.environ", {"LLM_MODEL": "   "}, clear=True)
    def test_empty_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "LLM_MODEL cannot be empty"):
            Settings.from_env()

    @patch.dict("os.environ", {"LLM_MODEL": ""}, clear=True)
    def test_blank_model_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "LLM_MODEL cannot be empty"):
            Settings.from_env()

    @patch.dict(
        "os.environ", {**BASE_ENV, "LLM_TEMPERATURE": "3"}, clear=True
    )
    def test_invalid_temperature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "LLM_TEMPERATURE"):
            Settings.from_env()

    @patch.dict("os.environ", {**BASE_ENV, "LLM_TIMEOUT": "0"}, clear=True)
    def test_non_positive_llm_timeout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "LLM_TIMEOUT"):
            Settings.from_env()

    @patch.dict("os.environ", {**BASE_ENV, "LLM_TIMEOUT": "180"}, clear=True)
    def test_llm_timeout_is_loaded(self):
        settings = Settings.from_env()
        self.assertEqual(settings.llm_timeout, 180.0)

    @patch.dict("os.environ", {**BASE_ENV, "TOOL_TIMEOUT": "-1"}, clear=True)
    def test_non_positive_tool_timeout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "TOOL_TIMEOUT"):
            Settings.from_env()

    @patch.dict(
        "os.environ", {**BASE_ENV, "AGENT_MAX_STEPS": "0"}, clear=True
    )
    def test_invalid_max_steps_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "AGENT_MAX_STEPS"):
            Settings.from_env()

    @patch.dict(
        "os.environ", {**BASE_ENV, "AGENT_MAX_STEPS": "abc"}, clear=True
    )
    def test_non_integer_max_steps_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "AGENT_MAX_STEPS"):
            Settings.from_env()

    @patch.dict("os.environ", BASE_ENV, clear=True)
    def test_missing_api_key_is_rejected_by_shared_validator(self):
        with self.assertRaisesRegex(RuntimeError, "No API key found"):
            require_api_key(Settings.from_env())

    @patch.dict(
        "os.environ", {**BASE_ENV, "LLM_API_KEY": "test-key"}, clear=True
    )
    def test_configured_api_key_passes_shared_validator(self):
        require_api_key(Settings.from_env())

    @patch.dict(
        "os.environ", {**BASE_ENV, "LLM_API_KEY": "test-key"}, clear=True
    )
    def test_api_key_is_loaded(self):
        settings = Settings.from_env()
        self.assertEqual(settings.llm_api_key, "test-key")

    @patch.dict(
        "os.environ",
        {**BASE_ENV, "LLM_API_BASE": "https://example.com/v1"},
        clear=True,
    )
    def test_api_base_is_loaded(self):
        settings = Settings.from_env()
        self.assertEqual(settings.llm_api_base, "https://example.com/v1")

    @patch.dict(
        "os.environ", {**BASE_ENV, "LOG_LEVEL": "debug"}, clear=True
    )
    def test_log_level_is_normalized(self):
        settings = Settings.from_env()
        self.assertEqual(settings.log_level, "DEBUG")

    @patch.dict(
        "os.environ", {**BASE_ENV, "LOG_LEVEL": "INVALID"}, clear=True
    )
    def test_invalid_log_level_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "LOG_LEVEL"):
            Settings.from_env()

    @patch.dict(
        "os.environ",
        {**BASE_ENV, "AGENT_MAX_CONTEXT_CHARS": "0"},
        clear=True,
    )
    def test_non_positive_context_chars_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "AGENT_MAX_CONTEXT_CHARS"
        ):
            Settings.from_env()


if __name__ == "__main__":
    unittest.main()