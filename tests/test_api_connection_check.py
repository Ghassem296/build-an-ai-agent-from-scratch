import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from examples import api_connection_check


class TestApiConnectionCheck(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = SimpleNamespace(
            llm_model="gemini/gemini-2.5-flash",
            llm_api_key="test-key",
            llm_api_base=None,
            llm_temperature=0.2,
            llm_timeout=90.0,
        )

    async def test_uses_shared_settings_and_provider_default_endpoint(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="API connection works."))],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=4),
        )
        completion = AsyncMock(return_value=response)
        output = io.StringIO()

        with (
            patch.object(api_connection_check, "get_settings", return_value=self.settings),
            patch.object(api_connection_check, "require_api_key"),
            patch.object(api_connection_check, "acompletion", completion),
            redirect_stdout(output),
        ):
            await api_connection_check.main()

        kwargs = completion.await_args.kwargs
        self.assertEqual(kwargs["model"], self.settings.llm_model)
        self.assertEqual(kwargs["api_key"], self.settings.llm_api_key)
        self.assertEqual(kwargs["temperature"], self.settings.llm_temperature)
        self.assertEqual(kwargs["timeout"], self.settings.llm_timeout)
        self.assertNotIn("api_base", kwargs)
        self.assertIn("provider default", output.getvalue())
        self.assertIn("API request succeeded.", output.getvalue())

    async def test_passes_configured_api_base(self):
        self.settings.llm_api_base = "http://localhost:11434"
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=None,
        )
        completion = AsyncMock(return_value=response)

        with (
            patch.object(api_connection_check, "get_settings", return_value=self.settings),
            patch.object(api_connection_check, "require_api_key"),
            patch.object(api_connection_check, "acompletion", completion),
            redirect_stdout(io.StringIO()),
        ):
            await api_connection_check.main()

        self.assertEqual(
            completion.await_args.kwargs["api_base"],
            "http://localhost:11434",
        )

    async def test_missing_api_key_stops_before_network_request(self):
        completion = AsyncMock()
        output = io.StringIO()

        with (
            patch.object(api_connection_check, "get_settings", return_value=self.settings),
            patch.object(
                api_connection_check,
                "require_api_key",
                side_effect=RuntimeError("No API key found."),
            ),
            patch.object(api_connection_check, "acompletion", completion),
            redirect_stdout(output),
        ):
            await api_connection_check.main()

        completion.assert_not_awaited()
        self.assertIn("No API key found.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
