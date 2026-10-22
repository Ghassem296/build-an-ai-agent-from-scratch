import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from tools.mcp_server import search_web


class TestSearchWeb(unittest.IsolatedAsyncioTestCase):
    async def test_search_web_returns_list(self):
        """search_web must return a list, not a coroutine."""
        fake_results = [
            {
                "title": "Test Title",
                "href": "https://example.com",
                "body": "Test snippet",
            }
        ]

        with patch("tools.mcp_server.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.return_value = fake_results
            mock_ddgs.return_value.__enter__.return_value = mock_instance

            result = await search_web("test query", max_results=1)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Test Title")
        self.assertEqual(result[0]["url"], "https://example.com")
        self.assertEqual(result[0]["snippet"], "Test snippet")

    async def test_search_web_rejects_empty_query(self):
        with self.assertRaises(ValueError):
            await search_web("", max_results=1)

    async def test_search_web_rejects_invalid_max_results(self):
        with self.assertRaises(ValueError):
            await search_web("test", max_results=0)

        with self.assertRaises(ValueError):
            await search_web("test", max_results=11)

    async def test_search_web_retries_on_failure(self):
        """search_web should retry up to 3 times before failing."""
        call_count = 0

        def failing_text(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Simulated DDGS failure")

        with patch("tools.mcp_server.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.side_effect = failing_text
            mock_ddgs.return_value.__enter__.return_value = mock_instance

            with self.assertRaises(RuntimeError):
                await search_web("test query", max_results=1)

        self.assertEqual(call_count, 3)
