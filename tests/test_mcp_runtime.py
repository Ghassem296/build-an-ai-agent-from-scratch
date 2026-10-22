import asyncio
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

from app.mcp_runtime import mcp_session


@asynccontextmanager
async def fake_stdio(_params):
    yield ("read", "write")


class SlowSession:
    def __init__(self, _read, _write):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def initialize(self):
        await asyncio.sleep(5)


class TestMcpSessionStartup(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_has_a_timeout(self):
        with (
            patch("app.mcp_runtime.stdio_client", fake_stdio),
            patch("app.mcp_runtime.ClientSession", SlowSession),
            patch("app.mcp_runtime.INIT_TIMEOUT_SECONDS", 0.05),
        ):
            with self.assertRaises(asyncio.TimeoutError):
                async with mcp_session(Path(".")):
                    pass
