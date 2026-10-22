"""Make the legacy agent-loop tests visible to ``unittest discover``.

``tests/legacy_agent_loop_unit.py`` holds three real tests, but the default
discovery pattern (``test*.py``) never picks the file up, so they were silently
skipped in CI. Importing the test case here puts it back into the run without
duplicating the test bodies.
"""

from __future__ import annotations

try:  # imported as ``tests.<module>`` (repo root on sys.path)
    from tests.legacy_agent_loop_unit import TestAgentLoop
except ModuleNotFoundError:  # imported as a top-level module by discovery
    from legacy_agent_loop_unit import TestAgentLoop  # type: ignore[no-redef]

__all__ = ["TestAgentLoop"]
