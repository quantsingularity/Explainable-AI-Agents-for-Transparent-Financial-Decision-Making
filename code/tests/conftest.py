"""Pytest configuration for the test suite (located under ``code/tests``).

The project is not a package: modules are imported with bare names such as
``from models.baseline_models import ...`` and ``from agents.xai_agent import ...``.
Those resolve when the ``code/`` directory is on ``sys.path``, so insert it here
regardless of where pytest is invoked from (no need to set ``PYTHONPATH=code``).
"""

import sys
from pathlib import Path

# code/tests/conftest.py -> parents[0]=tests, [1]=code
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
