"""Collection guard for FEniCSx benchmarks.

Outside the FE container (e.g. ``uv run pytest``) DOLFINx is not installed,
so these tests are not collected. Inside the container ``RCV_REQUIRE_FENICSX=1``
is set, and a missing DOLFINx is a hard error rather than a silent skip.
"""

import importlib.util
import os

import pytest

_HAVE_DOLFINX = importlib.util.find_spec("dolfinx") is not None
_REQUIRE = os.environ.get("RCV_REQUIRE_FENICSX") == "1"

if not _HAVE_DOLFINX:
    if _REQUIRE:
        raise pytest.UsageError(
            "RCV_REQUIRE_FENICSX=1 but dolfinx is not importable; "
            "run FE benchmarks inside the FE container."
        )
    collect_ignore_glob = ["test_*.py"]


def pytest_report_header(config):
    if _HAVE_DOLFINX:
        return None
    return "FEniCSx benchmarks not collected: dolfinx not installed (use docker compose)"
