"""Global fixture arrangement."""

from pathlib import Path
from typing import Generator

import pytest
import ray


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Path to test data resources."""
    return Path(__file__).resolve().parent / "data" / "resources" / "images"


@pytest.fixture(scope="session")
def ray_session() -> Generator[None, None, None]:
    """Create a Ray session."""
    ray.init(runtime_env={"env_vars": {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}})
    yield None
    ray.shutdown()
