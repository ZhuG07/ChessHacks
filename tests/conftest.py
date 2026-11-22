import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch.set_num_threads(2)


def _weights_available() -> bool:
    from src.engine import resolve_checkpoint

    path = resolve_checkpoint()
    return path.exists() and path.stat().st_size > 1024


requires_weights = pytest.mark.skipif(
    not _weights_available(),
    reason="model weights not downloaded (run `python scripts/fetch_weights.py`)",
)


@pytest.fixture(scope="session")
def engine():
    if not _weights_available():
        pytest.skip("model weights not downloaded")
    from src.engine import Engine

    return Engine(device="cpu")
