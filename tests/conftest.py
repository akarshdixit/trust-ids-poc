import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory):
    """A fresh fake N-BaIoT tree, same parameters as the shipped default."""
    from data.fake_nbaiot import generate
    root = tmp_path_factory.mktemp("fake_nbaiot")
    generate(str(root), rows=2500, seed=7)
    return str(root)
