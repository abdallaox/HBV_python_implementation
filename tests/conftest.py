import sys, os
# Ensure the repository root is on sys.path so tests can import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from HBV_model import HBVModel

@pytest.fixture
def model():
    return HBVModel()
