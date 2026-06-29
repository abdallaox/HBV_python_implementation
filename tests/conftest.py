import sys, os
# Ensure the repository root is on sys.path so tests can import the HBV_Lab package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from HBV_Lab import HBVModel

@pytest.fixture
def model():
    return HBVModel()
