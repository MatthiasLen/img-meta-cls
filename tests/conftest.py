"""pytest configuration for IMC top-level tests."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest marks."""
    config.addinivalue_line(
        "markers",
        "slow: mark tests that require pretrained model weights or a GPU; skip with -m 'not slow'",
    )
