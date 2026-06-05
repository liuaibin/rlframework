"""Shared pytest fixtures for rlframework tests."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def mock_backend():
    """A mock storage backend that records upload/download calls."""
    backend = MagicMock()
    backend.upload.side_effect = lambda local, remote: f"mock://{remote}"
    backend.download.side_effect = lambda remote, local: local
    return backend


@pytest.fixture
def sample_metrics():
    """Representative metrics dict as produced by RLlib training_step."""
    return {
        "env_runners": {
            "episode_return_mean": -200.5,
            "episode_len_mean": 100.0,
            "num_episodes": 10,
        },
        "learner_results": {
            "default_policy": {
                "total_loss": 0.42,
                "policy_loss": 0.21,
                "vf_loss": 0.18,
                "entropy": 0.65,
            }
        },
        "time_this_iter_s": 3.5,
        "num_env_steps_trained": 4000,
    }
