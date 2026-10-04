"""Shared synthetic cohort for the test suite.

Small and fast on purpose: every test uses a few dozen bootstrap replicates.
"""

import numpy as np
import pytest


@pytest.fixture
def cohort():
    rng = np.random.default_rng(0)
    n = 1500
    y = rng.binomial(1, 0.2, n).astype(float)
    p = np.clip(0.2 + 0.4 * (y - 0.2) + rng.normal(0, 0.15, n), 0.01, 0.99)
    return {
        "y": y,
        "p": p,
        "sex": rng.choice(["F", "M"], n),
        "race": rng.choice(["White", "Black", "Asian / Pacific Islander"], n, p=[0.6, 0.3, 0.1]),
        "age": rng.integers(18, 95, n),
    }


@pytest.fixture
def kw():
    return {"threshold": 0.2, "n_bootstrap": 30, "random_state": 1}
