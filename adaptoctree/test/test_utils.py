import numpy as np
import pytest

import adaptoctree.utils as utils


@pytest.mark.parametrize(
    'arr',
    [
        (np.array([0.5, 0.5, 0.5]))
    ]
)
def test_deterministic_hash(arr):

    h1 = utils.deterministic_hash(arr)
    h2 = utils.deterministic_hash(arr)

    assert h1 == h2
