from alchemtest.gmx import load_expanded_ensemble_case_1
from alchemlyb.parsing.util import anyopen


def test_gzip():
    """Test that gzip reads .gz files in the correct (text) mode.

    """
    dataset = load_expanded_ensemble_case_1()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            with anyopen(filename, 'r') as f:
                assert type(f.readline()) is str
