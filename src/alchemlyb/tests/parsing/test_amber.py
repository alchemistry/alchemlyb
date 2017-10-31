"""Amber parser tests.

"""

import sys
sys.path.insert(0, "/home/shuai/Desktop/alchemlyb/alchemlyb/src")
from alchemlyb.parsing.amber import extract_dHdl
from alchemtest.amber import load_simplesolvated


def test_dHdl():
    """Test that dHdl has the correct form when extracted from files.

    """
    dataset = load_simplesolvated()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            dHdl = extract_dHdl(filename,)

            assert dHdl.index.names == ['time', 'lambdas']
            assert dHdl.shape == (500, 1)

