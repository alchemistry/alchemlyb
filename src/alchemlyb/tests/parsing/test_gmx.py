"""Gromacs parser tests.

"""

from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk
from alchemtest.gmx import load_benzene

def test_dHdl():
    """Test that dHdl has the correct form when extracted from files.

    """
    dataset = load_benzene()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            dHdl = extract_dHdl(filename, T=300)

            assert dHdl.index.names == ['time', 'fep-lambda']
