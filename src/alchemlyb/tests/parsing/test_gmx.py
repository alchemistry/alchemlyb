"""Gromacs parser tests.

"""

from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk
from alchemtest.gmx import load_benzene
from alchemtest.gmx import load_expanded_ensemble_case_1
from alchemlyb.parsing.util import anyopen


def test_dHdl():
    """Test that dHdl has the correct form when extracted from files.

    """
    dataset = load_benzene()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            dHdl = extract_dHdl(filename, T=300)

            assert dHdl.index.names == ['time', 'fep-lambda']
            assert dHdl.shape == (4001, 1)

def test_u_nk():
    """Test that u_nk has the correct form when extracted from files.

    """
    dataset = load_benzene()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            u_nk = extract_u_nk(filename, T=300)

            assert u_nk.index.names == ['time', 'fep-lambda']
            if leg == 'Coulomb':
                assert u_nk.shape == (4001, 5)
            elif leg == 'VDW':
                    u_nk.shape == (4001, 16)

def test_gzip():
    """Test that gzip reads .gz files in the correct (text) mode.

    """
    dataset = load_expanded_ensemble_case_1()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            with anyopen(filename, 'r') as f:
                assert type(f.readline()) is str
