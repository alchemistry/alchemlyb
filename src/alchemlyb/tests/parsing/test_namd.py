"""NAMD parser tests.

"""
import pytest

from alchemlyb.parsing.namd import extract_u_nk
from alchemtest.namd import load_tyr2ala
from alchemtest.namd import load_idws


@pytest.fixture(scope="module")
def dataset():
    return load_tyr2ala()

@pytest.mark.parametrize("direction,shape",
                         [('forward', (21021, 21)),
                          ('backward', (21021, 21)),
                          ])
def test_u_nk(dataset, direction, shape):
    """Test that u_nk has the correct form when extracted from files.
    """
    for filename in dataset['data'][direction]:
        u_nk = extract_u_nk(filename, T=300)

        assert u_nk.index.names == ['time', 'fep-lambda']
        assert u_nk.shape == shape

def test_u_nk_idws():
    """Test that u_nk has the correct form when extracted from files.
    """

    filenames = load_idws()['data']['forward']
    u_nk = extract_u_nk(filenames, T=300)

    assert u_nk.index.names == ['time', 'fep-lambda']
    assert u_nk.shape == (2602, 11)
