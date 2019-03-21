"""NAMD parser tests.

"""

from alchemlyb.parsing.namd import extract_u_nk
from alchemtest.namd import load_tyr2ala
from numpy.testing import assert_almost_equal


def test_u_nk():
    """Test that u_nk has the correct form when extracted from files.
    """
    dataset = load_tyr2ala()

    for direction in dataset['data']:
        for filename in dataset['data'][direction]:
            u_nk = extract_u_nk(filename)

            assert u_nk.index.names == ['timestep', 'fep-lambda']
            if direction == 'forward':
                assert u_nk.shape == (21021, 21)
            elif direction == 'backward':
                assert u_nk.shape == (21021, 21)

def test_bar_namd():
    """Test BAR calculation on NAMD data.
    """
    from alchemlyb.estimators import BAR
    import numpy as np

    # load data
    dataset = load_tyr2ala()
    u_nk1 = extract_u_nk(dataset['data']['forward'][0])
    u_nk2 = extract_u_nk(dataset['data']['backward'][0])

    # combine dataframes of fwd and rev directions
    u_nk1.replace(0, np.nan, inplace=True)
    u_nk1[u_nk1.isnull()] = u_nk2
    u_nk1.replace(np.nan, 0, inplace=True)
    u_nk = u_nk1.sort_index(level=u_nk1.index.names[1:])

    # after loading BOTH fwd and rev data, do BAR calculation
    bar = BAR()
    bar.fit(u_nk)
    dg = (bar.delta_f_.iloc[0].iloc[-1])
    assert_almost_equal(dg, 6.03126982925, decimal=4)
