import pytest

from alchemtest.gmx import load_benzene
from alchemlyb.parsing import gmx
from alchemlyb.convergence import forward_backward_convergence

@pytest.fixture()
def gmx_benzene():
    dataset = load_benzene()
    return [gmx.extract_dHdl(dhdl, T=300) for dhdl in dataset['data']['Coulomb']], \
           [gmx.extract_u_nk(dhdl, T=300) for dhdl in dataset['data']['Coulomb']]

def test_convergence_ti(gmx_benzene):
    dHdl, u_nk = gmx_benzene
    convergence = forward_backward_convergence(dHdl, 'TI')
    assert convergence.shape == (10, 5)
    assert convergence.iloc[0, 0] == pytest.approx(3.07, 0.01)
    assert convergence.iloc[0, 2] == pytest.approx(3.11, 0.01)
    assert convergence.iloc[-1, 0] == pytest.approx(3.09, 0.01)
    assert convergence.iloc[-1, 2] == pytest.approx(3.09, 0.01)

def test_convergence_mbar(gmx_benzene):
    dHdl, u_nk = gmx_benzene
    convergence = forward_backward_convergence(u_nk, 'MBAR')
    assert convergence.shape == (10, 5)
    assert convergence.iloc[0, 0] == pytest.approx(3.02, 0.01)
    assert convergence.iloc[0, 2] == pytest.approx(3.06, 0.01)
    assert convergence.iloc[-1, 0] == pytest.approx(3.05, 0.01)
    assert convergence.iloc[-1, 2] == pytest.approx(3.04, 0.01)

def test_convergence_autombar(gmx_benzene):
    dHdl, u_nk = gmx_benzene
    convergence = forward_backward_convergence(u_nk, 'AutoMBAR')
    assert convergence.shape == (10, 5)
    assert convergence.iloc[0, 0] == pytest.approx(3.02, 0.01)
    assert convergence.iloc[0, 2] == pytest.approx(3.06, 0.01)
    assert convergence.iloc[-1, 0] == pytest.approx(3.05, 0.01)
    assert convergence.iloc[-1, 2] == pytest.approx(3.04, 0.01)

def test_convergence_bar(gmx_benzene):
    dHdl, u_nk = gmx_benzene
    convergence = forward_backward_convergence(u_nk, 'BAR')
    assert convergence.shape == (10, 5)
    assert convergence.iloc[0, 0] == pytest.approx(3.02, 0.01)
    assert convergence.iloc[0, 2] == pytest.approx(3.06, 0.01)
    assert convergence.iloc[-1, 0] == pytest.approx(3.05, 0.01)
    assert convergence.iloc[-1, 2] == pytest.approx(3.04, 0.01)

def test_convergence_wrong_estimator(gmx_benzene):
    dHdl, u_nk = gmx_benzene
    with pytest.raises(ValueError, match="{} is not a valid estimator".format("www")):
        convergence = forward_backward_convergence(u_nk, 'www')
