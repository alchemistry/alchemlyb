import numpy as np
import pandas as pd
import pytest

from alchemlyb.convergence import forward_backward_convergence, fwdrev_cumavg_Rc, A_c
from alchemlyb.convergence.convergence import _cummean


def test_convergence_ti(gmx_benzene_Coulomb_dHdl):
    convergence = forward_backward_convergence(gmx_benzene_Coulomb_dHdl, "TI")
    assert convergence.shape == (10, 5)

    assert convergence.loc[0, "Forward"] == pytest.approx(3.07, 0.01)
    assert convergence.loc[0, "Backward"] == pytest.approx(3.11, 0.01)
    assert convergence.loc[9, "Forward"] == pytest.approx(3.09, 0.01)
    assert convergence.loc[9, "Backward"] == pytest.approx(3.09, 0.01)


@pytest.mark.parametrize("estimator", ["MBAR", "BAR"])
def test_convergence_fep(gmx_benzene_Coulomb_u_nk, estimator):
    convergence = forward_backward_convergence(gmx_benzene_Coulomb_u_nk, estimator)
    assert convergence.shape == (10, 5)
    assert convergence.loc[0, "Forward"] == pytest.approx(3.02, 0.01)
    assert convergence.loc[0, "Backward"] == pytest.approx(3.06, 0.01)
    assert convergence.loc[9, "Forward"] == pytest.approx(3.05, 0.01)
    assert convergence.loc[9, "Backward"] == pytest.approx(3.04, 0.01)


def test_convergence_wrong_estimator(gmx_benzene_Coulomb_dHdl):
    with pytest.raises(ValueError, match="is not available in"):
        forward_backward_convergence(gmx_benzene_Coulomb_dHdl, "WWW")


def test_convergence_wrong_cases(gmx_benzene_Coulomb_u_nk):
    with pytest.warns(DeprecationWarning, match="Using lower-case strings for"):
        forward_backward_convergence(gmx_benzene_Coulomb_u_nk, "mbar")


def test_convergence_method(gmx_benzene_Coulomb_u_nk):
    convergence = forward_backward_convergence(
        gmx_benzene_Coulomb_u_nk, "MBAR", num=2, method="adaptive"
    )
    assert len(convergence) == 2


def test_cummean_short():
    """Test the case where the input is shorter than the expected output"""
    value = _cummean(np.empty(10), 100)
    assert len(value) == 10


def test_cummean_long():
    """Test the case where the input is longer than the expected output"""
    value = _cummean(np.empty(20), 10)
    assert len(value) == 10


def test_cummean_long_none_integter():
    """Test the case where the input is not a integer multiple of the expected output"""
    value = _cummean(np.empty(25), 10)
    assert len(value) == 10


def test_R_c_converged():
    data = pd.Series(data=[0] * 100)
    data.attrs["temperature"] = 310
    data.attrs["energy_unit"] = "kcal/mol"
    value, running_average = fwdrev_cumavg_Rc(data)
    np.testing.assert_allclose(value, 0.0)


def test_R_c_notconverged():
    data = pd.Series(data=range(21))
    data.attrs["temperature"] = 310
    data.attrs["energy_unit"] = "kcal/mol"
    value, running_average = fwdrev_cumavg_Rc(data, tol=0.1, precision=0.05)
    np.testing.assert_allclose(value, 1.0)


def test_R_c_real():
    data = pd.Series(data=np.hstack((range(10), [4.5] * 10)))
    data.attrs["temperature"] = 310
    data.attrs["energy_unit"] = "kcal/mol"
    value, running_average = fwdrev_cumavg_Rc(data, tol=2.0)
    np.testing.assert_allclose(value, 0.35)


def test_A_c_real():
    data = pd.Series(data=np.hstack((range(10), [4.5] * 10)))
    data.attrs["temperature"] = 310
    data.attrs["energy_unit"] = "kcal/mol"
    value = A_c([data] * 2, tol=2.0)
    np.testing.assert_allclose(value, 0.65)
