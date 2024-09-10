import numpy as np
import pandas as pd
import pytest

from alchemlyb import concat
from alchemlyb.convergence import (
    forward_backward_convergence,
    fwdrev_cumavg_Rc,
    A_c,
    block_average,
)
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


def test_block_average_ti(gmx_benzene_Coulomb_dHdl):
    df_avg = block_average(gmx_benzene_Coulomb_dHdl, "TI")
    assert df_avg.shape == (9, 2)
    assert df_avg.loc[1, "FE"] == pytest.approx(3.18, 0.01)
    assert df_avg.loc[1, "FE_Error"] == pytest.approx(0.07, 0.1)
    assert df_avg.loc[8, "FE"] == pytest.approx(3.15, 0.01)
    assert df_avg.loc[8, "FE_Error"] == pytest.approx(0.07, 0.1)


@pytest.mark.parametrize("estimator", ["DUMMY"])
def test_block_average_error_1(gmx_ABFE_complex_u_nk, estimator):
    with pytest.raises(ValueError, match=r"Estimator DUMMY is not available .*"):
        _ = block_average(gmx_ABFE_complex_u_nk, estimator)


@pytest.mark.parametrize("estimator", ["MBAR"])
def test_block_average_error_2_mbar(gmx_ABFE_complex_u_nk, estimator):
    df_list = gmx_ABFE_complex_u_nk[10:15]
    with pytest.raises(
        ValueError,
        match=r"Provided DataFrame, df_list\[0\] has more than one lambda value in df.index\[0\]",
    ):
        _ = block_average([concat(df_list)], estimator)

    df_list = gmx_ABFE_complex_u_nk[14:17]
    with pytest.raises(
        ValueError,
        match=r"Provided DataFrame, df_list\[0\] has more than one lambda value in df.index\[1\]",
    ):
        _ = block_average([concat(df_list)], estimator)


@pytest.mark.parametrize("estimator", ["BAR"])
def test_block_average_error_2_bar(gmx_ABFE_complex_u_nk, estimator):
    df_list = gmx_ABFE_complex_u_nk[10:13]
    with pytest.raises(
        ValueError,
        match=r"Restrict to two DataFrames, one with a fep-lambda value .*",
    ):
        _ = block_average(df_list, estimator)

    df_list = gmx_ABFE_complex_u_nk[14:17]
    with pytest.raises(
        ValueError,
        match=r"Restrict to two DataFrames, one with a fep-lambda value .*",
    ):
        _ = block_average(df_list, estimator)


@pytest.mark.parametrize("estimator", ["BAR"])
def test_block_average_error_3_bar(gmx_ABFE_complex_u_nk, estimator):
    # Test if lambda state column representing one of the two lambda
    # states in the df indices is missing from *both* dfs.
    df_list = gmx_ABFE_complex_u_nk[10:12]
    state1 = list(set(x[1:] for x in df_list[0].index))[0]
    df_list[0] = df_list[0].drop(state1, axis=1)
    df_list[1] = df_list[1].drop(state1, axis=1)
    with pytest.raises(
        ValueError,
        match=r"Indexed lambda state, .*",
    ):
        _ = block_average(df_list, estimator)


@pytest.mark.parametrize("estimator", ["BAR"])
def test_block_average_error_4_bar(gmx_ABFE_complex_u_nk, estimator):
    # Test if lambda state column representing one of the two lambda
    # states in the df indices is missing from *one* dfs.
    df_list = gmx_ABFE_complex_u_nk[10:12]
    state1 = list(set(x[1:] for x in df_list[0].index))[0]
    df_list[0] = df_list[0].drop(state1, axis=1)
    with pytest.raises(
        ValueError,
        match=r"u_nk does not contain energies computed between any adjacent .*",
    ):
        _ = block_average(df_list, estimator)


@pytest.mark.parametrize("estimator", ["BAR"])
def test_block_average_bar(gmx_ABFE_complex_u_nk, estimator):
    df_avg = block_average(gmx_ABFE_complex_u_nk[10:12], estimator)
    assert df_avg.shape == (9, 2)
    assert df_avg.loc[0, "FE"] == pytest.approx(3.701, 0.01)
    assert np.isnan(df_avg.loc[0, "FE_Error"])
    assert df_avg.loc[8, "FE"] == pytest.approx(3.603, 0.01)
    assert np.isnan(df_avg.loc[8, "FE_Error"])

    df_list = gmx_ABFE_complex_u_nk[14:16]
    df_list[-1] = df_list[-1].iloc[:-2]
    df_avg = block_average(df_list, estimator)
    assert df_avg.shape == (9, 2)
    assert df_avg.loc[0, "FE"] == pytest.approx(0.651, 0.01)
    assert np.isnan(df_avg.loc[0, "FE_Error"])
    assert df_avg.loc[8, "FE"] == pytest.approx(0.901, 0.01)
    assert np.isnan(df_avg.loc[8, "FE_Error"])


@pytest.mark.parametrize("estimator", ["MBAR"])
def test_block_average_mbar(gmx_benzene_Coulomb_u_nk, estimator):
    df_avg = block_average([gmx_benzene_Coulomb_u_nk[0]], estimator)
    assert df_avg.shape == (9, 2)
    assert df_avg.loc[0, "FE"] == pytest.approx(3.41, 0.01)
    assert df_avg.loc[0, "FE_Error"] == pytest.approx(0.22, 0.01)
    assert df_avg.loc[8, "FE"] == pytest.approx(2.83, 0.01)
    assert df_avg.loc[8, "FE_Error"] == pytest.approx(0.33, 0.01)


def test_convergence_wrong_estimator(gmx_benzene_Coulomb_dHdl):
    with pytest.raises(ValueError, match="is not available in"):
        forward_backward_convergence(gmx_benzene_Coulomb_dHdl, "WWW")


def test_convergence_wrong_cases(gmx_benzene_Coulomb_u_nk):
    with pytest.warns(DeprecationWarning, match="Using lower-case strings for"):
        forward_backward_convergence(gmx_benzene_Coulomb_u_nk, "mbar")


def test_convergence_bootstrap(gmx_benzene_Coulomb_u_nk, caplog):
    normal_c = forward_backward_convergence(gmx_benzene_Coulomb_u_nk, "mbar", num=2)
    bootstrap_c = forward_backward_convergence(
        gmx_benzene_Coulomb_u_nk, "mbar", error_tol=0.01, num=2
    )
    assert "use bootstrap error instead." in caplog.text
    assert (bootstrap_c["Forward_Error"] != normal_c["Forward_Error"]).all()


def test_convergence_method(gmx_benzene_Coulomb_u_nk):
    convergence = forward_backward_convergence(
        gmx_benzene_Coulomb_u_nk, "MBAR", num=2, method="adaptive"
    )
    assert len(convergence) == 2


@pytest.mark.parametrize("estimator", ["MBAR"])
def test_forward_backward_convergence_mbar(gmx_ABFE_complex_u_nk, estimator):
    df_list = gmx_ABFE_complex_u_nk[10:15]
    with pytest.raises(
        ValueError,
        match=r"Provided DataFrame, df_list\[0\] has more than one lambda value in df.index\[0\]",
    ):
        _ = forward_backward_convergence([concat(df_list)], estimator)

    df_list = gmx_ABFE_complex_u_nk[14:17]
    with pytest.raises(
        ValueError,
        match=r"Provided DataFrame, df_list\[0\] has more than one lambda value in df.index\[1\]",
    ):
        _ = forward_backward_convergence([concat(df_list)], estimator)


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
