"""Tests for all TI-based estimators in ``alchemlyb``.

"""
import pandas as pd
import numpy as np
import pytest

import alchemlyb
from alchemlyb.estimators import TI_GQ, TI
from alchemlyb.parsing import amber


@pytest.fixture
def ethanol_Coulomb(gmx_ethanol_Coulomb_dHdl):
    dHdl = alchemlyb.concat(gmx_ethanol_Coulomb_dHdl)
    return dHdl


@pytest.fixture
def ethanol_VDW(gmx_ethanol_VDW_dHdl):
    dHdl = alchemlyb.concat(gmx_ethanol_VDW_dHdl)
    return dHdl


@pytest.fixture
def ethanol(ethanol_Coulomb, ethanol_VDW):
    dHdl = alchemlyb.concat([ethanol_Coulomb, ethanol_VDW])
    return dHdl


@pytest.fixture
def tyk2_complex(amber_tyk2_example_complex):
    dHdl = alchemlyb.concat(amber_tyk2_example_complex)
    return dHdl


@pytest.fixture
def benzene_VDW(gmx_benzene_VDW_dHdl):
    dHdl = alchemlyb.concat(gmx_benzene_VDW_dHdl)
    return dHdl


@pytest.fixture
def ethanol_lambdas_means_variances(ethanol):
    dHdl = ethanol.sort_index(level=ethanol.index.names[1:])
    means = dHdl.groupby(level=dHdl.index.names[1:]).mean()
    variances = np.square(dHdl.groupby(level=dHdl.index.names[1:]).sem())
    lambdas, new_means, new_variances = TI_GQ().separate_mean_variance(means, variances)
    return lambdas, new_means, new_variances


@pytest.fixture
def tyk2_complex_lambdas_means_variances(tyk2_complex):
    dHdl = tyk2_complex.sort_index(level=tyk2_complex.index.names[1:])
    means = dHdl.groupby(level=dHdl.index.names[1:]).mean()
    variances = np.square(dHdl.groupby(level=dHdl.index.names[1:]).sem())
    lambdas, new_means, new_variances = TI_GQ().separate_mean_variance(means, variances)
    return lambdas, new_means, new_variances


class TIestimatorMixin:
    def test_get_delta_f(self, X_delta_f):
        dHdl, E, dE = X_delta_f
        est = self.cls().fit(dHdl)
        # Use .iloc[0, -1] as we want to cater for both
        # delta_f_.loc[0.0, 1.0] and delta_f_.loc[(0.0, 0.0), (0.0, 1.0)]
        delta_f = est.delta_f_.iloc[0, -1]
        d_delta_f = est.d_delta_f_.iloc[0, -1]

        assert E == pytest.approx(delta_f, rel=1e-3)
        assert dE == pytest.approx(d_delta_f, rel=1e-3)


class TestTIGQ(TIestimatorMixin):
    """Tests for TI_GQ."""

    cls = TI_GQ

    T = 300.0
    kT_amber = amber.k_b * T

    @pytest.fixture(
        params=[
            ("ethanol_Coulomb", 10.597, 0.04498),
            ("ethanol_VDW", -3.340, 0.07548),
            ("ethanol", 7.257, 0.08786),
            ("tyk2_complex", -50.504, 0.09930),
        ],
    )
    def X_delta_f(self, request):
        get_dHdl, E, dE = request.param
        return request.getfixturevalue(get_dHdl), E, dE


class Test_TI_GQ_separate_mean_and_variance_multi_column:
    """Tests for TI_GQ separate_mean_and_variance function with multiple-column data"""    

    def test_lambda_list(self, ethanol_lambdas_means_variances):
        assert isinstance(ethanol_lambdas_means_variances[0], list)

    def test_mean_list(self, ethanol_lambdas_means_variances):
        assert all([isinstance(means, pd.Series) for means in ethanol_lambdas_means_variances[1]])

    def test_variance_list(self, ethanol_lambdas_means_variances):
        assert all(
        [isinstance(variances, pd.Series) for variances in ethanol_lambdas_means_variances[2]]
    )
        
    def test_data_length(self, ethanol_lambdas_means_variances):
        assert sorted([len(means) for means in ethanol_lambdas_means_variances[1]]) == [5, 7]


class Test_TI_GQ_separate_mean_and_variance_single_column():
    """Tests for TI_GQ separate_mean_and_variance function with single-column data"""

    def test_lambda_list(self, tyk2_complex_lambdas_means_variances):
        assert isinstance(tyk2_complex_lambdas_means_variances[0], list)

    def test_mean_list(self, tyk2_complex_lambdas_means_variances):
        assert all([isinstance(means, pd.Series) for means in tyk2_complex_lambdas_means_variances[1]])

    def test_variance_list(self, tyk2_complex_lambdas_means_variances):
        assert all(
        [isinstance(variances, pd.Series) for variances in tyk2_complex_lambdas_means_variances[2]]
    )

    def test_data_length(self, tyk2_complex_lambdas_means_variances):
        assert [len(means) for means in tyk2_complex_lambdas_means_variances[1]] == [12]


def test_TI_GQ_separate_mean_variance_no_pertubed(tyk2_complex):
    """The test for the case where two lambda are there and one is not pertubed"""
    dHdl = tyk2_complex
    dHdl.insert(1, "bound-lambda", [1.0] * len(dHdl))
    dHdl.insert(1, "bound", [1.0] * len(dHdl))
    dHdl.set_index("bound-lambda", append=True, inplace=True)
    means = dHdl.groupby(level=dHdl.index.names[1:]).mean()
    variances = np.square(dHdl.groupby(level=dHdl.index.names[1:]).sem())
    _, new_means, _ = TI_GQ().separate_mean_variance(means, variances)
    assert all([isinstance(dhdl, pd.Series) for dhdl in new_means])
    assert [len(dhdl) for dhdl in new_means] == [12]


def test_TI_GQ_not_quadrature_points(benzene_VDW):
    """The test for the case where the simulation lambdas are not quadrature points"""
    dHdl = benzene_VDW
    with pytest.raises(ValueError):
        TI_GQ().fit(dHdl)


def test_TI_GQ_multi_lambda_scaling(ethanol_Coulomb):
    """The test for the case where multiple lambdas are scaled simultaneously (not supported)"""
    dHdl = ethanol_Coulomb
    # change the second lambda from all zeros to the same as the first lambda
    dHdl.reset_index(inplace=True)
    dHdl["vdw-lambda"] = dHdl["coul-lambda"]
    dHdl.set_index(["time", "coul-lambda", "vdw-lambda"], inplace=True)
    with pytest.raises(ValueError):
        TI_GQ().fit(dHdl)


class Test_Units:
    """Test the units."""

    def test_ti(self, tyk2_complex):
        ti = TI_GQ().fit(tyk2_complex)
        assert ti.delta_f_.attrs["temperature"] == 300
        assert ti.delta_f_.attrs["energy_unit"] == "kT"
        assert ti.d_delta_f_.attrs["temperature"] == 300
        assert ti.d_delta_f_.attrs["energy_unit"] == "kT"
        assert ti.dhdl.attrs["temperature"] == 300
        assert ti.dhdl.attrs["energy_unit"] == "kT"

    def test_ti_separate_mean_variance(self, tyk2_complex_lambdas_means_variances):
        dhdl_list = tyk2_complex_lambdas_means_variances[1]
        for dhdl in dhdl_list:
            assert dhdl.attrs["temperature"] == 300
            assert dhdl.attrs["energy_unit"] == "kT"


class Test_MultipleColumnUnits:
    """Test the case where the index has multiple columns"""

    @staticmethod
    @pytest.fixture
    def dhdl(ethanol):
        dhdl = ethanol
        return dhdl

    def test_ti_separate_mean_variance(self, ethanol_lambdas_means_variances):
        dhdl_list = ethanol_lambdas_means_variances[1]
        for dhdl in dhdl_list:
            assert dhdl.attrs["temperature"] == 300
            assert dhdl.attrs["energy_unit"] == "kT"


def test_TI_TIGQ_comparision(tyk2_complex):
    """Test for comparing TI and TI_GQ results"""
    
    ti = TI().fit(tyk2_complex)
    ti_gq = TI_GQ().fit(tyk2_complex)
    ti_energy_results = ti.delta_f_.iloc[0, -1]
    ti_gq_energy_results = ti_gq.delta_f_.iloc[0, -1]
    ti_variance_results = ti.d_delta_f_.iloc[0, -1]
    ti_gq_variance_results = ti_gq.d_delta_f_.iloc[0, -1]

    assert ti_energy_results == pytest.approx(ti_gq_energy_results, rel=1e-1)
    assert ti_variance_results == pytest.approx(ti_gq_variance_results, rel=1e-1)