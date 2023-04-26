"""Tests for all TI-based estimators in ``alchemlyb``.

"""
import pandas as pd
import pytest

import alchemlyb
from alchemlyb.estimators import TI_GQ
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


def test_TI_separate_dhdl_multiple_column(ethanol):
    dHdl = ethanol
    estimator = TI_GQ().fit(dHdl)
    assert isinstance(estimator.separate_dhdl()[0], list)
    assert all([isinstance(dhdl, pd.Series) for dhdl in estimator.separate_dhdl()[1]])
    assert all([isinstance(variances, pd.Series) for variances in estimator.separate_dhdl()[2]])
    assert sorted([len(dhdl) for dhdl in estimator.separate_dhdl()[1]]) == [5, 7]
    assert isinstance(estimator.get_quadrature_points(), dict)


def test_TI_separate_dhdl_single_column(tyk2_complex):
    dHdl = tyk2_complex
    estimator = TI_GQ().fit(dHdl)
    assert isinstance(estimator.separate_dhdl()[0], list)
    assert all([isinstance(dhdl, pd.Series) for dhdl in estimator.separate_dhdl()[1]])
    assert all([isinstance(variances, pd.Series) for variances in estimator.separate_dhdl()[2]])
    assert [len(dhdl) for dhdl in estimator.separate_dhdl()[1]] == [12]
    assert isinstance(estimator.get_quadrature_points(), dict)


def test_TI_separate_dhdl_no_pertubed(tyk2_complex):
    """The test for the case where two lambda are there and one is not pertubed"""
    dHdl = tyk2_complex
    dHdl.insert(1, "bound-lambda", [1.0] * len(dHdl))
    dHdl.insert(1, "bound", [1.0] * len(dHdl))
    dHdl.set_index("bound-lambda", append=True, inplace=True)
    estimator = TI_GQ().fit(dHdl)
    assert all([isinstance(dhdl, pd.Series) for dhdl in estimator.separate_dhdl()[1]])
    assert [len(dhdl) for dhdl in estimator.separate_dhdl()[1]] == [12]


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

    def test_ti_separate_dhdl(self, tyk2_complex):
        ti = TI_GQ().fit(tyk2_complex)
        dhdl_list = ti.separate_dhdl()[1]
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

    def test_ti_separate_dhdl(self, dhdl):
        ti = TI_GQ().fit(dhdl)
        dhdl_list = ti.separate_dhdl()[1]
        for dhdl in dhdl_list:
            assert dhdl.attrs["temperature"] == 300
            assert dhdl.attrs["energy_unit"] == "kT"
