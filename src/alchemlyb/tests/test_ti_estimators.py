"""Tests for all TI-based estimators in ``alchemlyb``.

"""

import pandas as pd
import pytest

import alchemlyb
from alchemlyb.estimators import TI
from alchemlyb.parsing import amber


@pytest.fixture
def Coulomb(gmx_benzene_Coulomb_dHdl):
    dHdl = alchemlyb.concat(gmx_benzene_Coulomb_dHdl)
    return dHdl


@pytest.fixture
def VDW(gmx_benzene_VDW_dHdl):
    dHdl = alchemlyb.concat(gmx_benzene_VDW_dHdl)
    return dHdl


@pytest.fixture
def expanded_ensemble_case_1(gmx_expanded_ensemble_case_1_dHdl):
    dHdl = alchemlyb.concat(gmx_expanded_ensemble_case_1_dHdl)
    return dHdl


@pytest.fixture
def expanded_ensemble_case_2(gmx_expanded_ensemble_case_2_dHdl):
    dHdl = alchemlyb.concat(gmx_expanded_ensemble_case_2_dHdl)
    return dHdl


@pytest.fixture
def expanded_ensemble_case_3(gmx_expanded_ensemble_case_3_dHdl):
    dHdl = alchemlyb.concat(gmx_expanded_ensemble_case_3_dHdl)
    return dHdl


@pytest.fixture
def water_particle_with_total_energy(gmx_water_particle_with_total_energy_dHdl):
    dHdl = alchemlyb.concat(gmx_water_particle_with_total_energy_dHdl)
    return dHdl


@pytest.fixture
def water_particle_with_potential_energy(
    gmx_water_particle_with_potential_energy_dHdl,
):
    dHdl = alchemlyb.concat(gmx_water_particle_with_potential_energy_dHdl)
    return dHdl


@pytest.fixture
def water_particle_without_energy(gmx_water_particle_without_energy_dHdl):
    dHdl = alchemlyb.concat(gmx_water_particle_without_energy_dHdl)
    return dHdl


@pytest.fixture
def simplesolvated_charge(amber_simplesolvated_charge_dHdl):
    dHdl = alchemlyb.concat(amber_simplesolvated_charge_dHdl)
    return dHdl


@pytest.fixture
def simplesolvated_vdw(amber_simplesolvated_vdw_dHdl):
    dHdl = alchemlyb.concat(amber_simplesolvated_vdw_dHdl)
    return dHdl


@pytest.fixture
def benzene(gomc_benzene_dHdl):
    dHdl = alchemlyb.concat(gomc_benzene_dHdl)
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


class TestTI(TIestimatorMixin):
    """Tests for TI."""

    cls = TI

    T = 298.0
    kT_amber = amber.k_b * T

    @pytest.fixture(
        params=[
            ("Coulomb", 3.089, 0.02157),
            ("VDW", -3.056, 0.04863),
            ("expanded_ensemble_case_1", 76.220, 0.15568),
            ("expanded_ensemble_case_2", 76.247, 0.15889),
            ("expanded_ensemble_case_3", 76.387, 0.12532),
            ("water_particle_with_total_energy", -11.696, 0.091775),
            ("water_particle_with_potential_energy", -11.751, 0.091149),
            ("water_particle_without_energy", -11.687, 0.091604),
            ("simplesolvated_charge", -60.114 / kT_amber, 0.08186 / kT_amber),
            ("simplesolvated_vdw", 3.824 / kT_amber, 0.13254 / kT_amber),
        ],
    )
    def X_delta_f(self, request):
        get_dHdl, E, dE = request.param
        return request.getfixturevalue(get_dHdl), E, dE


def test_TI_separate_dhdl_multiple_column(benzene):
    dHdl = benzene
    estimator = TI().fit(dHdl)
    assert all([isinstance(dhdl, pd.Series) for dhdl in estimator.separate_dhdl()])
    assert sorted([len(dhdl) for dhdl in estimator.separate_dhdl()]) == [8, 16]


def test_TI_separate_dhdl_single_column(Coulomb):
    dHdl = Coulomb
    estimator = TI().fit(dHdl)
    assert all([isinstance(dhdl, pd.Series) for dhdl in estimator.separate_dhdl()])
    assert [len(dhdl) for dhdl in estimator.separate_dhdl()] == [
        5,
    ]


def test_TI_separate_dhdl_no_pertubed(Coulomb):
    """The test for the case where two lambda are there and one is not pertubed"""
    dHdl = Coulomb
    dHdl.insert(1, "bound-lambda", [1.0] * len(dHdl))
    dHdl.insert(1, "bound", [1.0] * len(dHdl))
    dHdl.set_index("bound-lambda", append=True, inplace=True)
    estimator = TI().fit(dHdl)
    assert all([isinstance(dhdl, pd.Series) for dhdl in estimator.separate_dhdl()])
    assert [len(dhdl) for dhdl in estimator.separate_dhdl()] == [5]


class Test_Units:
    """Test the units."""

    def test_ti(self, Coulomb):
        ti = TI().fit(Coulomb)
        assert ti.delta_f_.attrs["temperature"] == 300
        assert ti.delta_f_.attrs["energy_unit"] == "kT"
        assert ti.d_delta_f_.attrs["temperature"] == 300
        assert ti.d_delta_f_.attrs["energy_unit"] == "kT"
        assert ti.dhdl.attrs["temperature"] == 300
        assert ti.dhdl.attrs["energy_unit"] == "kT"

    def test_ti_separate_dhdl(self, Coulomb):
        ti = TI().fit(Coulomb)
        dhdl_list = ti.separate_dhdl()
        for dhdl in dhdl_list:
            assert dhdl.attrs["temperature"] == 300
            assert dhdl.attrs["energy_unit"] == "kT"


class Test_MultipleColumnUnits:
    """Test the case where the index has multiple columns"""

    @staticmethod
    @pytest.fixture
    def dhdl(gmx_ABFE_complex_dHdl):
        dhdl = alchemlyb.concat(gmx_ABFE_complex_dHdl)
        return dhdl

    def test_ti_separate_dhdl(self, dhdl):
        ti = TI().fit(dhdl)
        dhdl_list = ti.separate_dhdl()
        for dhdl in dhdl_list:
            assert dhdl.attrs["temperature"] == 300
            assert dhdl.attrs["energy_unit"] == "kT"
