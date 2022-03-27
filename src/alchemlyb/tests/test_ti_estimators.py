"""Tests for all TI-based estimators in ``alchemlyb``.

"""
import pytest

import pandas as pd

import alchemlyb
from alchemlyb.parsing import gmx, amber, gomc
from alchemlyb.estimators import TI
import alchemtest.gmx
import alchemtest.amber
import alchemtest.gomc
from alchemtest.gmx import load_benzene, load_ABFE
from alchemlyb.parsing.gmx import extract_dHdl


def gmx_benzene_coul_dHdl():
    dataset = alchemtest.gmx.load_benzene()

    dHdl = alchemlyb.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['Coulomb']])

    return dHdl

def gmx_benzene_vdw_dHdl():
    dataset = alchemtest.gmx.load_benzene()

    dHdl = alchemlyb.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['VDW']])

    return dHdl

def gmx_expanded_ensemble_case_1_dHdl():
    dataset = alchemtest.gmx.load_expanded_ensemble_case_1()

    dHdl = alchemlyb.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return dHdl

def gmx_expanded_ensemble_case_2_dHdl():
    dataset = alchemtest.gmx.load_expanded_ensemble_case_2()

    dHdl = alchemlyb.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return dHdl

def gmx_expanded_ensemble_case_3_dHdl():
    dataset = alchemtest.gmx.load_expanded_ensemble_case_3()

    dHdl = alchemlyb.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return dHdl

def gmx_water_particle_with_total_energy_dHdl():
    dataset = alchemtest.gmx.load_water_particle_with_total_energy()

    dHdl = alchemlyb.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return dHdl

def gmx_water_particle_with_potential_energy_dHdl():
    dataset = alchemtest.gmx.load_water_particle_with_potential_energy()

    dHdl = alchemlyb.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return dHdl

def gmx_water_particle_without_energy_dHdl():
    dataset = alchemtest.gmx.load_water_particle_without_energy()

    dHdl = alchemlyb.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return dHdl

def amber_simplesolvated_charge_dHdl():
    dataset = alchemtest.amber.load_simplesolvated()

    dHdl = alchemlyb.concat([amber.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['charge']])

    return dHdl

def amber_simplesolvated_vdw_dHdl():
    dataset = alchemtest.amber.load_simplesolvated()

    dHdl = alchemlyb.concat([amber.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['vdw']])

    return dHdl

def gomc_benzene_dHdl():
    dataset = alchemtest.gomc.load_benzene()

    dHdl = alchemlyb.concat([gomc.extract_dHdl(filename, T=298)
                      for filename in dataset['data']])

    return dHdl


class TIestimatorMixin:

    def test_get_delta_f(self, X_delta_f):
        dHdl, E, dE = X_delta_f
        est = self.cls().fit(dHdl)
        delta_f = est.delta_f_.iloc[0, -1]
        d_delta_f = est.d_delta_f_.iloc[0, -1]

        assert E == pytest.approx(delta_f, rel=1e-3)
        assert dE == pytest.approx(d_delta_f, rel=1e-3)

class TestTI(TIestimatorMixin):
    """Tests for TI.

    """
    cls = TI

    T = 300
    kT_amber = amber.k_b * T

    @pytest.fixture(scope="class",
                    params = [(gmx_benzene_coul_dHdl, 3.089, 0.02157),
                              (gmx_benzene_vdw_dHdl, -3.056, 0.04863),
                              (gmx_expanded_ensemble_case_1_dHdl, 76.220, 0.15568),
                              (gmx_expanded_ensemble_case_2_dHdl, 76.247, 0.15889),
                              (gmx_expanded_ensemble_case_3_dHdl, 76.387, 0.12532),
                              (gmx_water_particle_with_total_energy_dHdl, -11.696, 0.091775),
                              (gmx_water_particle_with_potential_energy_dHdl, -11.751, 0.091149),
                              (gmx_water_particle_without_energy_dHdl, -11.687, 0.091604),
                              (amber_simplesolvated_charge_dHdl, -60.114/kT_amber, 0.08186/kT_amber),
                              (amber_simplesolvated_vdw_dHdl, 3.824/kT_amber, 0.13254/kT_amber),
                    ])
    def X_delta_f(self, request):
        get_dHdl, E, dE = request.param
        return get_dHdl(), E, dE

def test_TI_separate_dhdl_multiple_column():
    dHdl = gomc_benzene_dHdl()
    estimator = TI().fit(dHdl)
    assert all([isinstance(dhdl, pd.Series) for dhdl in estimator.separate_dhdl()])
    assert sorted([len(dhdl) for dhdl in estimator.separate_dhdl()]) == [8, 16]

def test_TI_separate_dhdl_single_column():
    dHdl = gmx_benzene_coul_dHdl()
    estimator = TI().fit(dHdl)
    assert all([isinstance(dhdl, pd.Series) for dhdl in estimator.separate_dhdl()])
    assert [len(dhdl) for dhdl in estimator.separate_dhdl()] == [5, ]

def test_TI_separate_dhdl_no_pertubed():
    '''The test for the case where two lambda are there and one is not pertubed'''
    dHdl = gmx_benzene_coul_dHdl()
    dHdl.insert(1, 'bound-lambda', [1.0, ] * len(dHdl))
    dHdl.insert(1, 'bound', [1.0, ] * len(dHdl))
    dHdl.set_index('bound-lambda', append=True, inplace=True)
    estimator = TI().fit(dHdl)
    assert all([isinstance(dhdl, pd.Series) for dhdl in estimator.separate_dhdl()])
    assert [len(dhdl) for dhdl in estimator.separate_dhdl()] == [5, ]

class Test_Units():
    '''Test the units.'''
    @staticmethod
    @pytest.fixture(scope='class')
    def dhdl():
        bz = load_benzene().data
        dHdl_coul = alchemlyb.concat(
            [extract_dHdl(xvg, T=300) for xvg in bz['Coulomb']])
        return dHdl_coul

    def test_ti(self, dhdl):
        ti = TI().fit(dhdl)
        assert ti.delta_f_.attrs['temperature'] == 300
        assert ti.delta_f_.attrs['energy_unit'] == 'kT'
        assert ti.d_delta_f_.attrs['temperature'] == 300
        assert ti.d_delta_f_.attrs['energy_unit'] == 'kT'
        assert ti.dhdl.attrs['temperature'] == 300
        assert ti.dhdl.attrs['energy_unit'] == 'kT'

    def test_ti_separate_dhdl(self, dhdl):
        ti = TI().fit(dhdl)
        dhdl_list = ti.separate_dhdl()
        for dhdl in dhdl_list:
            assert dhdl.attrs['temperature'] == 300
            assert dhdl.attrs['energy_unit'] == 'kT'

class Test_MultipleColumnUnits():
    '''Test the case where the index has multiple columns'''
    @staticmethod
    @pytest.fixture(scope='class')
    def dhdl():
        data = load_ABFE()['data']['complex']
        dhdl = alchemlyb.concat(
            [extract_dHdl(data[i],
                          300) for i in range(30)])
        return dhdl

    def test_ti_separate_dhdl(self, dhdl):
        ti = TI().fit(dhdl)
        dhdl_list = ti.separate_dhdl()
        for dhdl in dhdl_list:
            assert dhdl.attrs['temperature'] == 300
            assert dhdl.attrs['energy_unit'] == 'kT'