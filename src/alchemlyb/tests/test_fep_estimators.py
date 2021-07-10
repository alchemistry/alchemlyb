"""Tests for all FEP-based estimators in ``alchemlyb``.

"""
import pytest

import numpy as np
import pandas as pd

import alchemlyb
from alchemlyb.parsing import gmx, amber, namd, gomc
from alchemlyb.estimators import MBAR, BAR
import alchemtest.gmx
import alchemtest.amber
import alchemtest.gomc
import alchemtest.namd
from alchemtest.gmx import load_benzene
from alchemlyb.parsing.gmx import extract_u_nk

def gmx_benzene_coul_u_nk():
    dataset = alchemtest.gmx.load_benzene()

    u_nk = alchemlyb.concat([gmx.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['Coulomb']])

    return u_nk

def gmx_benzene_vdw_u_nk():
    dataset = alchemtest.gmx.load_benzene()

    u_nk = alchemlyb.concat([gmx.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['VDW']])

    return u_nk

def gmx_expanded_ensemble_case_1():
    dataset = alchemtest.gmx.load_expanded_ensemble_case_1()

    u_nk = alchemlyb.concat([gmx.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return u_nk

def gmx_expanded_ensemble_case_2():
    dataset = alchemtest.gmx.load_expanded_ensemble_case_2()

    u_nk = alchemlyb.concat([gmx.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return u_nk

def gmx_expanded_ensemble_case_3():
    dataset = alchemtest.gmx.load_expanded_ensemble_case_3()

    u_nk = alchemlyb.concat([gmx.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return u_nk

def gmx_water_particle_with_total_energy():
    dataset = alchemtest.gmx.load_water_particle_with_total_energy()

    u_nk = alchemlyb.concat([gmx.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return u_nk

def gmx_water_particle_with_potential_energy():
    dataset = alchemtest.gmx.load_water_particle_with_potential_energy()

    u_nk = alchemlyb.concat([gmx.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return u_nk

def gmx_water_particle_without_energy():
    dataset = alchemtest.gmx.load_water_particle_without_energy()

    u_nk = alchemlyb.concat([gmx.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return u_nk

def amber_bace_example_complex_vdw():
    dataset = alchemtest.amber.load_bace_example()

    u_nk = alchemlyb.concat([amber.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['complex']['vdw']])
    return u_nk

def gomc_benzene_u_nk():
    dataset = alchemtest.gomc.load_benzene()

    u_nk = alchemlyb.concat([gomc.extract_u_nk(filename, T=298)
                      for filename in dataset['data']])

    return u_nk

def namd_tyr2ala():
    dataset = alchemtest.namd.load_tyr2ala()
    u_nk1 = namd.extract_u_nk(dataset['data']['forward'][0], T=300)
    u_nk2 = namd.extract_u_nk(dataset['data']['backward'][0], T=300)

    # combine dataframes of fwd and rev directions
    u_nk1[u_nk1.isna()] = u_nk2
    u_nk = u_nk1.sort_index(level=u_nk1.index.names[1:])

    return u_nk

def namd_idws():
    dataset = alchemtest.namd.load_idws()
    u_nk = namd.extract_u_nk(dataset['data']['forward'], T=300)

    return u_nk

class FEPestimatorMixin:
    """Mixin for all FEP Estimator test classes.

    """

    def compare_delta_f(self, X_delta_f):
        est = self.cls().fit(X_delta_f[0])
        delta_f, d_delta_f = self.get_delta_f(est)

        assert X_delta_f[1] == pytest.approx(delta_f, rel=1e-3)
        assert X_delta_f[2] == pytest.approx(d_delta_f, rel=1e-3)

    def get_delta_f(self, est):
        return est.delta_f_.iloc[0, -1], est.d_delta_f_.iloc[0, -1]


class TestMBAR(FEPestimatorMixin):
    """Tests for MBAR.

    """
    cls = MBAR

    @pytest.fixture(scope="class",
                    params=[(gmx_benzene_coul_u_nk, 3.041, 0.02088),
                            (gmx_benzene_vdw_u_nk, -3.007, 0.04519),
                            (gmx_expanded_ensemble_case_1, 75.923, 0.14124),
                            (gmx_expanded_ensemble_case_2, 75.915, 0.14372),
                            (gmx_expanded_ensemble_case_3, 76.173, 0.11345),
                            (gmx_water_particle_with_total_energy, -11.680, 0.083655),
                            (gmx_water_particle_with_potential_energy, -11.675, 0.083589),
                            (gmx_water_particle_without_energy, -11.654, 0.083415),
                            (amber_bace_example_complex_vdw, 2.40200, 0.0618453),
                            (gomc_benzene_u_nk, -0.79994, 0.091579),
                    ])
    def X_delta_f(self, request):
        get_unk, E, dE = request.param
        return get_unk(), E, dE

    def test_mbar(self, X_delta_f):
        self.compare_delta_f(X_delta_f)


class TestBAR(FEPestimatorMixin):
    """Tests for BAR.

    """
    cls = BAR

    @pytest.fixture(scope="class",
                    params = [(gmx_benzene_coul_u_nk, 3.044, 0.01640),
                              (gmx_benzene_vdw_u_nk, -3.033, 0.03438),
                              (gmx_expanded_ensemble_case_1, 75.993, 0.11056),
                              (gmx_expanded_ensemble_case_2, 76.009, 0.11220),
                              (gmx_expanded_ensemble_case_3, 76.219, 0.08886),
                              (gmx_water_particle_with_total_energy, -11.675, 0.065055),
                              (gmx_water_particle_with_potential_energy, -11.724, 0.064964),
                              (gmx_water_particle_without_energy, -11.660, 0.064914),
                              (amber_bace_example_complex_vdw, 2.37846, 0.050899),
                              (namd_tyr2ala, 11.0044, 0.10235),
                              (namd_idws, 0.221147, 0.041003),
                              (gomc_benzene_u_nk, -0.87095, 0.071263),
                    ])
    def X_delta_f(self, request):
        get_unk, E, dE = request.param
        return get_unk(), E, dE

    def test_bar(self, X_delta_f):
        self.compare_delta_f(X_delta_f)

    def get_delta_f(self, est):
        ee = 0.0

        for i in range(len(est.d_delta_f_) - 1):
            ee += est.d_delta_f_.values[i][i+1]**2
        return est.delta_f_.iloc[0, -1], ee**0.5

class Test_Units():
    '''Test the units.'''

    @staticmethod
    @pytest.fixture(scope='class')
    def u_nk():
        bz = load_benzene().data
        u_nk_coul = alchemlyb.concat(
            [extract_u_nk(xvg, T=300) for xvg in bz['Coulomb']])
        u_nk_coul.attrs = extract_u_nk(load_benzene().data['Coulomb'][0], T=300).attrs
        return u_nk_coul

    def test_bar(self, u_nk):
        bar = BAR().fit(u_nk)
        assert bar.delta_f_.attrs['temperature'] == 300
        assert bar.delta_f_.attrs['energy_unit'] == 'kT'
        assert bar.d_delta_f_.attrs['temperature'] == 300
        assert bar.d_delta_f_.attrs['energy_unit'] == 'kT'

    def test_mbar(self, u_nk):
        mbar = MBAR().fit(u_nk)
        assert mbar.delta_f_.attrs['temperature'] == 300
        assert mbar.delta_f_.attrs['energy_unit'] == 'kT'
        assert mbar.d_delta_f_.attrs['temperature'] == 300
        assert mbar.d_delta_f_.attrs['energy_unit'] == 'kT'
