"""Tests for all FEP-based estimators in ``alchemlyb``.

"""
import pytest

import pandas as pd

from alchemlyb.parsing import gmx
from alchemlyb.estimators import MBAR
import alchemtest.gmx


def gmx_benzene_coul_u_nk():
    dataset = alchemtest.gmx.load_benzene()

    u_nk = pd.concat([gmx.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['Coulomb']])

    return u_nk

def gmx_benzene_vdw_u_nk():
    dataset = alchemtest.gmx.load_benzene()

    u_nk = pd.concat([gmx.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['VDW']])

    return u_nk

def gmx_expanded_ensemble_case_1():
    dataset = alchemtest.gmx.load_expanded_ensemble_case_1()

    u_nk = pd.concat([gmx.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return u_nk

def gmx_expanded_ensemble_case_2():
    dataset = alchemtest.gmx.load_expanded_ensemble_case_2()

    u_nk = pd.concat([gmx.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return u_nk

def gmx_expanded_ensemble_case_3():
    dataset = alchemtest.gmx.load_expanded_ensemble_case_3()

    u_nk = pd.concat([gmx.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return u_nk

def gmx_water_particle_with_total_energy():
    dataset = alchemtest.gmx.load_water_particle_with_total_energy()

    u_nk = pd.concat([gmx.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return u_nk

def gmx_water_particle_with_potential_energy():
    dataset = alchemtest.gmx.load_water_particle_with_potential_energy()

    u_nk = pd.concat([gmx.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return u_nk

def gmx_water_particle_without_energy():
    dataset = alchemtest.gmx.load_water_particle_without_energy()

    u_nk = pd.concat([gmx.extract_u_nk(filename, T=300)
                      for filename in dataset['data']['AllStates']])

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

    @pytest.mark.parametrize('X_delta_f', [
        (gmx_benzene_coul_u_nk(), 3.041, 0.02088),
        (gmx_benzene_vdw_u_nk(), -3.007, 0.04519),
        (gmx_expanded_ensemble_case_1(), 75.923, 0.14124),
        (gmx_expanded_ensemble_case_2(), 75.915, 0.14372),
        (gmx_expanded_ensemble_case_3(), 76.173, 0.11345),
        (gmx_water_particle_with_total_energy(), -11.680, 0.083655),
        (gmx_water_particle_with_potential_energy(), -11.675, 0.083589),
        (gmx_water_particle_without_energy(), -11.654, 0.083415)
    ])
    def test_mbar(self, X_delta_f):
        self.compare_delta_f(X_delta_f)
