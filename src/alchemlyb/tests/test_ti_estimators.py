"""Tests for all TI-based estimators in ``alchemlyb``.

"""
import pytest

import pandas as pd

from alchemlyb.parsing import gmx
from alchemlyb.parsing import amber
from alchemlyb.estimators import TI
import alchemtest.gmx
import alchemtest.amber


def gmx_benzene_coul_dHdl():
    dataset = alchemtest.gmx.load_benzene()

    dHdl = pd.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['Coulomb']])

    return dHdl

def gmx_benzene_vdw_dHdl():
    dataset = alchemtest.gmx.load_benzene()

    dHdl = pd.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['VDW']])

    return dHdl

def gmx_expanded_ensemble_case_1_dHdl():
    dataset = alchemtest.gmx.load_expanded_ensemble_case_1()

    dHdl = pd.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return dHdl

def gmx_expanded_ensemble_case_2_dHdl():
    dataset = alchemtest.gmx.load_expanded_ensemble_case_2()

    dHdl = pd.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return dHdl

def gmx_expanded_ensemble_case_3_dHdl():
    dataset = alchemtest.gmx.load_expanded_ensemble_case_3()

    dHdl = pd.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return dHdl

def amber_simplesolvated_charge_dHdl():
    dataset = alchemtest.amber.load_simplesolvated()

    dHdl = pd.concat([amber.extract_dHdl(filename)
                      for filename in dataset['data']['charge']])

    return dHdl

def amber_simplesolvated_vdw_dHdl():
    dataset = alchemtest.amber.load_simplesolvated()

    dHdl = pd.concat([amber.extract_dHdl(filename)
                      for filename in dataset['data']['vdw']])

    return dHdl


class TIestimatorMixin:

    @pytest.mark.parametrize('X_delta_f', ((gmx_benzene_coul_dHdl(), 3.089),
                                           (gmx_benzene_vdw_dHdl(), -3.056),
                                           (gmx_expanded_ensemble_case_1_dHdl(), 76.220),
                                           (gmx_expanded_ensemble_case_2_dHdl(), 76.247),
                                           (gmx_expanded_ensemble_case_3_dHdl(), 76.387),
                                           (amber_simplesolvated_charge_dHdl(), -60.114),
                                           (amber_simplesolvated_vdw_dHdl(), 3.824)))
    def test_get_delta_f(self, X_delta_f):
        est = self.cls().fit(X_delta_f[0])
        delta_f = est.delta_f_.iloc[0, -1]

        assert X_delta_f[1] == pytest.approx(delta_f, rel=1e-3)

class TestTI(TIestimatorMixin):
    """Tests for TI.

    """
    cls = TI 
