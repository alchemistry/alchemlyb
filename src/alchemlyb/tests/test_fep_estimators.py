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


class FEPestimatorMixin:
    """Mixin for all FEP Estimator test classes.

    """

    @pytest.mark.parametrize('X_delta_f', ((gmx_benzene_coul_u_nk(), 3.041),
                                           (gmx_benzene_vdw_u_nk(), -3.007)))
    def test_get_delta_f(self, X_delta_f):
        est = self.cls().fit(X_delta_f[0])
        delta_f = est.delta_f_.iloc[0, -1]

        assert X_delta_f[1] == pytest.approx(delta_f, rel=1e-3)


class TestMBAR(FEPestimatorMixin):
    """Tests for MBAR.

    """
    cls = MBAR
