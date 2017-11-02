"""Tests for all TI-based estimators in ``alchemlyb``.

"""
import pytest

import pandas as pd

from alchemlyb.parsing import amber 
from alchemlyb.estimators import TI
import alchemtest.amber


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

    @pytest.mark.parametrize('X_delta_f', ((amber_simplesolvated_charge_dHdl(), -60.114),
                                           (amber_simplesolvated_vdw_dHdl(), 3.824)))
    def test_get_delta_f(self, X_delta_f):
        est = self.cls().fit(X_delta_f[0])
        delta_f = est.delta_f_.iloc[0, -1]
        assert X_delta_f[1] == pytest.approx(delta_f, rel=1e-3)

class TestTI(TIestimatorMixin):
    """Tests for TI.

    """
    cls = TI 

