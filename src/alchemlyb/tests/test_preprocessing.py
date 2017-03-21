"""Tests for preprocessing functions.

"""
import pytest

import pandas as pd

from alchemlyb.parsing import gmx
from alchemlyb.preprocessing import slicing
from alchemlyb.preprocessing import statistical_inefficiency
from alchemlyb.preprocessing import equilibrium_detection

import alchemtest.gmx


def gmx_benzene_dHdl():
    dataset = alchemtest.gmx.load_benzene()
    return gmx.extract_dHdl(dataset['data']['Coulomb'][0], T=300)


def gmx_benzene_u_nk():
    dataset = alchemtest.gmx.load_benzene()
    return gmx.extract_u_nk(dataset['data']['Coulomb'][0], T=300)


class TestSlicing:
    """Test slicing functionality.

    """
    def slicer(self, *args, **kwargs):
        return slicing(*args, **kwargs)
    
    @pytest.mark.parametrize(('data', 'size'), [(gmx_benzene_dHdl(), 661),
                                                (gmx_benzene_u_nk(), 661)])
    def test_basic_slicing(self, data, size):
        assert len(self.slicer(data, lower=1000, upper=34000, step=5)) == size


class TestStatisticalInefficiency(TestSlicing):

    def slicer(self, *args, **kwargs):
        return statistical_inefficiency(*args, **kwargs)

    # resulting statistical inefficiency appears sensitive to machine
    #@pytest.mark.xfail
    @pytest.mark.parametrize(('data', 'size'), [(gmx_benzene_dHdl(), 4001),
                                                (gmx_benzene_u_nk(), 4001)])
    def test_statinef_subsampling(self, data, size):
        assert len(self.slicer(data, series=data.iloc[:, 0])) == size


class TestEquilibriumDetection(TestSlicing):

    def slicer(self, *args, **kwargs):
        return equilibrium_detection(*args, **kwargs)

    # resulting statistical inefficiency appears sensitive to machine
    #@pytest.mark.xfail
    @pytest.mark.parametrize(('data', 'size'), [(gmx_benzene_dHdl(), 4001),
                                                (gmx_benzene_u_nk(), 4001)])
    def test_equildet_subsampling(self, data, size):
        assert len(self.slicer(data, series=data.iloc[:, 0])) == size
