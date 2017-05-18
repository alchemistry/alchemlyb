"""Tests for preprocessing functions.

"""
import pytest

import pandas as pd
import numpy as np

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


def gmx_benzene_dHdl_full():
    dataset = alchemtest.gmx.load_benzene()
    return pd.concat([gmx.extract_dHdl(i, T=300) for i in dataset['data']['Coulomb']])


def gmx_benzene_u_nk_full():
    dataset = alchemtest.gmx.load_benzene()
    return pd.concat([gmx.extract_u_nk(i, T=300) for i in dataset['data']['Coulomb']])

class TestSlicing:
    """Test slicing functionality.

    """
    def slicer(self, *args, **kwargs):
        return slicing(*args, **kwargs)
    
    @pytest.mark.parametrize(('data', 'size'), [(gmx_benzene_dHdl(), 661),
                                                (gmx_benzene_u_nk(), 661)])
    def test_basic_slicing(self, data, size):
        assert len(self.slicer(data, lower=1000, upper=34000, step=5)) == size

    @pytest.mark.parametrize('data', [gmx_benzene_dHdl(),
                                      gmx_benzene_u_nk()])
    def test_disordered_exception(self, data):
        """Test that a shuffled DataFrame yields a KeyError.

        """
        indices = np.arange(len(data))
        np.random.shuffle(indices)

        df = data.iloc[indices]

        with pytest.raises(KeyError):
            self.slicer(df, lower=200)

    @pytest.mark.parametrize('data', [gmx_benzene_dHdl_full(),
                                      gmx_benzene_u_nk_full()])
    def test_duplicated_exception(self, data):
        """Test that a DataFrame with duplicate times yields a KeyError.

        """
        with pytest.raises(KeyError):
            self.slicer(data, lower=200)


class CorrelatedPreprocessors:

    @pytest.mark.parametrize(('data', 'size'), [(gmx_benzene_dHdl(), 4001),
                                                (gmx_benzene_u_nk(), 4001)])
    def test_subsampling(self, data, size):
        """Basic test for execution; resulting size of dataset sensitive to
        machine and depends on algorithm.
        """
        assert len(self.slicer(data, series=data.iloc[:, 0])) <= size

    @pytest.mark.parametrize('data', [gmx_benzene_dHdl(),
                                      gmx_benzene_u_nk()])
    def test_no_series(self, data):
        """Check that we get the same result as simple slicing with no Series.

        """
        df_sub = self.slicer(data, lower=200, upper=5000, step=2) 
        df_sliced = slicing(data, lower=200, upper=5000, step=2) 

        assert np.all((df_sub == df_sliced))


class TestStatisticalInefficiency(TestSlicing, CorrelatedPreprocessors):

    def slicer(self, *args, **kwargs):
        return statistical_inefficiency(*args, **kwargs)


class TestEquilibriumDetection(TestSlicing, CorrelatedPreprocessors):

    def slicer(self, *args, **kwargs):
        return equilibrium_detection(*args, **kwargs)
