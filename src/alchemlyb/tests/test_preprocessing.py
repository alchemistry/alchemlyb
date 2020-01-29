"""Tests for preprocessing functions.

"""
import pytest

import pandas as pd
import numpy as np

from alchemlyb.parsing import gmx
from alchemlyb.preprocessing import (slicing, statistical_inefficiency,
                                     equilibrium_detection,)

import alchemtest.gmx


def gmx_benzene_dHdl():
    dataset = alchemtest.gmx.load_benzene()
    return gmx.extract_dHdl(dataset['data']['Coulomb'][0], T=300)


def gmx_benzene_dHdl_duplicated():
    dataset = alchemtest.gmx.load_benzene()
    df = gmx.extract_dHdl(dataset['data']['Coulomb'][0], T=300)
    return pd.concat([df, df])


def gmx_benzene_u_nk():
    dataset = alchemtest.gmx.load_benzene()
    return gmx.extract_u_nk(dataset['data']['Coulomb'][0], T=300)


def gmx_benzene_u_nk_duplicated():
    dataset = alchemtest.gmx.load_benzene()
    df = gmx.extract_u_nk(dataset['data']['Coulomb'][0], T=300)
    return pd.concat([df, df])


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
    def test_disordered(self, data):
        """Test that a shuffled DataFrame yields same result as unshuffled.

        """
        indices = np.arange(len(data))
        np.random.shuffle(indices)

        df = data.iloc[indices]

        assert (self.slicer(df, lower=200) == self.slicer(data, lower=200)).all().all()

    @pytest.mark.parametrize('data', [gmx_benzene_dHdl_duplicated(),
                                      gmx_benzene_u_nk_duplicated()])
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
        assert len(self.slicer(data, data.columns[0])) <= size


class TestStatisticalInefficiency(CorrelatedPreprocessors):

    def slicer(self, *args, **kwargs):
        return statistical_inefficiency(*args, **kwargs)

    @pytest.mark.parametrize(('conservative', 'data', 'size'),
                             [
                                 (True, gmx_benzene_dHdl(), 2001),  # 0.00:  g = 1.0559445620585415
                                 (True, gmx_benzene_u_nk(), 2001),  # 'fep': g = 1.0560203916559594
                                 (False, gmx_benzene_dHdl(), 3789),
                                 (False, gmx_benzene_u_nk(), 3571),
                             ])
    def test_conservative(self, data, size, conservative):
        sliced = self.slicer(data, data.columns[0], conservative=conservative)
        # results can vary slightly with different machines
        # so possibly do
        # delta = 10
        # assert size - delta < len(sliced) < size + delta
        assert len(sliced) == size

class TestEquilibriumDetection(CorrelatedPreprocessors):

    def slicer(self, *args, **kwargs):
        return equilibrium_detection(*args, **kwargs)
