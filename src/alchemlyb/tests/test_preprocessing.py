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


@pytest.fixture()
def gmx_ABFE():
    dataset = alchemtest.gmx.load_ABFE()
    return gmx.extract_u_nk(dataset['data']['complex'][0], T=300)


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
            self.slicer(data.sort_index(0), lower=200)

    def test_multiindex_duplicated(self, gmx_ABFE):
        subsample = statistical_inefficiency(gmx_ABFE,
                                             gmx_ABFE.sum(axis=1))
        assert len(subsample) == 501

    def test_sort_off(self, gmx_ABFE):
        unsorted = pd.concat([gmx_ABFE[-500:], gmx_ABFE[:500]])
        with pytest.raises(KeyError):
            statistical_inefficiency(unsorted,
                                     unsorted.sum(axis=1),
                                     sort=False)

    def test_sort_on(self, gmx_ABFE):
        unsorted = pd.concat([gmx_ABFE[-500:], gmx_ABFE[:500]])
        subsample = statistical_inefficiency(unsorted,
                                             unsorted.sum(axis=1),
                                             sort=True)
        assert subsample.reset_index(0)['time'].is_monotonic_increasing

    def test_sort_on_noseries(self, gmx_ABFE):
        unsorted = pd.concat([gmx_ABFE[-500:], gmx_ABFE[:500]])
        subsample = statistical_inefficiency(unsorted,
                                             None,
                                             sort=True)
        assert subsample.reset_index(0)['time'].is_monotonic_increasing

    def test_duplication_off(self, gmx_ABFE):
        duplicated = pd.concat([gmx_ABFE, gmx_ABFE])
        with pytest.raises(KeyError):
            statistical_inefficiency(duplicated,
                                     duplicated.sum(axis=1),
                                     drop_duplicates=False)

    def test_duplication_on_dataframe(self, gmx_ABFE):
        duplicated = pd.concat([gmx_ABFE, gmx_ABFE])
        subsample = statistical_inefficiency(duplicated,
                                             duplicated.sum(axis=1),
                                             drop_duplicates=True)
        assert len(subsample) < 1000

    def test_duplication_on_dataframe_noseries(self, gmx_ABFE):
        duplicated = pd.concat([gmx_ABFE, gmx_ABFE])
        subsample = statistical_inefficiency(duplicated,
                                             None,
                                             drop_duplicates=True)
        assert len(subsample) == 1001

    def test_duplication_on_series(self, gmx_ABFE):
        duplicated = pd.concat([gmx_ABFE, gmx_ABFE])
        subsample = statistical_inefficiency(duplicated.sum(axis=1),
                                             duplicated.sum(axis=1),
                                             drop_duplicates=True)
        assert len(subsample) < 1000

    def test_duplication_on_series_noseries(self, gmx_ABFE):
        duplicated = pd.concat([gmx_ABFE, gmx_ABFE])
        subsample = statistical_inefficiency(duplicated.sum(axis=1),
                                             None,
                                             drop_duplicates=True)
        assert len(subsample) == 1001

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

    @pytest.mark.parametrize(('conservative', 'data', 'size'),
                             [
                                 (True, gmx_benzene_dHdl(), 2001),  # 0.00:  g = 1.0559445620585415
                                 (True, gmx_benzene_u_nk(), 2001),  # 'fep': g = 1.0560203916559594
                                 (False, gmx_benzene_dHdl(), 3789),
                                 (False, gmx_benzene_u_nk(), 3571),
                             ])
    def test_conservative(self, data, size, conservative):
        sliced = self.slicer(data, series=data.iloc[:, 0], conservative=conservative)
        # results can vary slightly with different machines
        # so possibly do
        # delta = 10
        # assert size - delta < len(sliced) < size + delta
        assert len(sliced) == size

    @pytest.mark.parametrize('series', [
        gmx_benzene_dHdl()['fep'][:20],   # wrong length
        gmx_benzene_dHdl()['fep'][::-1],  # wrong time stamps (reversed)
        ])
    def test_raise_ValueError_for_mismatched_data(self, series):
        data = gmx_benzene_dHdl()
        with pytest.raises(ValueError):
            self.slicer(data, series=series)


class TestEquilibriumDetection(TestSlicing, CorrelatedPreprocessors):

    def slicer(self, *args, **kwargs):
        return equilibrium_detection(*args, **kwargs)

