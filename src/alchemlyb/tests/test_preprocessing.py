"""Tests for preprocessing functions.

"""
import pytest

import pandas as pd
import numpy as np

from alchemlyb.parsing import gmx
from alchemlyb.preprocessing import (slicing, statistical_inefficiency,
                                     equilibrium_detection,)

from . import test_ti_estimators as tti
from . import test_fep_estimators as tfep

import alchemtest.gmx

@pytest.fixture(scope="module",
                params = [(tti.gmx_benzene_coul_dHdl, "single"),
                          (tti.gmx_benzene_vdw_dHdl, "single"),
                          (tti.gmx_expanded_ensemble_case_1_dHdl, "single"),
                          (tti.gmx_expanded_ensemble_case_2_dHdl, "repeat"),
                          (tti.gmx_expanded_ensemble_case_3_dHdl, "repeat"),
                          (tti.gmx_water_particle_with_total_energy_dHdl, "single"),
                          (tti.gmx_water_particle_with_potential_energy_dHdl, "single"),
                          (tti.gmx_water_particle_without_energy_dHdl, "single"),
                          (tti.amber_simplesolvated_charge_dHdl, "single"),
                          (tti.amber_simplesolvated_vdw_dHdl, "single")
                ],
                ids = ["tti.gmx_benzene_coul_dHdl",
                       "tti.gmx_benzene_vdw_dHdl",
                       "tti.gmx_expanded_ensemble_case_1_dHdl",
                       "tti.gmx_expanded_ensemble_case_2_dHdl",
                       "tti.gmx_expanded_ensemble_case_3_dHdl",
                       "tti.gmx_water_particle_with_total_energy_dHdl",
                       "tti.gmx_water_particle_with_potential_energy_dHdl",
                       "tti.gmx_water_particle_without_energy_dHdl",
                       "tti.amber_simplesolvated_charge_dHdl",
                       "tti.amber_simplesolvated_vdw_dHdl",
                ])
def dHdl(request):
    get_dHdl, nsims = request.param
    return get_dHdl(), nsims


@pytest.fixture(scope="class",
                params=[(tfep.gmx_benzene_coul_u_nk, "single"),
                        (tfep.gmx_benzene_vdw_u_nk, "single"),
                        (tfep.gmx_expanded_ensemble_case_1, "single"),
                        (tfep.gmx_expanded_ensemble_case_2, "repeat"),
                        (tfep.gmx_expanded_ensemble_case_3, "repeat"),
                        (tfep.gmx_water_particle_with_total_energy, "single"),
                        (tfep.gmx_water_particle_with_potential_energy, "single"),
                        (tfep.gmx_water_particle_without_energy, "single"),
                        (tfep.amber_bace_example_complex_vdw, "single"),
                        (tfep.gomc_benzene_u_nk, "single"),
                ],
                ids = ["tfep.gmx_benzene_coul_u_nk",
                       "tfep.gmx_benzene_vdw_u_nk",
                       "tfep.gmx_expanded_ensemble_case_1",
                       "tfep.gmx_expanded_ensemble_case_2",
                       "tfep.gmx_expanded_ensemble_case_3",
                       "tfep.gmx_water_particle_with_total_energy",
                       "tfep.gmx_water_particle_with_potential_energy",
                       "tfep.gmx_water_particle_without_energy",
                       "tfep.amber_bace_example_complex_vdw",
                       "tfep.gomc_benzene_u_nk",
                ])
def u_nk(request):
    get_unk, nsims = request.param
    return get_unk(), nsims


def gmx_benzene_dHdl():
    dataset = alchemtest.gmx.load_benzene()
    dHdl = gmx.extract_dHdl(dataset['data']['Coulomb'][0], T=300)

    return dHdl


def gmx_benzene_dHdl_duplicated():
    dataset = alchemtest.gmx.load_benzene()
    df = gmx.extract_dHdl(dataset['data']['Coulomb'][0], T=300)
    dHdl = pd.concat([df, df])

    return dHdl


def gmx_benzene_u_nk():
    dataset = alchemtest.gmx.load_benzene()
    u_nk = gmx.extract_u_nk(dataset['data']['Coulomb'][0], T=300)

    return u_nk


def gmx_benzene_u_nk_duplicated():
    dataset = alchemtest.gmx.load_benzene()
    df = gmx.extract_u_nk(dataset['data']['Coulomb'][0], T=300)
    u_nk = pd.concat([df, df])

    return u_nk


def gmx_benzene_dHdl_full():
    dataset = alchemtest.gmx.load_benzene()
    dHdl = pd.concat([gmx.extract_dHdl(i, T=300) for i in dataset['data']['Coulomb']])

    return dHdl


def gmx_benzene_u_nk_full():
    dataset = alchemtest.gmx.load_benzene()
    u_nk = pd.concat([gmx.extract_u_nk(i, T=300) for i in dataset['data']['Coulomb']])

    return u_nk


class TestSlicing:
    """Test slicing functionality.

    """
    def subsampler(self, *args, **kwargs):
        return slicing(*args, **kwargs)

    @pytest.mark.parametrize(('data', 'size'), [(gmx_benzene_dHdl(), 661),
                                                (gmx_benzene_u_nk(), 661)])
    def test_basic_slicing(self, data, size):
        assert len(self.subsampler(data, lower=1000, upper=34000, step=5)) == size

    @pytest.mark.parametrize('data', [gmx_benzene_dHdl(),
                                      gmx_benzene_u_nk()])
    def test_disordered(self, data):
        """Test that a shuffled DataFrame yields same result as unshuffled.

        """
        indices = np.arange(len(data))
        np.random.shuffle(indices)

        df = data.iloc[indices]

        assert (self.subsampler(df, lower=200) == self.subsampler(data, lower=200)).all().all()

    @pytest.mark.parametrize('data', [gmx_benzene_dHdl_duplicated(),
                                      gmx_benzene_u_nk_duplicated()])
    def test_duplicated_exception(self, data):
        """Test that a DataFrame with duplicate times for a lambda combination
        yields a KeyError.

        """
        with pytest.raises(KeyError):
            self.subsampler(data, lower=200)

    def test_slicing_dHdl(self, dHdl):
        data, nsims = dHdl

        if nsims == "single":
            dHdl_s = self.subsampler(data)
        elif nsims == "repeat":
            with pytest.raises(KeyError):
                dHdl_s = self.subsampler(data)

    def test_slicing_u_nk(self, u_nk):
        data, nsims = u_nk 
        
        if nsims == "single":
            u_nk_s = self.subsampler(data)
        elif nsims == "repeat":
            with pytest.raises(KeyError):
                u_nk_s = self.subsampler(data)


class CorrelatedPreprocessors:

    @pytest.mark.parametrize(('data', 'size', 'how'),
                             [(gmx_benzene_dHdl(), 4001, 'sum',),
                              (gmx_benzene_u_nk(), 4001, 'right')])
    def test_subsampling(self, data, size, how):
        """Basic test for execution; resulting size of dataset sensitive to
        machine and depends on algorithm.
        """
        assert len(self.subsampler(data, how=how)) <= size

    @pytest.mark.parametrize(('data', 'size', 'column'),
                             [(gmx_benzene_dHdl(), 20005, 0),
                              (gmx_benzene_u_nk(), 20005, 0)])
    def test_subsampling_column(self, data, size, column):
        assert len(self.subsampler(data, column=data.columns[column])) <= size

    def test_subsampling_dHdl(self, dHdl):
        data, nsims = dHdl

        if nsims == "single":
            dHdl_s = self.subsampler(data, how='sum')
            assert len(dHdl_s) < len(data)
        elif nsims == "repeat":
            with pytest.raises(KeyError):
                dHdl_s = self.subsampler(data, how='sum')

    def test_subsampling_u_nk(self, u_nk):
        data, nsims = u_nk 

        if nsims == "single":
            u_nk_s = self.subsampler(data, how='right')
            assert len(u_nk_s) < len(data)
        elif nsims == "repeat":
            with pytest.raises(KeyError):
                u_nk_s = self.subsampler(data, how='right')

    def test_subsampling_u_nk_left(self, u_nk):
        data, nsims = u_nk 

        if nsims == "single":
            u_nk_s = self.subsampler(data, how='left')
            assert len(u_nk_s) < len(data)
        elif nsims == "repeat":
            with pytest.raises(KeyError):
                u_nk_s = self.subsampler(data, how='left')

    def test_subsampling_u_nk_random(self, u_nk):
        data, nsims = u_nk 

        if nsims == "single":
            u_nk_s = self.subsampler(data, how='random', random_state=42)
            assert len(u_nk_s) < len(data)
        elif nsims == "repeat":
            with pytest.raises(KeyError):
                u_nk_s = self.subsampler(data, how='random', random_state=42)


class TestStatisticalInefficiency(CorrelatedPreprocessors):

    def subsampler(self, *args, **kwargs):
        return statistical_inefficiency(*args, **kwargs)

    @pytest.mark.parametrize(('conservative', 'data', 'size', 'how'),
                             [
                                 (True, gmx_benzene_dHdl(), 2001, 'sum'),  # 0.00:  g = 1.0559445620585415
                                 (True, gmx_benzene_u_nk(), 2001, 'right'),  # 'fep': g = 1.0560203916559594
                                 (False, gmx_benzene_dHdl(), 3789, 'sum'),
                                 (False, gmx_benzene_u_nk(), 3788, 'right'),
                             ])
    def test_conservative(self, data, size, conservative, how):
        sliced = self.subsampler(data, how=how, conservative=conservative)
        # results can vary slightly with different machines
        # so possibly do
        # delta = 10
        # assert size - delta < len(sliced) < size + delta
        assert len(sliced) == size


class TestEquilibriumDetection(CorrelatedPreprocessors):

    def subsampler(self, *args, **kwargs):
        return equilibrium_detection(*args, **kwargs)

    @pytest.mark.parametrize(('conservative', 'data', 'size', 'how'),
                             [
                                 (True, gmx_benzene_dHdl(), 1979, 'sum'),  # 0.00:  g = 1.0559445620585415
                                 (True, gmx_benzene_u_nk(), 1979, 'right'),  # 'fep': g = 1.0560203916559594
                                 (False, gmx_benzene_dHdl(), 2848, 'sum'),
                                 (False, gmx_benzene_u_nk(), 2849, 'right'),
                             ])
    def test_conservative(self, data, size, conservative, how):
        sliced = self.subsampler(data, how=how, conservative=conservative)
        # results can vary slightly with different machines
        # so possibly do
        # delta = 10
        # assert size - delta < len(sliced) < size + delta
        assert len(sliced) == size
