import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest

from alchemtest.gmx import load_benzene
from alchemlyb.parsing.gmx import extract_u_nk, extract_dHdl
from alchemlyb.preprocessing import (slicing, statistical_inefficiency,
                                     equilibrium_detection)
from alchemlyb.postprocessors.units import to_kcalmol, to_kT, to_kJmol
from alchemlyb.estimators import MBAR, TI, BAR
from alchemlyb.visualisation import plot_dF_state, plot_ti_dhdl




class Test_preprocessing():
    '''Test the preprocessing module.'''
    @staticmethod
    @pytest.fixture(scope='class')
    def dhdl():
        dataset = load_benzene()
        dhdl = extract_dHdl(dataset['data']['Coulomb'][0], 310)
        return dhdl

    def test_kt2kt(self, dhdl):
        new_dhdl = to_kT(dhdl)
        assert new_dhdl.attrs['energy_unit'] == 'kT'

    def test_kj2kt(self, dhdl):
        dhdl.attrs['energy_unit'] = 'kJ/mol'
        new_dhdl = to_kT(dhdl)
        assert new_dhdl.attrs['energy_unit'] == 'kT'

    def test_kcal2kt(self, dhdl):
        dhdl.attrs['energy_unit'] = 'kcal/mol'
        new_dhdl = to_kT(dhdl)
        assert new_dhdl.attrs['energy_unit'] == 'kT'

    def test_unknown2kt(self, dhdl):
        with pytest.raises(NameError):
            dhdl.attrs['energy_unit'] = 'ddd'
            new_dhdl = to_kT(dhdl)

    def test_slicing(self, dhdl):
        '''Test if extract_u_nk assign the attr correctly'''
        dataset = load_benzene()
        u_nk = extract_u_nk(dataset['data']['Coulomb'][0], 310)
        new_u_nk = slicing(u_nk)
        assert new_u_nk.attrs['temperature'] == 310
        assert new_u_nk.attrs['energy_unit'] == 'kT'

    def test_statistical_inefficiency(self, dhdl):
        '''Test if extract_u_nk assign the attr correctly'''
        dataset = load_benzene()
        dhdl = extract_dHdl(dataset['data']['Coulomb'][0], 310)
        new_dhdl = statistical_inefficiency(dhdl)
        assert new_dhdl.attrs['temperature'] == 310
        assert new_dhdl.attrs['energy_unit'] == 'kT'

    def test_equilibrium_detection(self, dhdl):
        '''Test if extract_u_nk assign the attr correctly'''
        dataset = load_benzene()
        dhdl = extract_dHdl(dataset['data']['Coulomb'][0], 310)
        new_dhdl = equilibrium_detection(dhdl)
        assert new_dhdl.attrs['temperature'] == 310
        assert new_dhdl.attrs['energy_unit'] == 'kT'

class Test_estimator():
    '''Test the estimator module.'''
    @staticmethod
    @pytest.fixture(scope='class')
    def dhdl():
        bz = load_benzene().data
        dHdl_coul = pd.concat(
            [extract_dHdl(xvg, T=300) for xvg in bz['Coulomb']])
        dHdl_coul.attrs = extract_dHdl(load_benzene().data['Coulomb'][0], T=300).attrs
        return dHdl_coul

    @staticmethod
    @pytest.fixture(scope='class')
    def u_nk():
        bz = load_benzene().data
        u_nk_coul = pd.concat(
            [extract_u_nk(xvg, T=300) for xvg in bz['Coulomb']])
        u_nk_coul.attrs = extract_dHdl(load_benzene().data['Coulomb'][0], T=300).attrs
        return u_nk_coul

    def test_ti(self, dhdl):
        ti = TI().fit(dhdl)
        assert ti.delta_f_.attrs['temperature'] == 300
        assert ti.delta_f_.attrs['energy_unit'] == 'kT'
        assert ti.d_delta_f_.attrs['temperature'] == 300
        assert ti.d_delta_f_.attrs['energy_unit'] == 'kT'
        assert ti.dhdl.attrs['temperature'] == 300
        assert ti.dhdl.attrs['energy_unit'] == 'kT'

    def test_ti_separate_dhdl(self, dhdl):
        ti = TI().fit(dhdl)
        dhdl_list = ti.separate_dhdl()
        for dhdl in dhdl_list:
            assert dhdl.attrs['temperature'] == 300
            assert dhdl.attrs['energy_unit'] == 'kT'

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

class Test_visualisation():
    @staticmethod
    @pytest.fixture(scope='class')
    def estimaters():
        bz = load_benzene().data
        dHdl_coul = pd.concat(
            [extract_dHdl(xvg, T=300) for xvg in bz['Coulomb']])
        dHdl_coul.attrs = extract_dHdl(load_benzene().data['Coulomb'][0], T=300).attrs
        ti = TI().fit(dHdl_coul)

        u_nk_coul = pd.concat(
            [extract_u_nk(xvg, T=300) for xvg in bz['Coulomb']])
        u_nk_coul.attrs = extract_dHdl(load_benzene().data['Coulomb'][0], T=300).attrs
        mbar = MBAR().fit(u_nk_coul)

        return ti, mbar

    def test_plot_dF_state_kT(self, estimaters):
        fig = plot_dF_state(estimaters, units='kT')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_dF_state_kJ(self, estimaters):
        fig = plot_dF_state(estimaters, units='kJ/mol')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_dF_state_kcal(self, estimaters):
        fig = plot_dF_state(estimaters, units='kcal/mol')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_dF_state_unknown(self, estimaters):
        with pytest.raises(NameError):
            fig = plot_dF_state(estimaters, units='ddd')

    def test_plot_ti_dhdl_kT(self, estimaters):
        ti, mbar = estimaters
        fig = plot_ti_dhdl(ti, units='kT')
        assert isinstance(fig, matplotlib.axes.Axes)

    def test_plot_ti_dhdl_kJ(self, estimaters):
        ti, mbar = estimaters
        fig = plot_ti_dhdl(ti, units='kJ/mol')
        assert isinstance(fig, matplotlib.axes.Axes)

    def test_plot_ti_dhdl_kcal(self, estimaters):
        ti, mbar = estimaters
        fig = plot_ti_dhdl(ti, units='kcal/mol')
        assert isinstance(fig, matplotlib.axes.Axes)

    def test_plot_ti_dhdl_unknown(self, estimaters):
        ti, mbar = estimaters
        with pytest.raises(NameError):
            fig = plot_ti_dhdl(ti, units='ddd')

