import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from alchemtest.gmx import load_benzene
from alchemlyb.parsing.gmx import extract_u_nk, extract_dHdl
from alchemlyb.estimators import MBAR, TI
from alchemlyb.visualisation.mbar_matrix import plot_mbar_overlap_matrix
from alchemlyb.visualisation.ti_dhdl import plot_ti_dhdl

def test_plot_mbar_omatrix():
    '''Just test if the plot runs'''
    bz = load_benzene().data
    u_nk_coul = pd.concat([extract_u_nk(xvg, T=300) for xvg in bz['Coulomb']])
    mbar_coul = MBAR()
    mbar_coul.fit(u_nk_coul)

    assert isinstance(plot_mbar_overlap_matrix(mbar_coul.overlap_matrix),
               matplotlib.axes.Axes)
    assert isinstance(plot_mbar_overlap_matrix(mbar_coul.overlap_matrix, [1,]),
               matplotlib.axes.Axes)

    # Bump up coverage
    overlap_maxtrix = mbar_coul.overlap_matrix
    overlap_maxtrix[0,0] = 0.0025
    overlap_maxtrix[-1, -1] = 0.9975
    assert isinstance(plot_mbar_overlap_matrix(overlap_maxtrix),
               matplotlib.axes.Axes)

def test_plot_ti_dhdl():
    '''Just test if the plot runs'''
    bz = load_benzene().data
    dHdl_coul = pd.concat([extract_dHdl(xvg, T=300) for xvg in bz['Coulomb']])
    ti_coul = TI()
    ti_coul.fit(dHdl_coul)
    assert isinstance(plot_ti_dhdl(ti_coul),
               matplotlib.axes.Axes)
    fig, ax = plt.subplots(figsize=(8, 6))
    assert isinstance(plot_ti_dhdl(ti_coul, ax=ax),
               matplotlib.axes.Axes)
    assert isinstance(plot_ti_dhdl(ti_coul, labels=['Coul']),
               matplotlib.axes.Axes)
    assert isinstance(plot_ti_dhdl(ti_coul, labels=['Coul'], colors=['r']),
               matplotlib.axes.Axes)
    dHdl_vdw = pd.concat([extract_dHdl(xvg, T=300) for xvg in bz['VDW']])
    ti_vdw = TI().fit(dHdl_vdw)
    assert isinstance(plot_ti_dhdl([ti_coul, ti_vdw]),
                      matplotlib.axes.Axes)
    dHdl_coul = pd.concat([extract_dHdl(xvg, T=300) for xvg in bz['Coulomb'][:3]])
    ti_coul = TI()
    ti_coul.fit(dHdl_coul)
    assert isinstance(plot_ti_dhdl(ti_coul),
               matplotlib.axes.Axes)

