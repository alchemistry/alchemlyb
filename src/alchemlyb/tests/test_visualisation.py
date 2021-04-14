import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest

from alchemtest.gmx import load_benzene
from alchemlyb.parsing.gmx import extract_u_nk, extract_dHdl
from alchemlyb.estimators import MBAR, TI, BAR
from alchemlyb.visualisation.mbar_matrix import plot_mbar_overlap_matrix
from alchemlyb.visualisation.ti_dhdl import plot_ti_dhdl
from alchemlyb.visualisation.dF_state import plot_dF_state
from alchemlyb.visualisation import plot_convergence

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
    ti_coul.dhdl = pd.DataFrame.from_dict(
        {'fep': range(100)},
        orient='index',
        columns=np.arange(100)/100).T
    assert isinstance(plot_ti_dhdl(ti_coul),
               matplotlib.axes.Axes)

def test_plot_dF_state():
    '''Just test if the plot runs'''
    bz = load_benzene().data
    u_nk_coul = pd.concat([extract_u_nk(xvg, T=300) for xvg in bz['Coulomb']])
    dHdl_coul = pd.concat([extract_dHdl(xvg, T=300) for xvg in bz['Coulomb']])
    u_nk_vdw = pd.concat([extract_u_nk(xvg, T=300) for xvg in bz['VDW']])
    dHdl_vdw = pd.concat([extract_dHdl(xvg, T=300) for xvg in bz['VDW']])

    ti_coul = TI().fit(dHdl_coul)
    ti_vdw = TI().fit(dHdl_vdw)
    bar_coul = BAR().fit(u_nk_coul)
    bar_vdw = BAR().fit(u_nk_vdw)
    mbar_coul = MBAR().fit(u_nk_coul)
    mbar_vdw = MBAR().fit(u_nk_vdw)

    dhdl_data = [(ti_coul, ti_vdw),
                 (bar_coul, bar_vdw),
                 (mbar_coul, mbar_vdw), ]
    fig = plot_dF_state(dhdl_data, orientation='portrait')
    assert isinstance(fig, matplotlib.figure.Figure)
    fig = plot_dF_state(dhdl_data, orientation='landscape')
    assert isinstance(fig, matplotlib.figure.Figure)
    fig = plot_dF_state(dhdl_data, labels=['MBAR', 'TI', 'BAR'])
    assert isinstance(fig, matplotlib.figure.Figure)
    with pytest.raises(ValueError):
        fig = plot_dF_state(dhdl_data, labels=['MBAR', 'TI',])
    fig = plot_dF_state(dhdl_data, colors=['#C45AEC', '#33CC33', '#F87431'])
    assert isinstance(fig, matplotlib.figure.Figure)
    with pytest.raises(ValueError):
        fig = plot_dF_state(dhdl_data, colors=['#C45AEC', '#33CC33'])
    with pytest.raises(NameError):
        fig = plot_dF_state(dhdl_data, orientation='xxx')
    fig = plot_dF_state(ti_coul, orientation='landscape')
    assert isinstance(fig, matplotlib.figure.Figure)
    fig = plot_dF_state(ti_coul, orientation='portrait')
    assert isinstance(fig, matplotlib.figure.Figure)
    fig = plot_dF_state([ti_coul, bar_coul])
    assert isinstance(fig, matplotlib.figure.Figure)
    fig = plot_dF_state([(ti_coul, ti_vdw)])
    assert isinstance(fig, matplotlib.figure.Figure)

def test_plot_convergence():
    bz = load_benzene().data
    data_list = [extract_u_nk(xvg, T=300) for xvg in bz['Coulomb']]
    forward = []
    forward_error = []
    backward = []
    backward_error = []
    num_points = 10
    for i in range(1, num_points+1):
        # Do the forward
        slice = int(len(data_list[0])/num_points*i)
        u_nk_coul = pd.concat([data[:slice] for data in data_list])
        estimate = MBAR().fit(u_nk_coul)
        forward.append(estimate.delta_f_.iloc[0,-1])
        forward_error.append(estimate.d_delta_f_.iloc[0,-1])
        # Do the backward
        u_nk_coul = pd.concat([data[-slice:] for data in data_list])
        estimate = MBAR().fit(u_nk_coul)
        backward.append(estimate.delta_f_.iloc[0,-1])
        backward_error.append(estimate.d_delta_f_.iloc[0,-1])

    assert isinstance(
        plot_convergence(forward, forward_error, backward, backward_error),
        matplotlib.axes.Axes)

