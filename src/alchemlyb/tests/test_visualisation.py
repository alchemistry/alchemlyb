import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import alchemlyb
from alchemlyb.convergence import forward_backward_convergence
from alchemlyb.estimators import MBAR, TI, BAR
from alchemlyb.visualisation import plot_convergence, plot_block_average
from alchemlyb.visualisation.dF_state import plot_dF_state
from alchemlyb.visualisation.mbar_matrix import plot_mbar_overlap_matrix
from alchemlyb.visualisation.ti_dhdl import plot_ti_dhdl


def test_plot_mbar_omatrix(gmx_benzene_Coulomb_u_nk):
    """Just test if the plot runs"""
    u_nk_coul = alchemlyb.concat(gmx_benzene_Coulomb_u_nk)
    mbar_coul = MBAR()
    mbar_coul.fit(u_nk_coul)

    assert isinstance(
        plot_mbar_overlap_matrix(mbar_coul.overlap_matrix), matplotlib.axes.Axes
    )
    assert isinstance(
        plot_mbar_overlap_matrix(
            mbar_coul.overlap_matrix,
            [
                1,
            ],
        ),
        matplotlib.axes.Axes,
    )

    # Bump up coverage
    overlap_maxtrix = mbar_coul.overlap_matrix
    overlap_maxtrix[0, 0] = 0.0025
    overlap_maxtrix[-1, -1] = 0.9975
    assert isinstance(plot_mbar_overlap_matrix(overlap_maxtrix), matplotlib.axes.Axes)


def test_plot_ti_dhdl(gmx_benzene_Coulomb_dHdl, gmx_benzene_VDW_dHdl):
    """Just test if the plot runs"""
    dHdl_coul = alchemlyb.concat(gmx_benzene_Coulomb_dHdl)
    ti_coul = TI()
    ti_coul.fit(dHdl_coul)

    ax = plot_ti_dhdl(ti_coul)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(ax.figure)

    fig, ax = plt.subplots(figsize=(8, 6))
    assert isinstance(plot_ti_dhdl(ti_coul, ax=ax), matplotlib.axes.Axes)
    assert isinstance(plot_ti_dhdl(ti_coul, labels=["Coul"]), matplotlib.axes.Axes)
    assert isinstance(
        plot_ti_dhdl(ti_coul, labels=["Coul"], colors=["r"]), matplotlib.axes.Axes
    )
    plt.close(fig)

    dHdl_vdw = alchemlyb.concat(gmx_benzene_VDW_dHdl)
    ti_vdw = TI().fit(dHdl_vdw)
    ax = plot_ti_dhdl([ti_coul, ti_vdw])
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(ax.figure)

    ti_coul.dhdl = pd.DataFrame.from_dict(
        {"fep": range(100)}, orient="index", columns=np.arange(100) / 100
    ).T
    ti_coul.dhdl.attrs = dHdl_vdw.attrs
    ax = plot_ti_dhdl(ti_coul)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(ax.figure)


def test_plot_dF_state(
    gmx_benzene_Coulomb_dHdl,
    gmx_benzene_Coulomb_u_nk,
    gmx_benzene_VDW_u_nk,
    gmx_benzene_VDW_dHdl,
):
    """Just test if the plot runs"""
    u_nk_coul = alchemlyb.concat(gmx_benzene_Coulomb_u_nk)
    dHdl_coul = alchemlyb.concat(gmx_benzene_Coulomb_dHdl)
    u_nk_vdw = alchemlyb.concat(gmx_benzene_VDW_u_nk)
    dHdl_vdw = alchemlyb.concat(gmx_benzene_VDW_dHdl)

    ti_coul = TI().fit(dHdl_coul)
    ti_vdw = TI().fit(dHdl_vdw)
    bar_coul = BAR().fit(u_nk_coul)
    bar_vdw = BAR().fit(u_nk_vdw)
    mbar_coul = MBAR().fit(u_nk_coul)
    mbar_vdw = MBAR().fit(u_nk_vdw)

    dhdl_data = [
        (ti_coul, ti_vdw),
        (bar_coul, bar_vdw),
        (mbar_coul, mbar_vdw),
    ]
    fig = plot_dF_state(dhdl_data, orientation="portrait")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

    fig = plot_dF_state(dhdl_data, orientation="landscape")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

    fig = plot_dF_state(dhdl_data, labels=["MBAR", "TI", "BAR"])
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

    with pytest.raises(ValueError):
        fig = plot_dF_state(
            dhdl_data,
            labels=[
                "MBAR",
                "TI",
            ],
        )

    fig = plot_dF_state(dhdl_data, colors=["#C45AEC", "#33CC33", "#F87431"])
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

    with pytest.raises(ValueError):
        fig = plot_dF_state(dhdl_data, colors=["#C45AEC", "#33CC33"])

    with pytest.raises(ValueError):
        fig = plot_dF_state(dhdl_data, orientation="xxx")

    fig = plot_dF_state(ti_coul, orientation="landscape")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

    fig = plot_dF_state(ti_coul, orientation="portrait")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

    fig = plot_dF_state([ti_coul, bar_coul])
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)

    fig = plot_dF_state([(ti_coul, ti_vdw)])
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_convergence_dataframe(gmx_benzene_Coulomb_u_nk):
    df = forward_backward_convergence([gmx_benzene_Coulomb_u_nk[0]], "MBAR")
    ax = plot_convergence(df)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(ax.figure)


def test_plot_convergence_dataframe_noerr():
    # Test the input from R_c
    data = pd.DataFrame(
        data={
            "Forward": range(100),
            "Backward": range(100),
            "data_fraction": np.linspace(0, 1, 100),
        }
    )
    data.attrs = {"temperature": 300, "energy_unit": "kT"}
    ax = plot_convergence(data, final_error=2)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(ax.figure)


def test_plot_convergence(gmx_benzene_Coulomb_u_nk):
    data_list = gmx_benzene_Coulomb_u_nk
    forward = []
    forward_error = []
    backward = []
    backward_error = []
    num_points = 10
    for i in range(1, num_points + 1):
        # Do the forward
        slice = int(len(data_list[0]) / num_points * i)
        u_nk_coul = alchemlyb.concat([data[:slice] for data in data_list])
        estimate = MBAR().fit(u_nk_coul)
        forward.append(estimate.delta_f_.loc[0.0, 1.0])
        forward_error.append(estimate.d_delta_f_.loc[0.0, 1.0])
        # Do the backward
        u_nk_coul = alchemlyb.concat([data[-slice:] for data in data_list])
        estimate = MBAR().fit(u_nk_coul)
        backward.append(estimate.delta_f_.loc[0.0, 1.0])
        backward_error.append(estimate.d_delta_f_.loc[0.0, 1.0])

    df = pd.DataFrame(
        data={
            "Forward": forward,
            "Forward_Error": forward_error,
            "Backward": backward,
            "Backward_Error": backward_error,
        }
    )
    df.attrs = estimate.delta_f_.attrs
    ax = plot_convergence(df)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(ax.figure)


def test_plot_convergence_final_nan():
    """Test the case where the Error of the final estimate is NaN."""
    df = pd.DataFrame(
        data={
            "Forward": [1, 2],
            "Forward_Error": [np.nan, np.nan],
            "Backward": [1, 2],
            "Backward_Error": [np.nan, np.nan],
        }
    )
    df.attrs = {"temperature": 300, "energy_unit": "kT"}
    ax = plot_convergence(df)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(ax.figure)


def test_plot_block_average(gmx_benzene_Coulomb_u_nk):
    data_list = gmx_benzene_Coulomb_u_nk
    fe = []
    fe_error = []
    num_points = 10
    for i in range(1, num_points + 1):
        slice = int(len(data_list[0]) / num_points * i)
        u_nk_coul = alchemlyb.concat([data[:slice] for data in data_list])
        estimate = MBAR().fit(u_nk_coul)
        fe.append(estimate.delta_f_.loc[0, 1])
        fe_error.append(estimate.d_delta_f_.loc[0, 1])

    df = pd.DataFrame(
        data={
            "FE": fe,
            "FE_Error": fe_error,
        }
    )
    df.attrs = estimate.delta_f_.attrs
    ax = plot_block_average(df)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(ax.figure)

    ax = plot_block_average(df, units="kJ/mol")
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(ax.figure)

    df = df.drop("FE_Error", axis=1)
    ax = plot_block_average(df)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(ax.figure)

    ax = plot_block_average(df, final_error=1)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(ax.figure)

    ax = plot_block_average(df, final_error=np.inf)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(ax.figure)


class Test_Units:
    @staticmethod
    @pytest.fixture()
    def estimaters(gmx_benzene_Coulomb_dHdl, gmx_benzene_Coulomb_u_nk):
        dHdl_coul = alchemlyb.concat(gmx_benzene_Coulomb_dHdl)
        ti = TI().fit(dHdl_coul)
        u_nk_coul = alchemlyb.concat(gmx_benzene_Coulomb_u_nk)
        mbar = MBAR().fit(u_nk_coul)

        return ti, mbar

    @staticmethod
    @pytest.fixture(scope="class")
    def convergence():
        df = pd.DataFrame(
            data={
                "Forward": range(10),
                "Forward_Error": range(10),
                "Backward": range(10),
                "Backward_Error": range(10),
            }
        )
        df.attrs = {"temperature": 300, "energy_unit": "kT"}
        return df

    @pytest.mark.parametrize("units", [None, "kT", "kJ/mol", "kcal/mol"])
    def test_plot_dF_state(self, estimaters, units):
        fig = plot_dF_state(estimaters, units=units)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_dF_state_unknown(self, estimaters):
        with pytest.raises(ValueError):
            plot_dF_state(estimaters, units="ddd")

    @pytest.mark.parametrize("units", [None, "kT", "kJ/mol", "kcal/mol"])
    def test_plot_ti_dhdl(self, estimaters, units):
        ti, mbar = estimaters
        ax = plot_ti_dhdl(ti, units=units)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)

    def test_plot_ti_dhdl_unknown(self, estimaters):
        ti, mbar = estimaters
        with pytest.raises(ValueError):
            plot_ti_dhdl(ti, units="ddd")

    @pytest.mark.parametrize("units", [None, "kT", "kJ/mol", "kcal/mol"])
    def test_plot_convergence(self, convergence, units):
        ax = plot_convergence(convergence)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(ax.figure)
