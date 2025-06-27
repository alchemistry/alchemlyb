import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties as FP

from ..postprocessors.units import get_unit_converter


def plot_convergence(dataframe, units=None, final_error=None, ax=None):
    """Plot the forward and backward convergence.

     The input could be the result from
     :func:`~alchemlyb.convergence.forward_backward_convergence` or
     :func:`~alchemlyb.convergence.fwdrev_cumavg_Rc`. The input should be a
     :class:`pandas.DataFrame` which has column `Forward`, `Backward` and
     :attr:`pandas.DataFrame.attrs` should compile with :ref:`note-on-units`.
     The errorbar will be plotted if column `Forward_Error` and `Backward_Error`
     is present.

     `Forward`: A column of free energy estimate from the first X% of data,
     where optional `Forward_Error` column is the corresponding error.

     `Backward`: A column of free energy estimate from the last X% of data.,
     where optional `Backward_Error` column is the corresponding error.

    `final_error` is the error of the final value and is shown as the error band around the
    final value. It can be provided in case an estimate is available that is more appropriate
    than the default, which is the error of the last value in `Backward`.

     Parameters
     ----------
     dataframe : Dataframe
         Output Dataframe has column `Forward`, `Backward` or optionally
         `Forward_Error`, `Backward_Error` see :ref:`plot_convergence <plot_convergence>`.
     units : str
         The unit of the estimate. The default is `None`, which is to use the
         unit in the input. Setting this will change the output unit.
     final_error : float
         The error of the final value in ``units``. If not given, takes the last
         error in `backward_error`.
     ax : matplotlib.axes.Axes
         Matplotlib axes object where the plot will be drawn on. If ``ax=None``,
         a new axes will be generated.

     Returns
     -------
     matplotlib.axes.Axes
         An axes with the forward and backward convergence drawn.

     Note
     ----
     The code is taken and modified from
     `Alchemical Analysis <https://github.com/MobleyLab/alchemical-analysis>`_.


     .. versionchanged:: 1.0.0
         Keyword arg final_error for plotting a horizontal error bar.
         The array input has been deprecated.
         The units default to `None` which uses the units in the input.
     .. versionchanged:: 0.6.0
         data now takes in dataframe
     .. versionadded:: 0.4.0

    """
    if units is not None:
        dataframe = get_unit_converter(units)(dataframe)
    forward = dataframe["Forward"].to_numpy()
    if "Forward_Error" in dataframe:
        forward_error = dataframe["Forward_Error"].to_numpy()
    else:
        forward_error = np.zeros(len(forward))
    backward = dataframe["Backward"].to_numpy()
    if "Backward_Error" in dataframe:
        backward_error = dataframe["Backward_Error"].to_numpy()
    else:
        backward_error = np.zeros(len(backward))

    if ax is None:  # pragma: no cover
        fig, ax = plt.subplots(figsize=(8, 6))

    plt.setp(ax.spines["bottom"], color="#D2B9D3", lw=3, zorder=-2)
    plt.setp(ax.spines["left"], color="#D2B9D3", lw=3, zorder=-2)

    for dire in ["top", "right"]:
        ax.spines[dire].set_color("none")

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    f_ts = np.linspace(0, 1, len(forward) + 1)[1:]
    r_ts = np.linspace(0, 1, len(backward) + 1)[1:]

    if final_error is None:
        final_error = backward_error[-1]

    if np.isfinite(backward[-1]) and np.isfinite(final_error):
        ax.fill_between(
            [0, 1],
            backward[-1] - final_error,
            backward[-1] + final_error,
            color="#D2B9D3",
            zorder=1,
        )
    line1 = ax.errorbar(
        f_ts,
        forward,
        yerr=forward_error,
        color="#736AFF",
        lw=3,
        zorder=2,
        marker="o",
        mfc="w",
        mew=2.5,
        mec="#736AFF",
        ms=12,
    )
    line2 = ax.errorbar(
        r_ts,
        backward,
        yerr=backward_error,
        color="#C11B17",
        lw=3,
        zorder=3,
        marker="o",
        mfc="w",
        mew=2.5,
        mec="#C11B17",
        ms=12,
    )

    xticks_spacing = len(r_ts) // 10 or 1
    xticks = r_ts[::xticks_spacing]
    plt.xticks(xticks, [f"{i:.2f}" for i in xticks], fontsize=10)
    plt.yticks(fontsize=10)

    ax.legend(
        (line1[0], line2[0]),
        ("Forward", "Reverse"),
        loc="best",
        prop=FP(size=18),
        frameon=False,
    )
    ax.set_xlabel(r"Fraction of the simulation time", fontsize=16, color="#151B54")
    ax.set_ylabel(r"$\Delta G$ ({})".format(units), fontsize=16, color="#151B54")
    plt.tick_params(axis="x", color="#D2B9D3")
    plt.tick_params(axis="y", color="#D2B9D3")
    plt.tight_layout()
    return ax


def plot_block_average(dataframe, units=None, final_error=None, ax=None):
    """Plot the forward and backward convergence.

     The input could be the result from
     :func:`~alchemlyb.convergence.forward_backward_convergence` or
     :func:`~alchemlyb.convergence.fwdrev_cumavg_Rc`. The input should be a
     :class:`pandas.DataFrame` which has column `FE` and
     :attr:`pandas.DataFrame.attrs` should compile with :ref:`note-on-units`.
     The errorbar will be plotted if column `FE_Error` and `Backward_Error`
     is present.

     `FE`: A column of free energy estimate from some X% block of the data,
     where optional `FE_Error` column is the corresponding error.

    `final_error` is the error of the final value and is shown as the error band around the
    final value. It can be provided in case an estimate is available that is more appropriate
    than the default, which is the error of the last value in `Backward`.

     Parameters
     ----------
     dataframe : Dataframe
         Output Dataframe has column `Forward`, `Backward` or optionally
         `Forward_Error`, `Backward_Error` see :ref:`plot_convergence <plot_convergence>`.
     units : str
         The unit of the estimate. The default is `None`, which is to use the
         unit in the input. Setting this will change the output unit.
     final_error : float
         The error (standard deviation) of the final value in ``units``. If not given, takes the
         overall error of the time blocks, unless these were not provided, it which case it
         equals 1 kT.
     ax : matplotlib.axes.Axes
         Matplotlib axes object where the plot will be drawn on. If ``ax=None``,
         a new axes will be generated.

     Returns
     -------
     matplotlib.axes.Axes
         An axes with the forward and backward convergence drawn.

     Note
     ----
     The code is taken and modified from
     `Alchemical Analysis <https://github.com/MobleyLab/alchemical-analysis>`_.

     .. versionadded:: 2.4.0

    """
    if units is not None:
        dataframe = get_unit_converter(units)(dataframe)
    df_avg = dataframe["FE"].to_numpy()
    if "FE_Error" in dataframe:
        df_avg_error = dataframe["FE_Error"].to_numpy()
    else:
        df_avg_error = np.zeros(len(df_avg))

    if ax is None:  # pragma: no cover
        fig, ax = plt.subplots(figsize=(8, 6))

    plt.setp(ax.spines["bottom"], color="#D2B9D3", lw=3, zorder=-2)
    plt.setp(ax.spines["left"], color="#D2B9D3", lw=3, zorder=-2)

    for dire in ["top", "right"]:
        ax.spines[dire].set_color("none")

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    f_ts = np.linspace(0, 1, len(df_avg) + 1)[1:]

    if final_error is None:
        if np.sum(df_avg_error) != 0:
            final_error = np.std(df_avg)
        else:
            final_error = 1.0

    if np.isfinite(final_error):
        ax.fill_between(
            [0, 1],
            np.mean(df_avg) - final_error,
            np.mean(df_avg) + final_error,
            color="#D2B9D3",
            zorder=1,
        )
    ax.errorbar(
        f_ts,
        df_avg,
        yerr=df_avg_error,
        color="#736AFF",
        lw=3,
        zorder=2,
        marker="o",
        mfc="w",
        mew=2.5,
        mec="#736AFF",
        ms=12,
        label="Avg FE",
    )

    xticks_spacing = len(f_ts) // 10 or 1
    xticks = f_ts[::xticks_spacing]
    plt.xticks(xticks, [f"{i:.2f}" for i in xticks], fontsize=10)
    plt.yticks(fontsize=10)

    ax.legend(
        loc="best",
        prop=FP(size=18),
        frameon=False,
    )
    ax.set_xlabel(r"Fraction of the Simulation Time", fontsize=16, color="#151B54")
    ax.set_ylabel(r"$\Delta G$ ({})".format(units), fontsize=16, color="#151B54")
    plt.tick_params(axis="x", color="#D2B9D3")
    plt.tick_params(axis="y", color="#D2B9D3")
    plt.tight_layout()
    return ax
