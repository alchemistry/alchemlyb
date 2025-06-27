"""Functions for Plotting the dF states.

To assess the quality of the free energy estimation, The dF between adjacent
lambda states can be plotted to assess the quality of the estimation.

The code for producing the dF states plot is modified based on
`Alchemical Analysis <https://github.com/MobleyLab/alchemical-analysis>`_.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties as FP

from ..estimators import TI, BAR, MBAR
from ..postprocessors.units import get_unit_converter


def plot_dF_state(
    estimators, labels=None, colors=None, units=None, orientation="portrait", nb=10
):
    """Plot the dhdl of TI.

    Parameters
    ----------
    estimators : :class:`~alchemlyb.estimators` or list
        One or more :class:`~alchemlyb.estimators`, where the
        dhdl value will be taken from. For more than one estimators
        with more than one alchemical transformation, a list of list format
        is used.
    labels : List
        list of labels for labelling different estimators.
    colors : List
        list of colors for plotting different estimators.
    units : str
        The unit of the estimate. The default is `None`, which is to use the
        unit in the input. Setting this will change the output unit.
    orientation : string
        The orientation of the figure. Can be `portrait` or `landscape`
    nb : int
        Maximum number of dF states in one row in the `portrait` mode

    Returns
    -------
    matplotlib.figure.Figure
        An Figure with the dF states drawn.

    Note
    ----
    The code is taken and modified from
    `Alchemical Analysis <https://github.com/MobleyLab/alchemical-analysis>`_.


    .. versionchanged:: 1.0.0
        If no units is given, the `units` in the input will be used.

    .. versionchanged:: 0.5.0
        The `units` will be used to change the underlying data instead of only
        changing the figure legend.

    .. versionadded:: 0.4.0
    """
    try:
        len(estimators)
    except TypeError:
        estimators = [
            estimators,
        ]

    formatted_data = []
    for dhdl in estimators:
        try:
            len(dhdl)
            formatted_data.append(dhdl)
        except TypeError:
            formatted_data.append(
                [
                    dhdl,
                ]
            )

    if units is None:
        units = formatted_data[0][0].delta_f_.attrs["energy_unit"]

    estimators = formatted_data

    # Get the dF
    dF_list = []
    error_list = []
    max_length = 0
    convert = get_unit_converter(units)
    for dhdl_list in estimators:
        len_dF = sum([len(dhdl.delta_f_) - 1 for dhdl in dhdl_list])
        if len_dF > max_length:
            max_length = len_dF
        dF = []
        error = []
        for dhdl in dhdl_list:
            for i in range(len(dhdl.delta_f_) - 1):
                dF.append(convert(dhdl.delta_f_).iloc[i, i + 1])
                error.append(convert(dhdl.d_delta_f_).iloc[i, i + 1])

        dF_list.append(dF)
        error_list.append(error)

    # Get the determine orientation
    if orientation == "landscape":
        if max_length < 8:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig, ax = plt.subplots(figsize=(max_length, 6))
        axs = [
            ax,
        ]
        xs = [
            np.arange(max_length),
        ]
    elif orientation == "portrait":
        if max_length < nb:
            xs = [
                np.arange(max_length),
            ]
            fig, ax = plt.subplots(figsize=(8, 6))
            axs = [
                ax,
            ]
        else:
            xs = np.array_split(np.arange(max_length), max_length / nb + 1)
            fig, axs = plt.subplots(nrows=len(xs), figsize=(8, 6))
        mnb = max([len(i) for i in xs])
    else:
        raise ValueError(
            "Not recognising {}, only supports 'landscape' or 'portrait'.".format(
                orientation
            )
        )

    # Sort out the colors
    if colors is None:
        colors_dict = {
            "TI": "#C45AEC",
            "TI-CUBIC": "#33CC33",
            "DEXP": "#F87431",
            "IEXP": "#FF3030",
            "GINS": "#EAC117",
            "GDEL": "#347235",
            "BAR": "#6698FF",
            "UBAR": "#817339",
            "RBAR": "#C11B17",
            "MBAR": "#F9B7FF",
        }
        colors = []
        for dhdl in estimators:
            dhdl = dhdl[0]
            if isinstance(dhdl, TI):
                colors.append(colors_dict["TI"])
            elif isinstance(dhdl, BAR):
                colors.append(colors_dict["BAR"])
            elif isinstance(dhdl, MBAR):
                colors.append(colors_dict["MBAR"])
    else:
        if len(colors) >= len(estimators):
            pass
        else:
            raise ValueError(
                "Number of colors ({}) should be larger than the number of data ({})".format(
                    len(colors), len(estimators)
                )
            )

    # Sort out the labels
    if labels is None:
        labels = []
        for dhdl in estimators:
            dhdl = dhdl[0]
            if isinstance(dhdl, TI):
                labels.append("TI")
            elif isinstance(dhdl, BAR):
                labels.append("BAR")
            elif isinstance(dhdl, MBAR):
                labels.append("MBAR")
    else:
        if len(labels) == len(estimators):
            pass
        else:
            raise ValueError(
                "Length of labels ({}) should be the same as the number of data ({})".format(
                    len(labels), len(estimators)
                )
            )

    # Plot the figure
    width = 1.0 / (len(estimators) + 1)
    elw = 30 * width
    ndx = 1
    for x, ax in zip(xs, axs):
        lines = []
        for i, (dF, error) in enumerate(zip(dF_list, error_list)):
            y = [dF[j] for j in x]
            ye = [error[j] for j in x]
            if orientation == "landscape":
                lw = 0.1 * elw
            elif orientation == "portrait":
                lw = 0.05 * elw
            line = ax.bar(
                x + len(lines) * width,
                y,
                width,
                color=colors[i],
                yerr=ye,
                lw=lw,
                error_kw=dict(elinewidth=elw, ecolor="black", capsize=0.5 * elw),
            )
            lines += (line[0],)
        for dir in ["left", "right", "top", "bottom"]:
            if dir == "left":
                ax.yaxis.set_ticks_position(dir)
            else:
                ax.spines[dir].set_color("none")

        if orientation == "landscape":
            plt.yticks(fontsize=8)
            ax.set_xlim(x[0] - width, x[-1] + len(lines) * width)
            plt.xticks(
                x + 0.5 * width * len(estimators),
                tuple([f"{i}--{i + 1}" for i in x]),
                fontsize=8,
            )
        elif orientation == "portrait":
            plt.yticks(fontsize=10)
            ax.xaxis.set_ticks([])
            for i in x + 0.5 * width * len(estimators):
                ax.annotate(
                    r"$\mathrm{%d-%d}$" % (i, i + 1),
                    xy=(i, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -2),
                    size=10,
                    textcoords="offset points",
                    va="top",
                    ha="center",
                )
            ax.set_xlim(x[0] - width, x[-1] + len(lines) * width + (mnb - len(x)))
        ndx += 1
    x = np.arange(max_length)

    ax = plt.gca()

    for tick in ax.get_xticklines():
        tick.set_visible(False)
    if orientation == "landscape":
        leg = plt.legend(lines, labels, loc=3, ncol=2, prop=FP(size=10), fancybox=True)
        plt.title("The free energy change breakdown", fontsize=12)
        plt.xlabel("States", fontsize=12, color="#151B54")
        plt.ylabel(r"$\Delta G$ ({})".format(units), fontsize=12, color="#151B54")
    elif orientation == "portrait":
        leg = ax.legend(
            lines,
            labels,
            loc=0,
            ncol=2,
            prop=FP(size=8),
            title=r"$\Delta G$ ({})".format(units) + r"$\mathit{vs.}$ lambda pair",
            fancybox=True,
        )

    leg.get_frame().set_alpha(0.5)
    return fig
