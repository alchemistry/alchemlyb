"""Functions for Plotting the dhdl for the TI estimator.

To assess the quality of the TI estimator, the dhdl from lambda state 0
to lambda state 1 can plotted to assess if the change in dhdl is sampled
thoroughly.

The code for producing the dhdl plot is modified based on
`Alchemical Analysis <https://github.com/MobleyLab/alchemical-analysis>`_.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties as FP

from ..postprocessors.units import get_unit_converter


def plot_ti_dhdl(dhdl_data, labels=None, colors=None, units=None, ax=None):
    """Plot the dhdl of TI.

    Parameters
    ----------
    dhdl_data : :class:`~alchemlyb.estimators.TI` or list
        One or more :class:`~alchemlyb.estimators.TI` estimator, where the
        dhdl value will be taken from.
    labels : List
        list of labels for labelling all the alchemical transformations.
    colors : List
        list of colors for plotting all the alchemical transformations.
        Default: ['r', 'g', '#7F38EC', '#9F000F', 'b', 'y']
    units : str
        The unit of the estimate. The default is `None`, which is to use the
        unit in the input. Setting this will change the output unit.
    ax : matplotlib.axes.Axes
        Matplotlib axes object where the plot will be drawn on. If ``ax=None``,
        a new axes will be generated.

    Returns
    -------
    matplotlib.axes.Axes
        An axes with the TI dhdl drawn.

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
    # Make it into a list
    # separate_dhdl method is used so that the input for the actual plotting
    # Function are a uniformed list of series object which only contains one
    # lambda.
    if not isinstance(dhdl_data, list):
        dhdl_list = dhdl_data.separate_dhdl()
    else:
        dhdl_list = []
        for dhdl in dhdl_data:
            dhdl_list.extend(dhdl.separate_dhdl())

    # Convert unit
    if units is None:
        units = dhdl_list[0].attrs["energy_unit"]

    new_unit = []
    convert = get_unit_converter(units)
    for dhdl in dhdl_list:
        new_unit.append(convert(dhdl))

    dhdl_list = new_unit
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    for k, spine in ax.spines.items():
        spine.set_zorder(12.2)

    # Make level names
    if labels is None:
        lv_names2 = []
        for dhdl in dhdl_list:
            # Assume that the dhdl has only one columns
            lv_names2.append(dhdl.name.capitalize())
    else:
        if len(labels) == len(dhdl_list):
            lv_names2 = labels
        else:  # pragma: no cover
            raise ValueError(
                "Length of labels ({}) should be the same as the number of data ({})".format(
                    len(labels), len(dhdl_list)
                )
            )

    if colors is None:
        colors = ["r", "g", "#7F38EC", "#9F000F", "b", "y"]
    else:
        if len(colors) >= len(dhdl_list):
            pass
        else:  # pragma: no cover
            raise ValueError(
                "Number of colors ({}) should be larger than the number of data ({})".format(
                    len(labels), len(dhdl_list)
                )
            )

    # Get the real data out
    xs, ndx, _dx = [0], 0, 0.001
    min_y, max_y = 0, 0
    for dhdl in dhdl_list:
        x = dhdl.index.values
        y = dhdl.values.ravel()

        min_y = min(y.min(), min_y)
        max_y = max(y.max(), max_y)

        for i in range(len(x) - 1):
            if i % 2 == 0:
                ax.fill_between(
                    x[i : i + 2] + ndx, 0, y[i : i + 2], color=colors[ndx], alpha=1.0
                )
            else:
                ax.fill_between(
                    x[i : i + 2] + ndx, 0, y[i : i + 2], color=colors[ndx], alpha=0.5
                )

        xlegend = [-100 * wnum for wnum in range(len(lv_names2))]
        ax.plot(
            xlegend,
            [0 * wnum for wnum in xlegend],
            ls="-",
            color=colors[ndx],
            label=lv_names2[ndx],
        )
        xs += (x + ndx).tolist()[1:]
        ndx += 1

    # Make sure the tick labels are not overcrowded.
    xs = np.array(xs)
    dl_mat = np.array([xs - i for i in xs])
    ri = range(len(xs))

    def getInd(r=ri, z=[0]):
        primo = r[0]
        min_dl = ndx * 0.02 * 2 ** (primo > 10)
        if dl_mat[primo].max() < min_dl:
            return z
        for i in r:  # pragma: no cover
            for j in range(len(xs)):
                if dl_mat[i, j] > min_dl:
                    z.append(j)
                    return getInd(ri[j:], z)

    xt = []
    for i in range(len(xs)):
        if i in getInd():
            xt.append(i)
        else:
            xt.append("")

    plt.xticks(xs[1:], xt[1:], fontsize=10)
    ax.yaxis.label.set_size(10)

    # Remove the abscissa ticks and set up the axes limits.
    for tick in ax.get_xticklines():
        tick.set_visible(False)
    ax.set_xlim(0, ndx)
    min_y *= 1.01
    max_y *= 1.01

    # Modified so that the x label won't conflict with the lambda label
    min_y -= (max_y - min_y) * 0.1

    ax.set_ylim(min_y, max_y)

    for i, j in zip(xs[1:], xt[1:]):
        ax.annotate(
            ("{:.2f}".format(i - 1.0 if i > 1.0 else i) if not j == "" else ""),
            xy=(i, 0),
            size=10,
            rotation=90,
            va="bottom",
            ha="center",
            color="#151B54",
        )
    if ndx > 1:
        lenticks = len(ax.get_ymajorticklabels()) - 1
        if min_y < 0:
            lenticks -= 1
        if lenticks < 5:  # pragma: no cover
            from matplotlib.ticker import AutoMinorLocator as AML

            ax.yaxis.set_minor_locator(AML())
    ax.grid(which="both", color="w", lw=0.25, axis="y", zorder=12)
    ax.set_ylabel(
        r"$\langle{\frac{\partial U}{\partial\lambda}}\rangle_{\lambda}$"
        + "({})".format(units),
        fontsize=20,
        color="#151B54",
    )
    ax.annotate(
        r"$\mathit{\lambda}$",
        xy=(0, 0),
        xytext=(0.5, -0.05),
        size=18,
        textcoords="axes fraction",
        va="top",
        ha="center",
        color="#151B54",
    )
    lege = ax.legend(prop=FP(size=14), frameon=False, loc=1)
    for legend_handle in lege.legend_handles:
        legend_handle.set_linewidth(10)
    return ax
