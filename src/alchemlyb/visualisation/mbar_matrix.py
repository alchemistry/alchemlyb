"""Functions for Plotting the overlay matrix for the MBAR estimator.

To assess the quality of the MBAR estimator, the overlap matrix between
the lambda states can be computed and the more overlap is observed between
the states, the more reliable the estimate is. One way of accessing the
quality of the overlap matrix is by plotting it.

The code for producing the overlap matrix plot is taken from
`Alchemical Analysis <https://github.com/MobleyLab/alchemical-analysis>`_.

"""

import matplotlib.pyplot as plt
import numpy as np


def plot_mbar_overlap_matrix(matrix, skip_lambda_index=[], ax=None):
    """Plot the MBAR overlap matrix.

    Parameters
    ----------
    matrix : numpy.matrix
        DataFrame of the overlap matrix obtained from
        :attr:`~alchemlyb.estimators.MBAR.overlap_matrix`
    skip_lambda_index : List
        list of lambda indices to be omitted from plotting process.
        Default: ``[]``.
    ax : matplotlib.axes.Axes
        Matplotlib axes object where the plot will be drawn on. If ``ax=None``,
        a new axes will be generated.

    Returns
    -------
    matplotlib.axes.Axes
        An axes with the overlap matrix drawn.

    Note
    ----
    The code is taken and modified from
    `Alchemical Analysis <https://github.com/MobleyLab/alchemical-analysis>`_.


    .. versionadded:: 0.4.0

    """
    # Compute the size of the figure, if ax is not given.
    max_prob = matrix.max()
    size = len(matrix)
    if ax is None:
        fig, ax = plt.subplots(figsize=(size / 2, size / 2))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    for i in range(size):
        if i != 0:
            ax.axvline(x=i, ls="-", lw=0.5, color="k", alpha=0.25)
            ax.axhline(y=i, ls="-", lw=0.5, color="k", alpha=0.25)
        for j in range(size):
            if matrix[j, i] < 0.005:
                ii = ""
            elif matrix[j, i] > 0.995:
                ii = "1.00"
            else:
                ii = "{:.2f}".format(matrix[j, i])[1:]
            alf = matrix[j, i] / max_prob
            ax.fill_between(
                [i, i + 1],
                [size - j, size - j],
                [size - (j + 1), size - (j + 1)],
                color="k",
                alpha=alf,
            )
            ax.annotate(
                ii,
                xy=(i, j),
                xytext=(i + 0.5, size - (j + 0.5)),
                size=8,
                textcoords="data",
                va="center",
                ha="center",
                color=("k" if alf < 0.5 else "w"),
            )

    if skip_lambda_index:
        ks = [int(l) for l in skip_lambda_index]
        ks = np.delete(np.arange(size + len(ks)), ks)
    else:
        ks = range(size)
    for i in range(size):
        ax.annotate(
            ks[i],
            xy=(i + 0.5, 1),
            xytext=(i + 0.5, size + 0.5),
            size=10,
            textcoords=("data", "data"),
            va="center",
            ha="center",
            color="k",
        )
        ax.annotate(
            ks[i],
            xy=(-0.5, size - (size - 0.5)),
            xytext=(-0.5, size - (i + 0.5)),
            size=10,
            textcoords=("data", "data"),
            va="center",
            ha="center",
            color="k",
        )
    ax.annotate(
        r"$\lambda$",
        xy=(-0.5, size - (size - 0.5)),
        xytext=(-0.5, size + 0.5),
        size=10,
        textcoords=("data", "data"),
        va="center",
        ha="center",
        color="k",
    )
    ax.plot([0, size], [0, 0], "k-", lw=4.0, solid_capstyle="butt")
    ax.plot([size, size], [0, size], "k-", lw=4.0, solid_capstyle="butt")
    ax.plot([0, 0], [0, size], "k-", lw=2.0, solid_capstyle="butt")
    ax.plot([0, size], [size, size], "k-", lw=2.0, solid_capstyle="butt")

    cx = np.repeat(range(size + 1), 2)
    cy = sorted(np.repeat(range(size + 1), 2), reverse=True)
    ax.plot(cx[2:-1], cy[1:-2], "k-", lw=2.0)
    ax.plot(np.array(cx[2:-3]) + 1, cy[1:-4], "k-", lw=2.0)
    ax.plot(cx[1:-2], np.array(cy[:-3]) - 1, "k-", lw=2.0)
    ax.plot(cx[1:-4], np.array(cy[:-5]) - 2, "k-", lw=2.0)

    ax.set_xlim(-1, size)
    ax.set_ylim(0, size + 1)
    return ax
