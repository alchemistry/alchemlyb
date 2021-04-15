import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as FP
import numpy as np

def plot_convergence(forward, forward_error, backward, backward_error,
                     units='kT', ax=None):
    """Plot the forward and backward convergence.

    Parameters
    ----------
    forward : List
        A list of free energy estimate from the first X% of data.
    forward_error : List
        A list of error from the first X% of data.
    backward : List
        A list of free energy estimate from the last X% of data.
    backward_error : List
        A list of error from the last X% of data.
    units : str
        The label for the unit of the estimate. Default: `kT`
    ax : matplotlib.axes.Axes
        Matplotlib axes object where the plot will be drawn on. If ax=None,
        a new axes will be generated.

    Returns
    -------
    matplotlib.axes.Axes
        An axes with the forward and backward convergence drawn.

    Note
    ----
    The code is taken and modified from
    : `Alchemical Analysis <https://github.com/MobleyLab/alchemical-analysis>`_

    The units variable is for labelling only. Changing it doesn't change the
    unit of the underlying variable, which is in the unit of kT. The
    scaling_factor is used to change the number to the desired unit.
    """
    if ax is None: # pragma: no cover
        fig, ax = plt.subplots(figsize=(8, 6))

    plt.setp(ax.spines['bottom'], color='#D2B9D3', lw=3, zorder=-2)
    plt.setp(ax.spines['left'], color='#D2B9D3', lw=3, zorder=-2)

    for dire in ['top', 'right']:
        ax.spines[dire].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    f_ts = np.linspace(0, 1, len(forward) + 1)[1:]
    r_ts = np.linspace(0, 1, len(backward) + 1)[1:]

    line0 = ax.fill_between([0, 1], backward[-1] - backward_error[-1],
                            backward[-1] + backward_error[-1], color='#D2B9D3',
                             zorder=1)
    line1 = ax.errorbar(f_ts, forward, yerr=forward_error, color='#736AFF',
                        lw=3, zorder=2, marker='o',
                        mfc='w', mew=2.5, mec='#736AFF', ms=12,)
    line2 = ax.errorbar(r_ts, backward, yerr=backward_error, color='#C11B17',
                        lw=3, zorder=3, marker='o',
                        mfc='w', mew=2.5, mec='#C11B17', ms=12, )

    plt.xticks(r_ts[::2], fontsize=10)
    plt.yticks(fontsize=10)

    ax.legend((line1[0], line2[0]), (r'$Forward$', r'$Reverse$'), loc=9,
                    prop=FP(size=18), frameon=False)
    ax.set_xlabel(r'Fraction of the simulation time', fontsize=16,
                  color='#151B54')
    ax.set_ylabel(r'$\Delta G$ ({})'.format(units), fontsize=16, color='#151B54')
    plt.xticks(f_ts, ['%.2f' % i for i in f_ts])
    plt.tick_params(axis='x', color='#D2B9D3')
    plt.tick_params(axis='y', color='#D2B9D3')
    return ax


