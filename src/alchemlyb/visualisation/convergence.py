import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as FP
import numpy as np

def plot_convergence(forward, forward_error, backward, backward_error,
                     units='kBT', ax=None):
    """Plots the free energy change computed using the equilibrated snapshots between the proper target time frames (f_ts and r_ts)
    in both forward (data points are stored in F_df and F_ddf) and reverse (data points are stored in R_df and R_ddf) directions."""
    if ax is None:
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
                        lw=3, solid_capstyle='round', zorder=2, marker='o',
                        mfc='w', mew=2.5, mec='#736AFF', ms=12,)
    line2 = ax.errorbar(r_ts, backward, yerr=backward_error, color='#C11B17',
                        lw=3, solid_capstyle='round', zorder=3, marker='o',
                        mfc='w', mew=2.5, mec='#C11B17', ms=12, )

    # ax.set_xlim(0,0.5)

    plt.xticks(r_ts[::2], fontsize=10)
    plt.yticks(fontsize=10)

    leg = plt.legend((line1[0], line2[0]), (r'$Forward$', r'$Reverse$'), loc=9,
                    prop=FP(size=18), frameon=False)
    plt.xlabel(r'$\mathrm{Fraction\/of\/the\/simulation\/time}$', fontsize=16,
              color='#151B54')
    plt.ylabel(r'$\mathrm{\Delta G\/%s}$' % units, fontsize=16,
              color='#151B54')
    plt.xticks(f_ts, ['%.2f' % i for i in f_ts])
    plt.tick_params(axis='x', color='#D2B9D3')
    plt.tick_params(axis='y', color='#D2B9D3')
    return ax


