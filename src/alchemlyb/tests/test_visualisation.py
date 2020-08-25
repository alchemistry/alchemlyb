import matplotlib
import pandas as pd

from alchemtest.gmx import load_benzene
from alchemlyb.parsing.gmx import extract_u_nk
from alchemlyb.estimators import MBAR
from alchemlyb.visualisation.mbar_matrix import plot_mbar_overlap_matrix

def test_plot_mbar_omatrix():
    '''Just test if the plot runs'''
    bz = load_benzene().data
    u_nk_coul = pd.concat([extract_u_nk(xvg, T=300) for xvg in bz['Coulomb']])
    mbar_coul = MBAR()
    mbar_coul.fit(u_nk_coul)

    isinstance(plot_mbar_overlap_matrix(mbar_coul.overlap_maxtrix),
               matplotlib.axes.Axes)
    isinstance(plot_mbar_overlap_matrix(mbar_coul.overlap_maxtrix, [1,]),
               matplotlib.axes.Axes)

    # Bump up coverage
    overlap_maxtrix = mbar_coul.overlap_maxtrix
    overlap_maxtrix[0,0] = 0.0025
    overlap_maxtrix[-1, -1] = 0.9975
    isinstance(overlap_maxtrix,
               matplotlib.axes.Axes)

