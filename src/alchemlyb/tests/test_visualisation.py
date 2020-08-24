
def test_plot_mbar_omatrix():
    '''Just test if the plot runs'''
    import pandas as pd
    from alchemtest.gmx import load_benzene
    from alchemlyb.parsing.gmx import extract_u_nk
    from alchemlyb.estimators import MBAR

    bz = load_benzene().data
    u_nk_coul = pd.concat([extract_u_nk(xvg, T=300) for xvg in bz['Coulomb']])
    mbar_coul = MBAR()
    mbar_coul.fit(u_nk_coul)

    from alchemlyb.visualisation.mbar_martix import plot_mbar_omatrix
    plot_mbar_omatrix(mbar_coul.overlap_maxtrix)
    plot_mbar_omatrix(mbar_coul.overlap_maxtrix, [1,])

    # Bump up coverage
    overlap_maxtrix = mbar_coul.overlap_maxtrix
    overlap_maxtrix[0,0] = 0.0025
    overlap_maxtrix[-1, -1] = 0.9975
    plot_mbar_omatrix(overlap_maxtrix)

