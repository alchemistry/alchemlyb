Visualisation of the results
============================
It is quite often that the user want to visualise the results to gain
confidence on the computed free energy. **alchemlyb** provides various
visualisation tools to help user to judge the estimate.

.. _plot_overlap_matrix:
Overlap Matrix of the MBAR
--------------------------
The accuracy of the :class:`~alchemlyb.estimators.MBAR` estimator depends on
the overlap between different lambda states. The overlap matrix from the
:class:`~alchemlyb.estimators.MBAR` estimator could be plotted to check
the degree of overlap. It is recommended that there should be at least
**0.05** overlap between neighboring states. ::

    >>> import pandas as pd
    >>> from alchemtest.gmx import load_benzene
    >>> from alchemlyb.parsing.gmx import extract_u_nk
    >>> from alchemlyb.estimators import MBAR

    >>> bz = load_benzene().data
    >>> u_nk_coul = pd.concat([extract_u_nk(xvg, T=300) for xvg in bz['Coulomb']])
    >>> mbar_coul = MBAR()
    >>> mbar_coul.fit(u_nk_coul)

    >>> from alchemlyb.visualisation.mbar_martix import plot_mbar_omatrix
    >>> ax = plot_mbar_omatrix(mbar_coul.overlap_maxtrix)
    >>> ax.figure.savefig('O_MBAR.pdf', bbox_inches='tight', pad_inches=0.0)
