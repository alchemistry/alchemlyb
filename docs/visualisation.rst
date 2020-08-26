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
:class:`~alchemlyb.estimators.MBAR` estimator could be plotted using
:func:`~alchemlyb.visualisation.plot_mbar_overlap_matrix` to check
the degree of overlap. It is recommended that there should be at least
**0.03** [Klimovich2015]_ overlap between neighboring states. ::

    >>> import pandas as pd
    >>> from alchemtest.gmx import load_benzene
    >>> from alchemlyb.parsing.gmx import extract_u_nk
    >>> from alchemlyb.estimators import MBAR

    >>> bz = load_benzene().data
    >>> u_nk_coul = pd.concat([extract_u_nk(xvg, T=300) for xvg in bz['Coulomb']])
    >>> mbar_coul = MBAR()
    >>> mbar_coul.fit(u_nk_coul)

    >>> from alchemlyb.visualisation import plot_mbar_overlap_matrix
    >>> ax = plot_mbar_overlap_matrix(mbar_coul.overlap_matrix)
    >>> ax.figure.savefig('O_MBAR.pdf', bbox_inches='tight', pad_inches=0.0)


Will give a plot looks like this

.. image:: images/O_MBAR.png

.. [Klimovich2015] Klimovich, P.V., Shirts, M.R. & Mobley, D.L. Guidelines for
   the analysis of free energy calculations. J Comput Aided Mol Des 29, 397–411
   (2015). https://doi.org/10.1007/s10822-015-9840-9
