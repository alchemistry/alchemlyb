.. _estimators_fep:

FEP-based estimators
====================
FEP-based estimators such as :class:`~alchemlyb.estimators.MBAR` take as input :ref:`u_nk <u_nk>` reduced potentials for the calculation of free energy differences.
All FEP-based estimators make use of the overlap between distributions of these values for each sampled :math:`\lambda`, differing in *how* they use this overlap information to give their free energy difference estimate.

As a usage example, we'll use :class:`~alchemlyb.estimators.MBAR` to calculate the free energy of solvation of benzene in water.
We'll use the benzene-in-water dataset from :mod:`alchemtest.gmx`::

    >>> from alchemtest.gmx import load_benzene
    >>> bz = load_benzene().data

and parse the datafiles separately for each alchemical leg using :func:`alchemlyb.parsing.gmx.extract_u_nk` to obtain :ref:`u_nk <u_nk>` reduced potentials::

    >>> from alchemlyb.parsing.gmx import extract_u_nk
    >>> import pandas as pd

    >>> u_nk_coul = alchemlyb.concat([extract_u_nk(xvg, T=300) for xvg in bz['Coulomb']])
    >>> u_nk_vdw = alchemlyb.concat([extract_u_nk(xvg, T=300) for xvg in bz['VDW']])

We can now use the :class:`~alchemlyb.estimators.MBAR` estimator to obtain the free energy differences between each :math:`\lambda` window sampled.
The :meth:`~alchemlyb.estimators.MBAR.fit` method is used to perform the free energy estimate, given the gradient data::

    >>> from alchemlyb.estimators import MBAR

    >>> mbar_coul = MBAR()
    >>> mbar_coul.fit(u_nk_coul)
    MBAR(initial_f_k=None, maximum_iterations=10000, method=({'method': 'hybr'},),
       relative_tolerance=1e-07, verbose=False)

    # we could also just call the `fit` method
    # directly, since it returns the `MBAR` object
    >>> mbar_vdw = MBAR().fit(u_nk_vdw)

The sum of the endpoint free energy differences will be the free energy of solvation for benzene in water.
The free energy differences (in units of :math:`k_B T`) between each :math:`\lambda` window can be accessed via the ``delta_f_`` attribute::

    >>> mbar_coul.delta_f_
              0.00      0.25      0.50      0.75      1.00
    0.00  0.000000  1.619069  2.557990  2.986302  3.041156
    0.25 -1.619069  0.000000  0.938921  1.367232  1.422086
    0.50 -2.557990 -0.938921  0.000000  0.428311  0.483165
    0.75 -2.986302 -1.367232 -0.428311  0.000000  0.054854
    1.00 -3.041156 -1.422086 -0.483165 -0.054854  0.000000

So we can get the endpoint differences (free energy difference between :math:`\lambda = 0` and :math:`\lambda = 1`) of each with::

    >>> mbar_coul.delta_f_.loc[0.00, 1.00]
    3.0411558818767954

    >>> mbar_vdw.delta_f_.loc[0.00, 1.00]
    -3.0067874666136074

giving us a solvation free energy in units of :math:`k_B T` for benzene of::

    >>> mbar_coul.delta_f_.loc[0.00, 1.00] + mbar_vdw.delta_f_.loc[0.00, 1.00]
    0.034368415263188012

In addition to the free energy differences, we also have access to the errors on these differences via the ``d_delta_f_`` attribute::

    >>> mbar_coul.d_delta_f_
              0.00      0.25      0.50      0.75      1.00
    0.00  0.000000  0.008802  0.014432  0.018097  0.020879
    0.25  0.008802  0.000000  0.006642  0.011404  0.015143
    0.50  0.014432  0.006642  0.000000  0.005362  0.009983
    0.75  0.018097  0.011404  0.005362  0.000000  0.005133
    1.00  0.020879  0.015143  0.009983  0.005133  0.000000


List of FEP-based estimators
----------------------------

.. currentmodule:: alchemlyb.estimators

.. autosummary::
    :toctree: estimators

    MBAR
    BAR
