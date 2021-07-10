.. _estimators_ti:

TI-based estimators
===================
TI-based estimators such as :class:`~alchemlyb.estimators.TI` take as input :ref:`dHdl <dHdl>` gradients for the calculation of free energy differences.
All TI-based estimators integrate these gradients with respect to :math:`\lambda`, differing only in *how* they numerically perform this integration.

As a usage example, we'll use :class:`~alchemlyb.estimators.TI` to calculate the free energy of solvation of benzene in water.
We'll use the benzene-in-water dataset from :mod:`alchemtest.gmx`::

    >>> from alchemtest.gmx import load_benzene
    >>> bz = load_benzene().data

and parse the datafiles separately for each alchemical leg using :func:`alchemlyb.parsing.gmx.extract_dHdl` to obtain :ref:`dHdl <dHdl>` gradients::

    >>> from alchemlyb.parsing.gmx import extract_dHdl
    >>> import pandas as pd

    >>> dHdl_coul = alchemlyb.concat([extract_dHdl(xvg, T=300) for xvg in bz['Coulomb']])
    >>> dHdl_vdw = alchemlyb.concat([extract_dHdl(xvg, T=300) for xvg in bz['VDW']])

We can now use the :class:`~alchemlyb.estimators.TI` estimator to obtain the free energy differences between each :math:`\lambda` window sampled.
The :meth:`~alchemlyb.estimators.TI.fit` method is used to perform the free energy estimate, given the gradient data::

    >>> from alchemlyb.estimators import TI

    >>> ti_coul = TI()
    >>> ti_coul.fit(dHdl_coul)
    TI(verbose=False)

    # we could also just call the `fit` method
    # directly, since it returns the `TI` object
    >>> ti_vdw = TI().fit(dHdl_vdw)

The sum of the endpoint free energy differences will be the free energy of solvation for benzene in water.
The free energy differences (in units of :math:`k_B T`) between each :math:`\lambda` window can be accessed via the ``delta_f_`` attribute::

    >>> ti_coul.delta_f_
              0.00      0.25      0.50      0.75      1.00
    0.00  0.000000  1.620328  2.573337  3.022170  3.089027
    0.25 -1.620328  0.000000  0.953009  1.401842  1.468699
    0.50 -2.573337 -0.953009  0.000000  0.448832  0.515690
    0.75 -3.022170 -1.401842 -0.448832  0.000000  0.066857
    1.00 -3.089027 -1.468699 -0.515690 -0.066857  0.000000

So we can get the endpoint differences (free energy difference between :math:`\lambda = 0` and :math:`\lambda = 1`) of each with::

    >>> ti_coul.delta_f_.loc[0.00, 1.00]
    3.0890270218676896

    >>> ti_vdw.delta_f_.loc[0.00, 1.00]
    -3.0558175199846058

giving us a solvation free energy in units of :math:`k_B T` for benzene of::
    
    >>> ti_coul.delta_f_.loc[0.00, 1.00] + ti_vdw.delta_f_.loc[0.00, 1.00]
    0.033209501883083803

In addition to the free energy differences, we also have access to the errors on these differences via the ``d_delta_f_`` attribute::

    >>> ti_coul.d_delta_f_
              0.00      0.25      0.50      0.75      1.00
    0.00  0.000000  0.009706  0.013058  0.015038  0.016362
    0.25  0.009706  0.000000  0.008736  0.011486  0.013172
    0.50  0.013058  0.008736  0.000000  0.007458  0.009858
    0.75  0.015038  0.011486  0.007458  0.000000  0.006447
    1.00  0.016362  0.013172  0.009858  0.006447  0.000000


List of TI-based estimators
---------------------------

.. currentmodule:: alchemlyb.estimators

.. autosummary::
    :toctree: estimators

    TI
