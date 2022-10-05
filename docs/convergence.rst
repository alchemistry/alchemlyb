.. module:: alchemlyb.convergence

Using functions to estimate Convergence
=======================================

For a result to be valid, we need to ensure that longer simulation time
would not result in different results. Various functions will be provided in
this module to estimate the convergence of the estimate and help user determine
the simulation end point.

Time Convergence
----------------
One way of determining the simulation end point is to compute and plot the
forward and backward convergence of the estimate using
:func:`~alchemlyb.convergence.forward_backward_convergence` and
:func:`~alchemlyb.visualisation.plot_convergence`. ::

    >>> from alchemtest.gmx import load_benzene
    >>> from alchemlyb.parsing.gmx import extract_u_nk
    >>> from alchemlyb.visualisation import plot_convergence
    >>> from alchemlyb.convergence import forward_backward_convergence

    >>> bz = load_benzene().data
    >>> data_list = [extract_u_nk(xvg, T=300) for xvg in bz['Coulomb']]
    >>> df = forward_backward_convergence(data_list, 'mbar')
    >>> ax = plot_convergence(df)
    >>> ax.figure.savefig('dF_t.pdf')

Will give a plot looks like this

.. figure:: images/dF_t.png

   A convergence plot of showing that the forward and backward has converged
   fully.

Fraction Convergence
--------------------

Another way of assessing whether the simulation has converged is to check the
energy files. In [Fan2021]_, :func:`~alchemlyb.convergence.R_c` and
:func:`~alchemlyb.convergence.A_c` are two criteria of checking the
convergence. :func:`~alchemlyb.convergence.R_c` takes a decorrelated
:class:`pandas.Series` as input and gives the metric
:func:`~alchemlyb.convergence.R_c`, which is 0 for fully-equilibrated
simulation and 1 for fully-unequilibrated simulation. ::

    >>> from alchemtest.gmx import load_ABFE
    >>> from alchemlyb.parsing.gmx import extract_dHdl
    >>> from alchemlyb.preprocessing import decorrelate_dhdl, dhdl2series
    >>> from alchemlyb.convergence import R_c
    >>> from alchemlyb.visualisation import plot_convergence

    >>> file = load_ABFE().data['ligand'][0]
    >>> dhdl = extract_dHdl(file, T=300)
    >>> decorrelated = decorrelate_dhdl(dhdl, remove_burnin=True)
    >>> value, running_average = R_c(dhdl2series(decorrelated), tol=2)
    >>> print(value)
    0.02
    >>> plot_convergence(running_average, final_error=2, units='kcal/mol')


Will give a plot like this.

.. figure:: images/R_c.png


The :func:`~alchemlyb.convergence.A_c` on the other hand, takes in a list of
decorrelated :class:`pandas.Series` and gives a metric of how converged the set
is, where 0 fully-unequilibrated and 1.0 is fully-equilibrated. ::

    >>> from alchemlyb.convergence import A_c
    >>> dhdl_list = []
    >>> for file in load_ABFE().data['ligand']:
    >>>     dhdl = extract_dHdl(file, T=300)
    >>>     decorrelated = decorrelate_dhdl(dhdl, remove_burnin=True)
    >>>     decorrelated = dhdl2series(decorrelated)
    >>>     dhdl_list.append(decorrelated)
    >>> value = A_c(dhdl_list)
    0.7085


Convergence functions
---------------------

The currently available connvergence functions:

.. currentmodule:: alchemlyb.convergence

.. autosummary::
    :toctree: convergence

    convergence
    R_c
    A_c

References
----------

.. [Fan2021] Fan, S., Nedev, H., Vijayan, R., Iorga, B.I., and Beckstein, O.
    (2021). Precise force-field-based calculations of octanol-water partition
    coefficients for the SAMPL7 molecules. Journal of Computer-Aided Molecular
    Design 35, 853–87
