.. _subsampling:

Subsampling
===========

.. automodule:: alchemlyb.preprocessing.subsampling

The functions featured in this module can be used to easily subsample either
:ref:`dHdl <dHdl>` or :ref:`u_nk <u_nk>` datasets to give less correlated
timeseries.

High-level functions
--------------------
Two high-level functions
:func:`~alchemlyb.preprocessing.subsampling.decorrelate_u_nk` and
:func:`~alchemlyb.preprocessing.subsampling.decorrelate_dhdl` can be used to
preprocess the :ref:`dHdl <dHdl>` or :ref:`u_nk <u_nk>` in an automatic
fashion. The following code could remove the initial burnin period and
decorrelate the data. ::

    >>> from alchemlyb.parsing.gmx import extract_u_nk, extract_dHdl
    >>> from alchemlyb.preprocessing.subsampling import (decorrelate_u_nk,
    >>>     decorrelate_dhdl)
    >>> bz = load_benzene().data
    >>> u_nk = extract_u_nk(bz['Coulomb'], T=300)
    >>> decorrelated_u_nk = decorrelate_u_nk(u_nk, method='dhdl',
    >>>     remove_burnin=True)
    >>> dhdl = extract_dHdl(bz['Coulomb'], T=300)
    >>> decorrelated_dhdl = decorrelate_dhdl(dhdl, remove_burnin=True)

Low-level functions
--------------------
To decorrelate the data, in addition to the dataframe that contains the
:ref:`dHdl <dHdl>` or :ref:`u_nk <u_nk>`, a :mod:`pandas.Series` is needed for
the autocorrection analysis. The series could be generated with
:func:`~alchemlyb.preprocessing.subsampling.u_nk2series` or
:func:`~alchemlyb.preprocessing.subsampling.dhdl2series` and feed into
:func:`~alchemlyb.preprocessing.subsampling.statistical_inefficiency` or
:func:`~alchemlyb.preprocessing.subsampling.equilibrium_detection`. ::

    >>> from alchemlyb.parsing.gmx import extract_u_nk, extract_dHdl
    >>> from alchemlyb.preprocessing.subsampling import (u_nk2series,
    >>>     dhdl2series, statistical_inefficiency, equilibrium_detection)
    >>> bz = load_benzene().data
    >>> u_nk = extract_u_nk(bz['Coulomb'], T=300)
    >>> u_nk_series = u_nk2series(u_nk, method='dhdl')
    >>> decorrelate_u_nk = statistical_inefficiency(u_nk, series=u_nk_series)
    >>> decorrelate_u_nk = equilibrium_detection(u_nk, series=u_nk_series)
    >>> dhdl = extract_dHdl(bz['Coulomb'], T=300)
    >>> dhdl_series = dhdl2series(dhdl)
    >>> decorrelate_dhdl = statistical_inefficiency(dhdl, series=dhdl_series)
    >>> decorrelate_dhdl = equilibrium_detection(dhdl, series=dhdl_series)

API Reference
-------------
.. autofunction:: alchemlyb.preprocessing.subsampling.decorrelate_u_nk
.. autofunction:: alchemlyb.preprocessing.subsampling.decorrelate_dhdl
.. autofunction:: alchemlyb.preprocessing.subsampling.u_nk2series
.. autofunction:: alchemlyb.preprocessing.subsampling.dhdl2series
.. autofunction:: alchemlyb.preprocessing.subsampling.slicing
.. autofunction:: alchemlyb.preprocessing.subsampling.statistical_inefficiency
.. autofunction:: alchemlyb.preprocessing.subsampling.equilibrium_detection
