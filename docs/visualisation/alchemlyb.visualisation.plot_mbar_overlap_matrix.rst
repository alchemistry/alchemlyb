.. _visualisation_plot_mbar_overlap_matrix:

Plot Overlap Matrix from MBAR
=============================

The function :func:`~alchemlyb.visualisation.plot_mbar_overlap_matrix` allows
the user to plot the overlap matrix from
:attr:`~alchemlyb.estimators.MBAR.overlap_matrix`. The user can pass
:class:`matplotlib.axes.Axes` into the function to have the overlap maxtrix
drawn on a specific axes. The user could also specify a list of lambda states
to be skipped when labelling the states.

Please check :ref:`How to plot MBAR overlap matrix <plot_overlap_matrix>` for
usage.

API Reference
-------------
.. autofunction:: alchemlyb.visualisation.plot_mbar_overlap_matrix