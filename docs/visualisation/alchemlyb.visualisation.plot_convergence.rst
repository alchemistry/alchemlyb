.. _visualisation_plot_convergence:

Plot the Forward and Backward Convergence
=========================================

The function :func:`~alchemlyb.visualisation.plot_convergence` allows
the user to visualise the convergence by plotting the free energy change
computed using the equilibrated snapshots between the proper target time frames
in both forward (data points are stored in `forward` and `forward_error`) and
reverse (data points are stored in `backward` and `backward_error`) directions.
The unit in the y axis could be labelled to other units by setting *units*,
which by default is :math:`kT`. The user can pass :class:`matplotlib.axes.Axes` into
the function to have the convergence drawn on a specific axes.

Please check :ref:`How to plot convergence <plot_convergence>` for usage.

API Reference
-------------
.. autofunction:: alchemlyb.visualisation.plot_convergence