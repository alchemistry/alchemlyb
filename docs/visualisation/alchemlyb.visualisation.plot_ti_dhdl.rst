.. _visualisation_plot_mbar_plot_ti_dhdl:

Plot dhdl from TI
=================

The function :func:`~alchemlyb.visualisation.plot_ti_dhdl` allows
the user to plot the dhdl from :class:`~alchemlyb.estimators.TI` estimator.
Several :class:`~alchemlyb.estimators.TI` estimators could be passed to the
function to give a concerted picture of the whole alchemical transformation.
When custom labels are desirable, the user could pass a list of strings to the
*labels* for labelling each alchemical transformation differently. The color of
each alchemical transformation could also be set by passing a list of color
string to the *colors*. The unit in the y axis could be labelled to other units
by setting *units*, which by default is :math:`kT`. The user can pass
:class:`matplotlib.axes.Axes` into the function to have the dhdl drawn on a
specific axes.

Please check :ref:`How to plot TI dhdl <plot_TI_dhdl>` for usage.

API Reference
-------------
.. autofunction:: alchemlyb.visualisation.plot_ti_dhdl