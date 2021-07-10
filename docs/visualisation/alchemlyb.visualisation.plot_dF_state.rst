.. _visualisation_plot_dF_state:

Plot dF states from multiple estimators
=======================================

The function :func:`~alchemlyb.visualisation.plot_dF_state` allows the user to
plot and compare the free energy difference between states ("dF") from various
kinds of :class:`~alchemlyb.estimators`.

To compare the dF states of a single alchemical transformation among various
:class:`~alchemlyb.estimators`, the user can pass a list of `estimators`. (e.g.
`estimators` = [:class:`~alchemlyb.estimators.TI`,
:class:`~alchemlyb.estimators.BAR`, :class:`~alchemlyb.estimators.MBAR`])

To compare the dF states of a multiple alchemical transformations, results from
the same :class:`~alchemlyb.estimators` can be concatenated into a list, which
is then bundled to to another list of different :class:`~alchemlyb.estimators`.
(e.g. `estimators` = [(:class:`~alchemlyb.estimators.TI`,
:class:`~alchemlyb.estimators.TI`), (:class:`~alchemlyb.estimators.BAR`,
:class:`~alchemlyb.estimators.BAR`), (:class:`~alchemlyb.estimators.MBAR`,
:class:`~alchemlyb.estimators.MBAR`)])

The figure could be plotted in *portrait* or *landscape* mode by setting the
`orientation`. `nb` is used to control the number of dF states in one row.
The user could pass a list of strings to `labels` to name the
:class:`~alchemlyb.estimators` or a list of strings to `colors` to color
the estimators differently. The unit in the y axis could be labelled to other
units by setting `units`, which by default is :math:`kT`.

Please check :ref:`How to plot dF states <plot_dF_states>` for a complete
example.

API Reference
-------------
.. autofunction:: alchemlyb.visualisation.plot_dF_state
