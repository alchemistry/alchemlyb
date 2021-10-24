.. _estimators_MBAR:

MBAR
====
The :class:`~alchemlyb.estimators.MBAR` estimator is a light wrapper around the reference implementation of MBAR from :mod:`pymbar` (:class:`pymbar.mbar.MBAR`).
As a generalization of BAR, it uses information from all sampled states to generate an estimate for the free energy difference between each state.

A more robust version of :class:`~alchemlyb.estimators.MBAR` is provided as
:class:`~alchemlyb.estimators.AutoMBAR`, where the class will iteratively
try different means of solving the MBAR estimate to avoid unconverged results.

API Reference
-------------
.. autoclass:: alchemlyb.estimators.MBAR
    :members:
    :inherited-members:

.. autoclass:: alchemlyb.estimators.AutoMBAR
