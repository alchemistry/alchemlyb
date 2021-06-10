.. _estimators_BAR:

BAR
===
The :class:`~alchemlyb.estimators.BAR` estimator is a light wrapper around the implementation of the Bennett Acceptance Ratio (BAR) method from :mod:`pymbar` (:class:`pymbar.mbar.BAR`).
It uses information from neighboring sampled states to generate an estimate for the free energy difference between these state.

.. SeeAlso::
   :class:`alchemlyb.estimators.MBAR`

API Reference
-------------
.. autoclass:: alchemlyb.estimators.BAR
    :members:
    :inherited-members:
