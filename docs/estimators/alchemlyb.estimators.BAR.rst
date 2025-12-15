.. _estimators_BAR:

BAR
===
The :class:`~alchemlyb.estimators.BAR` estimator is a light wrapper around the implementation of the Bennett Acceptance Ratio (BAR) method [Bennett1976]_ from :mod:`pymbar` (:func:`pymbar.other_estimators.bar`).
It uses information from neighboring sampled states to generate an estimate for the free energy difference between these state.

.. SeeAlso::
   :class:`alchemlyb.estimators.MBAR`

API Reference
-------------
.. autoclass:: alchemlyb.estimators.BAR
    :members:
    :inherited-members:
