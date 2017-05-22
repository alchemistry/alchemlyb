.. _estimatators_TI:

TI
==
The :class:`~alchemlyb.estimators.TI` estimator is a simple implementation of `thermodynamic integration <https://en.wikipedia.org/wiki/Thermodynamic_integration>`_ that uses the trapezoid rule for integrating the space between :math:`\left<\frac{dH}{d\lambda}\right>` values for each :math:`\lambda` sampled.

API Reference
-------------
.. autoclass:: alchemlyb.estimators.TI
    :members:
    :inherited-members:
