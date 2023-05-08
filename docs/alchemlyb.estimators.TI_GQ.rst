.. _estimatators_TI_GQ:

TI_GQ
==
The :class:`~alchemlyb.estimators.TI_GQ` estimator is a simple implementation of `thermodynamic integration <https://en.wikipedia.org/wiki/Thermodynamic_integration>`_ that uses the `gaussian quadrature <https://en.wikipedia.org/wiki/Gaussian_quadrature>` for integrating the space between :math:`\left<\frac{dH}{d\lambda}\right>` values for each :math:`\lambda` sampled.
To use this method, make sure that the simulations are performed at certain :math:`\lambda` values using fixed gaussian quadrature points (e.g., [He2020]_). Otherwise, there would be associated errors.

API Reference
-------------
.. autoclass:: alchemlyb.estimators.TI_GQ
    :members:
    :inherited-members:
