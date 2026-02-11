.. _estimators_TI_GQ:

TI_GQ
=====
The :class:`~alchemlyb.estimators.TI_GQ` estimator is an implementation of `thermodynamic integration <https://en.wikipedia.org/wiki/Thermodynamic_integration>`_ that uses the `gaussian quadrature <https://en.wikipedia.org/wiki/Gaussian_quadrature>`_ for integrating the space between :math:`\left<\frac{dH}{d\lambda}\right>` values for each :math:`\lambda` sampled.
To use this method, please make sure that the simulations are performed at certain :math:`\lambda` values using fixed gaussian quadrature points (e.g., [He2020]_). Currently, up to 16 gaussian quadrature points are supported (see the table below).

.. list-table:: gaussian quadrature points
    :widths: 5 30
    :header-rows: 1

    * - number of :math:`\lambda`
      - :math:`\lambda` values
    * - 1
      - 0.5
    * - 2
      - 0.21132, 0.78867
    * - 3
      - 0.1127, 0.5, 0.88729
    * - 4
      - 0.06943, 0.33001, 0.66999, 0.93057
    * - 5
      - 0.04691, 0.23076, 0.5, 0.76923, 0.95308
    * - 6
      - 0.03377, 0.1694 , 0.38069, 0.61931, 0.8306 , 0.96623
    * - 7
      - 0.02544, 0.12923, 0.29707, 0.5, 0.70292, 0.87076, 0.97455
    * - 8
      - 0.01986, 0.10167, 0.23723, 0.40828, 0.59172, 0.76277, 0.89833, 0.98014
    * - 9
      - 0.01592, 0.08198, 0.19331, 0.33787, 0.5, 0.66213, 0.80669, 0.91802, 0.98408
    * - 10
      - 0.01305, 0.06747, 0.1603, 0.2833, 0.42556, 0.57444, 0.7167, 0.8397, 0.93253, 0.98695
    * - 11
      - 0.01089, 0.05647, 0.13492, 0.24045, 0.36523, 0.5, 0.63477, 0.75955, 0.86508, 0.94353, 0.98911
    * - 12
      - 0.00922, 0.04794, 0.11505, 0.20634, 0.31608, 0.43738, 0.56262, 0.68392, 0.79366, 0.88495, 0.95206, 0.99078
    * - 13
      - 0.00791, 0.0412, 0.09921, 0.17883, 0.27575, 0.38477, 0.5, 0.61523, 0.72425, 0.82117, 0.90079, 0.9588, 0.99209
    * - 14
      - 0.00686, 0.03578, 0.0864, 0.15635, 0.24238, 0.34044, 0.44597, 0.55403, 0.65956, 0.75762, 0.84365, 0.9136, 0.96422, 0.99314
    * - 15
      - 0.006, 0.03136, 0.0759, 0.13779, 0.21451, 0.30292, 0.3994 , 0.5, 0.6006, 0.69708, 0.78549, 0.86221, 0.9241, 0.96864, 0.994
    * - 16
      - 0.0053, 0.02771, 0.06718, 0.1223, 0.19106, 0.27099, 0.3592, 0.45249, 0.54751, 0.6408, 0.72901, 0.80894, 0.8777, 0.93282, 0.97229, 0.9947

API Reference
-------------
.. autoclass:: alchemlyb.estimators.TI_GQ
    :members:
    :inherited-members:
