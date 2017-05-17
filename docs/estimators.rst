.. _estimators:

Using estimators to obtain free energies
========================================
Calculating free energy differences from raw alchemical data requires the use of some *estimator*.
All estimators in **alchemlyb** conform to a common design pattern, with a form similar to that of estimators found in `scikit-learn <http://scikit-learn.org>`_.
If you have familiarity with the usage of estimators in **scikit-learn**, then working with estimators in **alchemlyb** should be somewhat straightforward.

**alchemlyb** provides implementations of many commonly-used estimators, which come in two varieties: TI-based and FEP-based.

.. toctree::
    :maxdepth: 2

    estimators-ti
    estimators-fep
