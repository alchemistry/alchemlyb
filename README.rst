alchemlyb: the simple alchemistry library
=========================================

|doi| |docs| |build| |cov|

**Warning**: This library is young. It is **not** API stable. It is a
nucleation point. By all means use and help improve it, but note that it will
change with time.

**alchemlyb** is an attempt to make alchemical free energy calculations easier
to do by leveraging the full power and flexibility of the PyData stack. It
includes: 

1. Parsers for extracting raw data from output files of common molecular
   dynamics engines such as GROMACS [Abraham2015]_. 

2. Subsamplers for obtaining uncorrelated samples from timeseries data.

3. Estimators for obtaining free energies directly from this data, using
   best-practices approaches for multistate Bennett acceptance ratio (MBAR)
   [Shirts2008]_ and thermodynamic integration (TI).

In particular, it uses internally the excellent `pymbar
<http://pymbar.readthedocs.io/>`_ library for performing MBAR and extracting
independent, equilibrated samples [Chodera2016]_.

.. [Abraham2015] Abraham, M.J., Murtola, T., Schulz, R., Páll, S., Smith, J.C.,
    Hess, B., and Lindahl, E. (2015). GROMACS: High performance molecular
    simulations through multi-level parallelism from laptops to supercomputers.
    SoftwareX 1–2, 19–25.

.. [Shirts2008] Shirts, M.R., and Chodera, J.D. (2008). Statistically optimal
    analysis of samples from multiple equilibrium states. The Journal of Chemical
    Physics 129, 124105.

.. [Chodera2016] Chodera, J.D. (2016). A Simple Method for Automated
    Equilibration Detection in Molecular Simulations. Journal of Chemical Theory
    and Computation 12, 1799–1805.

.. |doi| image:: https://zenodo.org/badge/68669096.svg
    :alt: Zenodo DOI
    :scale: 100%
    :target: https://zenodo.org/badge/latestdoi/68669096

.. |docs| image:: https://readthedocs.org/projects/alchemlyb/badge/?version=latest
    :alt: Documentation
    :scale: 100%
    :target: http://alchemlyb.readthedocs.io/en/latest/

.. |build| image:: https://travis-ci.org/alchemistry/alchemlyb.svg?branch=master
    :alt: Build Status
    :scale: 100%
    :target: https://travis-ci.org/alchemistry/alchemlyb

.. |cov| image:: https://codecov.io/gh/alchemistry/alchemlyb/branch/master/graph/badge.svg
    :alt: Code coverage
    :scale: 100%
    :target: https://codecov.io/gh/alchemistry/alchemlyb

