alchemlyb: the simple alchemistry library
=========================================

|doi| |docs| |build| |cov|

**alchemlyb** makes alchemical free energy calculations easier to do
by leveraging the full power and flexibility of the PyData stack. It
includes:

1. Parsers for extracting raw data from output files of common
   molecular dynamics engines such as `GROMACS`_, `AMBER`_, `NAMD`_
   and `other simulation codes`_.

2. Subsamplers for obtaining uncorrelated samples from timeseries data.

3. Estimators for obtaining free energies directly from this data, using
   best-practices approaches for multistate Bennett acceptance ratio (MBAR)
   [Shirts2008]_ and thermodynamic integration (TI).

In particular, it uses internally the excellent `pymbar
<http://pymbar.readthedocs.io/>`_ library for performing MBAR and extracting
independent, equilibrated samples [Chodera2016]_.

.. [Shirts2008] Shirts, M.R., and Chodera, J.D. (2008). Statistically optimal
    analysis of samples from multiple equilibrium states. The Journal of Chemical
    Physics 129, 124105.

.. [Chodera2016] Chodera, J.D. (2016). A Simple Method for Automated
    Equilibration Detection in Molecular Simulations. Journal of Chemical Theory
    and Computation 12, 1799–1805.

.. _GROMACS: http://www.gromacs.org/

.. _AMBER: http://ambermd.org/

.. _NAMD: http://www.ks.uiuc.edu/Research/namd/

.. _`other simulation codes`: https://alchemlyb.readthedocs.io/en/latest/parsing.html
    
.. |doi| image:: https://zenodo.org/badge/68669096.svg
    :alt: Zenodo DOI
    :scale: 100%
    :target: https://zenodo.org/badge/latestdoi/68669096

.. |docs| image:: https://readthedocs.org/projects/alchemlyb/badge/?version=latest
    :alt: Documentation
    :scale: 100%
    :target: http://alchemlyb.readthedocs.io/en/latest/

.. |build| image:: https://github.com/alchemistry/alchemlyb/actions/workflows/ci.yaml/badge.svg?branch=master
    :alt: Build Status
    :scale: 100%
    :target: https://github.com/alchemistry/alchemlyb/actions/workflows/ci.yaml

.. |cov| image:: https://codecov.io/gh/alchemistry/alchemlyb/branch/master/graph/badge.svg
    :alt: Code coverage
    :scale: 100%
    :target: https://codecov.io/gh/alchemistry/alchemlyb

