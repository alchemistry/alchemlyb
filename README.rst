alchemlyb: the simple alchemistry library
=========================================

|doi| |docs| |build| |cov| |anaconda|

**alchemlyb** makes alchemical free energy calculations easier to do
by leveraging the full power and flexibility of the PyData stack. It
includes:

1. Parsers for extracting raw data from output files of common
   molecular dynamics engines such as `GROMACS`_, `AMBER`_, `NAMD`_
   and `other simulation codes`_.

2. Subsamplers for obtaining uncorrelated samples from timeseries data
   (including extracting independent, equilibrated samples
   [Chodera2016]_ as implemented in the pymbar_ package).

3. Estimators for obtaining free energies directly from this data, using
   best-practices approaches for multistate Bennett acceptance ratio (MBAR)
   [Shirts2008]_ and BAR (from pymbar_) and thermodynamic integration (TI).

.. _GROMACS: http://www.gromacs.org/
.. _AMBER: http://ambermd.org/
.. _NAMD: http://www.ks.uiuc.edu/Research/namd/
.. _`other simulation codes`: https://alchemlyb.readthedocs.io/en/latest/parsing.html
.. _`pymbar`: http://pymbar.readthedocs.io/


Installation
------------

**Install** via ``pip`` from `PyPi (alchemlyb)`_ ::

  pip install alchemlyb

or as a `conda`_ package from the `conda-forge (alchemlyb)`_ channel
::

  conda install -c conda-forge alchemlyb 


**Update** with ``pip`` ::

  pip install --update alchemlyb

or with ``conda`` run ::

  conda update -c conda-forge alchemlyb

to get the latest released version.

.. _`PyPi (alchemlyb)`: https://pypi.org/project/alchemlyb/
.. _`conda`: https://conda.io/
.. _`conda-forge (alchemlyb)`: https://anaconda.org/conda-forge/alchemlyb

Getting involved
----------------

Contributions of all kinds are very welcome.

If you have questions or want to discuss alchemlyb please post in the `alchemlyb Discussions`_.

If you have bug reports or feature requests then please get in touch with us through the `Issue Tracker`_.

We also welcome code contributions: have a look at our `Developer Guide`_. Open an issue with the proposed fix or change in the `Issue Tracker`_ and submit a pull request against the `alchemistry/alchemlyb`_ GitHub repository.

.. _`alchemlyb Discussions`: https://github.com/alchemistry/alchemlyb/discussions
.. _`Developer Guide`: https://github.com/alchemistry/alchemlyb/wiki/Developer-Guide
.. _`Issue Tracker`: https://github.com/alchemistry/alchemlyb/issues
.. _`alchemistry/alchemlyb`: https://github.com/alchemistry/alchemlyb

   

References
----------

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

.. |build| image:: https://github.com/alchemistry/alchemlyb/actions/workflows/ci.yaml/badge.svg?branch=master
    :alt: Build Status
    :scale: 100%
    :target: https://github.com/alchemistry/alchemlyb/actions/workflows/ci.yaml

.. |cov| image:: https://codecov.io/gh/alchemistry/alchemlyb/branch/master/graph/badge.svg
    :alt: Code coverage
    :scale: 100%
    :target: https://codecov.io/gh/alchemistry/alchemlyb


.. |anaconda| image:: https://anaconda.org/conda-forge/alchemlyb/badges/version.svg
   :alt: anaconda package
   :scale: 100%	 
   :target: https://anaconda.org/conda-forge/alchemlyb
