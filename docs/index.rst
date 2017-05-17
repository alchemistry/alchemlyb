.. alchemlyb documentation master file, created by
   sphinx-quickstart on Wed Feb  1 15:01:17 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

alchemlyb: the simple alchemistry library
=========================================

.. warning:: This library is in an **alpha** state. The library and the documentation is incomplete. Use in production at your own risk.

**alchemlyb** is a library for doing alchemical free energy calculations more easily and with less prone for error.
It includes functions for parsing data from formats common to existing MD engines, subsampling these data, and fitting these data with an estimator to obtain free energies.
These functions are simple in usage and pure in scope, and can be chained together to build customized analyses of data.

**alchemlyb** seeks to be as boring and simple as possible to enable more complex work.
Its components allow work at all scales, from use on small systems using a single workstation to larger datasets that require distributed computing using libraries such as `dask`_.

.. _`dask`: http://dask.pydata.org

Core philosophy
---------------
With its goal to remain simple to use, **alchemlyb**'s design philosophy follows the following points:

1. Use functions when possible, classes only when necessary (or for estimators, see (2)).
2. For estimators, mimic the **scikit-learn** API as much as possible.
3. Aim for a consistent interface throughout, e.g. all parsers take similar inputs and yield a common set of outputs.

Development model
-----------------
This is an open-source project, the hope of which is to produce a library with which the community is happy.
To enable this, the library is a community effort.
Development is done in the open on `GitHub`_, with a `Gitter`_ channel for discussion among developers for fast turnaround on ideas.

.. _`GitHub`: https://github.com/alchemistry/alchemlyb
.. _`Gitter`: https://gitter.im/alchemistry/alchemlyb

Software engineering best-practices are used throughout, including continuous integration testing via Travis CI, up-to-date documentation, and regular releases.

.. toctree::
    :maxdepth: 1
    :caption: User Documentation

    install
    parsing
    preprocessing
    estimators

.. toctree::
   :maxdepth: 1
   :caption: For Developers

   api_proposal

