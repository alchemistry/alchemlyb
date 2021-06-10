.. alchemlyb documentation master file, created by
   sphinx-quickstart on Wed Feb  1 15:01:17 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

alchemlyb: the simple alchemistry library
=========================================

**alchemlyb** is a library for doing alchemical free energy calculations more easily.
It includes functions for parsing data from formats common to existing MD engines, subsampling these data, and fitting these data with an estimator to obtain free energies.
These functions are simple in usage and pure in scope, and can be chained together to build customized analyses of data.

**alchemlyb** seeks to be as boring and simple as possible to enable more complex work.
Its components allow work at all scales, from use on small systems using a single workstation to larger datasets that require distributed computing using libraries such as `dask`_.

The library is *under active development* and the API is still somewhat in flux. However, it is used by multiple groups in a production environment. We use `semantic versioning`_ to indicate clearly what kind of changes you may expect between releases. See :ref:`contact` for how to get in touch if you have questions or find problems.

.. _`dask`: http://dask.pydata.org
.. _`semantic versioning`: https://semver.org

Core philosophy
---------------
With its goal to remain simple to use, **alchemlyb**'s design philosophy follows the following points:

1. Use functions when possible, classes only when necessary (or for estimators, see (2)).
2. For estimators, mimic the **scikit-learn** API as much as possible.
3. Aim for a consistent interface throughout, e.g. all parsers take similar inputs and yield a common set of outputs.

For more details, see the Roadmap_.

.. _Roadmap: https://github.com/alchemistry/alchemlyb/wiki/Roadmap


Development model
-----------------
This is an open-source project, the hope of which is to produce a library with which the community is happy.
To enable this, the library is a community effort.
Development is done in the open on `GitHub`_.

Software engineering best-practices are used throughout, including continuous integration testing via Travis CI, up-to-date documentation, and regular releases.

.. Note::
   With release 0.5.0, the alchemlyb project adopted `NEP 29`_ to determine which versions of Python and NumPy_ will be supported.
   When we release a new major or minor version, alchemlyb will support *at least all minor versions of Python introduced and released in the prior 42 months from the release date with a minimum of 2 minor versions of Python*, and *all minor versions of NumPy released in the prior 24 months from the anticipated release date with a minimum of 3 minor versions of NumPy*.

The pandas_ package (one of our other primary dependencies) also follows `NEP 29`_ so this support policy will make it easier to maintain **alchemlyb** in the future.

   
.. _`GitHub`: https://github.com/alchemistry/alchemlyb
.. _`NEP 29`: https://numpy.org/neps/nep-0029-deprecation_policy.html
.. _NumPy: https://numpy.org
.. _pandas: https://pandas.pydata.org/

.. _contact:

Contributing
------------
Contributions are very welcome. If you have bug reports or feature requests or questions then please get in touch with us through the `Issue Tracker`_. We also welcome code contributions: have a look at our `Developer Guide`_ and submit a pull request against the `alchemistry/alchemlyb`_ GitHub repository.

.. _`Developer Guide`: https://github.com/alchemistry/alchemlyb/wiki/Developer-Guide
.. _`Issue Tracker`: https://github.com/alchemistry/alchemlyb/issues
.. _`alchemistry/alchemlyb`: https://github.com/alchemistry/alchemlyb

.. toctree::
    :maxdepth: 1
    :caption: User Documentation

    install
    parsing
    preprocessing
    estimators
    visualisation

.. toctree::
   :maxdepth: 1
   :caption: For Developers

   api_principles

