.. -*- coding: utf-8 -*-

API principles
==============

The following is an overview over the guiding principles and ideas that underpin the API of alchemlyb.


`alchemlyb`
-----------

`alchemlyb` is a library that seeks to make doing alchemical free energy calculations easier and less error prone.
It includes functions for parsing data from formats common to existing MD engines, subsampling these data, and fitting these data with an estimator to obtain free energies.
These functions are simple in usage and pure in scope, and can be chained together to build customized analyses of data.

`alchemlyb` seeks to be as boring and simple as possible to enable more complex work.
Its components allow work at all scales, from use on small systems using a single workstation to larger datasets that require distributed computing using libraries such as dask.

First and foremost, scientific code must be *correct* and we try to ensure this requirement by following best software engineering practices during development, close to full test coverage of all code in the library, and providing citations to published papers for included algorithms. We use a curated, public data set (`alchemtest`_) for automated testing.

.. _alchemtest: https://github.com/alchemistry/alchemtest


Core philosophy
---------------

1. Use functions when possible, classes only when necessary (or for estimators, see (2)).
2. For estimators, mimic the **scikit-learn** API as much as possible.
3. Aim for a consistent interface throughout, e.g. all parsers take similar inputs and yield a common set of outputs.
4. Have all functionality tested.
   

API components
--------------

The library is structured as follows, following a similar style to
**scikit-learn**::

    alchemlyb
    ├── parsing
    │   ├── amber.py
    │   ├── gmx.py
    │   ├── gomc.py
    │   ├── namd.py
    │   └── ...
    ├── preprocessing
    │   ├── subsampling.py
    │   └── ...
    ├── estimators
    │   ├── bar_.py
    │   ├── mbar_.py
    │   ├── ti_.py
    │   └── ...        
    ├── convergence          ### NOT IMPLEMENTED
    │   ├── convergence.py
    │   └── ...    
    └── visualisation
	├── convergence.py
	├── dF_state.py
	├── mbar_matrix.py
	├── ti_dhdl.py
	└── ...
  

	 

The :mod:`~alchemlyb.parsing` submodule contains parsers for individual MD engines, since the output files needed to perform alchemical free energy calculations vary widely and are not standardized.
Each module at the very least provides an `extract_u_nk` function for extracting reduced potentials (needed for MBAR), as well as an `extract_dHdl` function for extracting derivatives required for thermodynamic integration.
Other helper functions may be exposed for additional processing, such as generating an XVG file from an EDR file in the case of GROMACS.
All `extract\_*` functions take similar arguments (a file path,
parameters such as temperature), and produce standard outputs
(:class:`pandas.DataFrame` for reduced potentials, :class:`pandas.Series` for derivatives).

The :mod:`~alchemlyb.preprocessing` submodule features functions for subsampling timeseries, as may be desired before feeding them to an estimator.
So far, these are limited to `slicing`, `statistical_inefficiency`, and `equilibrium_detection` functions, many of which make use of subsampling schemes available from :mod:`pymbar`.
These functions are written in such a way that they can be easily composed as parts of complex processing pipelines.

The :mod:`~alchemlyb.estimators` module features classes *a la* **scikit-learn** that can be initialized with parameters that determine their behavior and then "trained" on a `fit` method.
MBAR, BAR, and thermodynamic integration (TI) as the major methods are all implemented.
Correct error estimates require the use of time series with independent samples.

The :mod:`~alchemlyb.convergence` submodule will feature convenience functions/classes for doing convergence analysis using a given dataset and a chosen estimator, though the form of this is not yet thought-out.
However, the `gist a41e5756a58e1775e3e3a915f07bfd37`_ shows an example for how this can be done already in practice.

The :mod:`visualization` submodule contains convenience plotting functions as known from, for example, `alchemical-analysis.py`_.

All of these components lend themselves well to writing clear and flexible pipelines for processing data needed for alchemical free energy calculations, and furthermore allow for scaling up via libraries like `dask`_ or `joblib`_.

.. _`alchemical-analysis.py`: https://github.com/MobleyLab/alchemical-analysis/

.. _dask: https://dask.org/

.. _joblib: https://joblib.readthedocs.io


Development model
-----------------

This is an open-source project, the hope of which is to produce a library with which the community is happy.
To enable this, the library will be a community effort.
Development is done in the open on GitHub.
Software engineering best-practices will be used throughout, including continuous integration testing via Travis CI, up-to-date documentation, and regular releases.

Following discussion, refinement, and consensus on this proposal, issues for each need will be posted and work will begin on filling out the rest of the library.
In particular, parsers will be crowdsourced from the existing community and refined into the consistent form described above.


Historical notes
----------------

Some of the components were originally demoed in `gist a41e5756a58e1775e3e3a915f07bfd37`_.

.. _`gist a41e5756a58e1775e3e3a915f07bfd37`:
  https://gist.github.com/dotsdl/a41e5756a58e1775e3e3a915f07bfd37

David Dotson (@dotsdl) started the project while employed as a software engineer by Oliver Beckstein (@orbeckst), and this project was a primary point of focus for him in this position.
