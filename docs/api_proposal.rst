API Proposal
============
The following is an API proposal for the library.
This proposal has been prototyped, with some of the components described already implemented at a basic level.
This functionality is demoed in [this gist](https://gist.github.com/dotsdl/a41e5756a58e1775e3e3a915f07bfd37).


`alchemlyb`
-----------
`alchemlyb` is a library that seeks to make doing alchemical free energy calculations easier and less error prone.
It will include functions for parsing data from formats common to existing MD engines, subsampling these data, and fitting these data with an estimator to obtain free energies.
These functions will be simple in usage and pure in scope, and can be chained together to build customized analyses of data.

`alchemlyb` seeks to be as boring and simple as possible to enable more complex work.
Its components allow work at all scales, from use on small systems using a single workstation to larger datasets that require distributed computing using libraries such as dask.


Core philosophy
---------------

1. Use functions when possible, classes only when necessary (or for estimators, see (2)).
2. For estimators, mimic the **scikit-learn** API as much as possible.
3. Aim for a consistent interface throughout, e.g. all parsers take similar inputs and yield a common set of outputs.


API components
--------------

The library is structured as follows, following a similar style to **scikit-learn**::

    alchemlyb
    |
     -- parsing
     |  |
     |   -- gmx
     |  |
     |   -- amber
     |  |
     |   -- openmm
     |  |
     |   -- namd
     |
      -- preprocessing
     |  |
     |   -- subsampling
     |
      -- estimators
        |
         -- mbar_
        |
         -- ti_

The ``parsing`` submodule contains parsers for individual MD engines, since the output files needed to perform alchemical free energy calculations vary widely and are not standardized.
Each module at the very least provides an `extract_u_nk` function for extracting reduced potentials (needed for MBAR), as well as an `extract_DHdl` function for extracting derivatives required for thermodynamic integration.
Other helper functions may be exposed for additional processing, such as generating an XVG file from an EDR file in the case of GROMACS.
All `extract\_*` functions take similar arguments (a file path, parameters such as temperature), and produce standard outputs (`pandas.DataFrame`s for reduced potentials, `pandas.Series` for derivatives).

The `preprocessing` submodule features functions for subsampling timeseries, as may be desired before feeding them to an estimator.
So far, these are limited to `slicing`, `statistical_inefficiency`, and `equilibrium_detection` functions, many of which make use of subsampling schemes available from `pymbar`.
These functions are written in such a way that they can be easily composed as parts of complex processing pipelines.

The `estimators` module features classes *a la* **scikit-learn** that can be initialized with parameters that determine their behavior and then "trained" on a `fit` method.
So far, `MBAR` has been partially implemented, and because the numerical heavy-lifting is already well-implemented in `pymbar.MBAR`, this class serves to give an interface that will be familiar and consistent with the others.
Thermodynamic integration is not yet implemented.

The `convergence` submodule will feature convenience functions/classes for doing convergence analysis using a given dataset and a chosen estimator, though the form of this is not yet thought-out.
However, the gist shows an example for how this can be done already in practice.

All of these components lend themselves well to writing clear and flexible pipelines for processing data needed for alchemical free energy calculations, and furthermore allow for scaling up via libraries like `dask` or `joblib`.


Development model
-----------------

This is an open-source project, the hope of which is to produce a library with which the community is happy.
To enable this, the library will be a community effort.
Development is done in the open on GitHub.
Software engineering best-practices will be used throughout, including continuous integration testing via Travis CI, up-to-date documentation, and regular releases.

David Dotson (@dotsdl) is employed as a software engineer by Oliver Beckstein (@orbeckst), and this project is a primary point of focus for him in this position.
Ian Kenney (@ianmkenney) and Hannes Loeffler (@halx) have also expressed interest in direct development.

Following discussion, refinement, and consensus on this proposal, issues for each need will be posted and work will begin on filling out the rest of the library.
In particular, parsers will be crowdsourced from the existing community and refined into the consistent form described above.
Expertise in ensuring theoretical correctness of each component, in particular estimators, will be needed from David Mobley (@davidmobley), John Chodera (@jchodera), and Michael Shirts (@mrshirts).
