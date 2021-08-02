Automatic workflow
==================
Though **alchemlyb** is a library offering great flexibility in deriving free
energy estimate, it also provide a easy pipeline that is similar to
`Alchemical Analysis `_ and a
step-by-step version that allows more flexibility.

Note
----
This is an experimental feature and is not API stable.

Fully Automatic analysis
------------------------
A interface similar to
`Alchemical Analysis `_
could be excuted with a single line of command. ::

    >>> import os
    >>> from alchemtest.gmx import load_ABFE
    >>> from alchemlyb.workflows import ABFE
    >>> # Obtain the path of the data
    >>> dir = os.path.dirname(load_ABFE()['data']['complex'][0])
    >>> print(dir)
    'alchemtest/gmx/ABFE/complex'
    >>> workflow = ABFE(units='kcal/mol', software='Gromacs', dir=dir,
    >>>                 prefix='dhdl', suffix='xvg', T=298, skiptime=10,
    >>>                 uncorr='dhdl', threshold=50,
    >>>                 methods=('mbar', 'bar', 'ti'), out='./',
    >>>                 resultfilename='result.out', overlap='O_MBAR.pdf',
    >>>                 breakdown=True, forwrev=10, log='result.log')

This would give the free energy estimate using all of
:class:`~alchemlyb.estimators.TI`, :class:`~alchemlyb.estimators.BAR`,
:class:`~alchemlyb.estimators.MBAR` and the result will be written to the text
file `result.out`. ::

    ------------ --------------------- --------------------- ---------------------
       States         MBAR (kcal/mol)        BAR (kcal/mol)         TI (kcal/mol)
    ------------ --------------------- --------------------- ---------------------
       0 -- 1         0.041  +-  0.001      0.041  +-  0.001      0.041  +-  0.001
       1 -- 2         0.056  +-  0.001      0.055  +-  0.001      0.056  +-  0.001
       2 -- 3         0.082  +-  0.001      0.082  +-  0.002      0.083  +-  0.002
    ...
      26 -- 27        0.766  +-  0.007      0.768  +-  0.010      0.770  +-  0.010
      27 -- 28        0.694  +-  0.008      0.691  +-  0.011      0.690  +-  0.010
      28 -- 29        0.620  +-  0.010      0.616  +-  0.011      0.625  +-  0.011
    ------------ --------------------- --------------------- ---------------------
         coul:        6.290  +-  0.021      6.168  +-  0.026      6.168  +-  0.030
          vdw:       13.872  +-  0.061     13.852  +-  0.037     13.877  +-  0.066
       bonded:        1.469  +-  0.009      1.447  +-  0.003      1.461  +-  0.013
        TOTAL:       21.631  +-  0.064     21.467  +-  0.054     21.506  +-  0.074

The :ref:`overlay matrix for the MBAR estimator <plot_overlap_matrix>` will be
plotted and saved to `O_MBAR.pdf`.

The :ref:`dHdl for TI <plot_TI_dhdl>` will be plotted to `dhdl_TI.pdf`.

The :ref:`dF states <plot_dF_states>` will be plotted to `dF_state.pdf` in
portrait model and `dF_state_long.pdf` in landscape model.

The forward and backward convergence will be plotted to `dF_t.pdf` using
:class:`~alchemlyb.estimators.MBAR`.

.. currentmodule:: alchemlyb.workflows

.. autoclass:: ABFE
   :noindex:

Semi-automatic analysis
-----------------------
The same analysis could also performed in steps allowing access and modification
to the data generated at each stage of the analysis. ::

    >>> import os
    >>> from alchemtest.gmx import load_ABFE
    >>> from alchemlyb.workflows import ABFE
    >>> # Obtain the path of the data
    >>> dir = os.path.dirname(load_ABFE()['data']['complex'][0])
    >>> print(dir)
    'alchemtest/gmx/ABFE/complex'
    >>> # Load the data
    >>> workflow = ABFE(software='Gromacs', dir=dir,
    >>>                 prefix='dhdl', suffix='xvg', T=298, out='./',
    >>>                 log='result.log')
    >>> # Set the unit.
    >>> workflow.update_units('kcal/mol')
    >>> # Decorrelate the data.
    >>> workflow.preprocess(skiptime=10, uncorr='dhdl', threshold=50)
    >>> # Run the estimator
    >>> workflow.estimate(methods=('mbar', 'bar', 'ti'))
    >>> # write the result
    >>> workflow.write(resultfilename='result.out')
    >>> # Plot the overlap matrix
    >>> workflow.plot_overlap_matrix(overlap='O_MBAR.pdf')
    >>> # Plot the dHdl for TI
    >>> workflow.plot_ti_dhdl(dhdl_TI='dhdl_TI.pdf')
    >>> # Plot the dF states
    >>> workflow.plot_dF_state(dF_state='dF_state.pdf')
    >>> # Convergence analysis
    >>> workflow.check_convergence(10, dF_t='dF_t.pdf')


.. currentmodule:: alchemlyb.workflows.ABFE

.. autofunction:: update_units
.. autofunction:: preprocess
.. autofunction:: estimate
.. autofunction:: write
.. autofunction:: plot_overlap_matrix
.. autofunction:: plot_ti_dhdl
.. autofunction:: plot_dF_state
.. autofunction:: check_convergence

.. _Alchemical Analysis: https://github.com/MobleyLab/alchemical-analysis