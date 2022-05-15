The ABFE workflow
==================
Though **alchemlyb** is a library offering great flexibility in deriving free
energy estimate, it also provide a easy pipeline that is similar to
`Alchemical Analysis <https://github.com/MobleyLab/alchemical-analysis>`_ and a
step-by-step version that allows more flexibility.

Fully Automatic analysis
------------------------
A interface similar to
`Alchemical Analysis <https://github.com/MobleyLab/alchemical-analysis>`_
could be excuted with two lines of command. ::

    >>> import os
    >>> from alchemtest.gmx import load_ABFE
    >>> from alchemlyb.workflows import ABFE
    >>> # Obtain the path of the data
    >>> dir = os.path.dirname(load_ABFE()['data']['complex'][0])
    >>> print(dir)
    'alchemtest/gmx/ABFE/complex'
    >>> workflow = ABFE(units='kcal/mol', software='Gromacs', dir=dir,
    >>>                 prefix='dhdl', suffix='xvg', T=298, out='./')
    >>> workflow.run(skiptime=10, uncorr='dhdl', threshold=50,
    >>>              methods=('mbar', 'bar', 'ti'), overlap='O_MBAR.pdf',
    >>>              breakdown=True, forwrev=10)

This would give the free energy estimate using all of
:class:`~alchemlyb.estimators.TI`, :class:`~alchemlyb.estimators.BAR`,
:class:`~alchemlyb.estimators.MBAR` and the result will be stored in
:attr:`~alchemlyb.workflows.ABFE.summary` as :class:`pandas.Dataframe`. ::


                          MBAR  MBAR_Error        BAR  BAR_Error         TI  TI_Error
    States 0 -- 1     0.065967    0.001293   0.066544   0.001661   0.066663  0.001675
           1 -- 2     0.089774    0.001398   0.089303   0.002101   0.089566  0.002144
           2 -- 3     0.132036    0.001638   0.132687   0.002990   0.133292  0.003055
    ...
           26 -- 27   1.243745    0.011239   1.245873   0.015711   1.248959  0.015762
           27 -- 28   1.128429    0.012859   1.124554   0.016999   1.121892  0.016962
           28 -- 29   1.010313    0.016442   1.005444   0.017692   1.019747  0.017257
    Stages coul      10.215658    0.033903  10.017838   0.041839  10.017854  0.048744
           vdw       22.547489    0.098699  22.501150   0.060092  22.542936  0.106723
           bonded     2.374144    0.014995   2.341631   0.005507   2.363828  0.021078
           TOTAL     35.137291    0.103580  34.860619   0.087022  34.924618  0.119206

The :ref:`overlay matrix for the MBAR estimator <plot_overlap_matrix>` will be
plotted and saved to `O_MBAR.pdf`.

The :ref:`dHdl for TI <plot_TI_dhdl>` will be plotted to `dhdl_TI.pdf`.

The :ref:`dF states <plot_dF_states>` will be plotted to `dF_state.pdf` in
portrait model and `dF_state_long.pdf` in landscape model.

The forward and backward convergence will be plotted to `dF_t.pdf` using
:class:`~alchemlyb.estimators.MBAR` and save in
:attr:`~alchemlyb.workflows.ABFE.convergence`.

.. currentmodule:: alchemlyb.workflows

.. autoclass:: ABFE
    :no-index:
    :members:
    .. automethod:: run





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
    >>>                 prefix='dhdl', suffix='xvg', T=298, out='./')
    >>> # Set the unit.
    >>> workflow.update_units('kcal/mol')
    >>> # Read the data
    >>> workflow.read()
    >>> # Decorrelate the data.
    >>> workflow.preprocess(skiptime=10, uncorr='dhdl', threshold=50)
    >>> # Run the estimator
    >>> workflow.estimate(methods=('mbar', 'bar', 'ti'))
    >>> # Retrieve the result
    >>> summary = workflow.generate_result()
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
.. autofunction:: read
.. autofunction:: preprocess
.. autofunction:: estimate
.. autofunction:: generate_result
.. autofunction:: plot_overlap_matrix
.. autofunction:: plot_ti_dhdl
.. autofunction:: plot_dF_state
.. autofunction:: check_convergence