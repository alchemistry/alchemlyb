Automatic workflow
==================
Though **alchemlyb** is a library offering great flexibility in deriving free
energy estimate, it also provide a easy pipeline that is similar to
`Alchemical Analysis <https://github.com/MobleyLab/alchemical-analysis>`_ and a
step-by-step version that allows more flexibility.

Note
----
This is an experimental feature and is not API stable.

Fully Automatic analysis
------------------------
A interface similar to
`Alchemical Analysis <https://github.com/MobleyLab/alchemical-analysis>`_
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
    >>>                 overlap='O_MBAR.pdf',
    >>>                 breakdown=True, forwrev=10, log='result.log')

This would give the free energy estimate using all of
:class:`~alchemlyb.estimators.TI`, :class:`~alchemlyb.estimators.BAR`,
:class:`~alchemlyb.estimators.MBAR` and the result will be given as
:class:`pandas.DataFrame` to :attr:`alchemlyb.workflows.ABFE.summary` ::

                          MBAR  MBAR_Error        BAR  BAR_Error         TI  TI_Error
    States 0 -- 1     0.065967    0.001293   0.066544   0.001661   0.066663  0.001675
           1 -- 2     0.089774    0.001398   0.089303   0.002101   0.089566  0.002144
           2 -- 3     0.132036    0.001638   0.132687   0.002990   0.133292  0.003055
           3 -- 4     0.116494    0.001213   0.116348   0.002691   0.116845  0.002750
           4 -- 5     0.105251    0.000980   0.106344   0.002337   0.106603  0.002362
           5 -- 6     0.349320    0.002781   0.343399   0.006839   0.350568  0.007393
           6 -- 7     0.402346    0.002767   0.391368   0.006641   0.395754  0.006961
           7 -- 8     0.322284    0.002058   0.319395   0.005333   0.321542  0.005434
           8 -- 9     0.434999    0.002683   0.425680   0.006823   0.430251  0.007155
           9 -- 10    0.355672    0.002219   0.350564   0.005472   0.352745  0.005591
           10 -- 11   3.574227    0.008744   3.513595   0.018711   3.514790  0.018078
           11 -- 12   2.896685    0.009905   2.821760   0.017844   2.823210  0.018088
           12 -- 13   2.223769    0.011229   2.188885   0.018438   2.189784  0.018478
           13 -- 14   1.520978    0.012526   1.493598   0.019155   1.490070  0.019288
           14 -- 15   0.911279    0.009527   0.894878   0.015023   0.896010  0.015140
           15 -- 16   0.892365    0.010558   0.886706   0.015260   0.884698  0.015392
           16 -- 17   1.737971    0.025315   1.720643   0.031416   1.741028  0.030624
           17 -- 18   1.790706    0.025560   1.788112   0.029435   1.801695  0.029244
           18 -- 19   1.998635    0.023340   2.007404   0.027447   2.019213  0.027096
           19 -- 20   2.263475    0.020286   2.265322   0.025023   2.282040  0.024566
           20 -- 21   2.565680    0.016695   2.561324   0.023611   2.552977  0.023753
           21 -- 22   1.384094    0.007553   1.385837   0.011672   1.381999  0.011991
           22 -- 23   1.428567    0.007504   1.422689   0.012524   1.416010  0.013012
           23 -- 24   1.440581    0.008059   1.412517   0.013125   1.408267  0.013539
           24 -- 25   1.411329    0.009022   1.419167   0.013356   1.411446  0.013795
           25 -- 26   1.340320    0.010167   1.360679   0.015213   1.356953  0.015260
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
    >>> # Generate the results
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
.. autofunction:: preprocess
.. autofunction:: estimate
.. autofunction:: generate_result
.. autofunction:: plot_overlap_matrix
.. autofunction:: plot_ti_dhdl
.. autofunction:: plot_dF_state
.. autofunction:: check_convergence

.. _Alchemical Analysis: https://github.com/MobleyLab/alchemical-analysis