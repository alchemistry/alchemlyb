.. currentmodule:: alchemlyb.preprocessing.subsampling

.. _subsampling:

Subsampling
===========

.. automodule:: alchemlyb.preprocessing.subsampling

The functions featured in this module can be used to easily subsample either :math:`\frac{dH}{d\lambda}` (:ref:`dHdl <dHdl>`) or :math:`u_{nk}` (:ref:`u_nk <u_nk>`) datasets to give uncorrelated timeseries.

Each of these functions splits the dataset into groups based on :math:`\lambda` index values, with each group featuring all samples with the same :math:`\lambda` values.
The function then performs its core operation (described below) on each group individually, with samples sorted according to the outermost ``time`` index beforehand.
Each of these groups therefore functions as an individual *timeseries* being subsampled.
The resulting :py:class:`pandas.DataFrame` is the concatenation of all groups after they have been subsampled.


Slicing
-------
The :func:`~alchemlyb.preprocessing.subsampling.slicing` function only performs slicing of the dataset on the basis of ``time``.
The ``lower`` and ``upper`` keyword specify the lower and upper bounds, *inclusive*, while ``step`` indicates the degree to which rows are skipped (e.g. ``step=2`` means to keep every other sample within the bounds).

For example, if we have::

    >>> u_nk
                            0.00      0.25      0.50       0.75       1.00
    time    fep-lambda                                                    
    0.0     0.0         0.309323  3.656838  7.004353  10.351868  13.699383
    10.0    0.0         0.308844  2.616688  4.924532   7.232376   9.540220
    20.0    0.0         0.300940  1.626739  2.952538   4.278337   5.604136
    30.0    0.0         0.309712  1.579647  2.849583   4.119518   5.389454
    40.0    0.0         0.299979  2.255387  4.210794   6.166202   8.121610
    ...                      ...       ...       ...        ...        ...
    39960.0 1.0         3.913594  3.011197  2.108801   1.206405   0.304009
    39970.0 1.0        -0.365724 -0.197390 -0.029055   0.139279   0.307614
    39980.0 1.0         1.495407  1.199280  0.903152   0.607024   0.310897
    39990.0 1.0         1.578606  1.260235  0.941863   0.623492   0.305121
    40000.0 1.0         0.715197  0.611187  0.507178   0.403169   0.299160
    
    [20005 rows x 5 columns]

Then we can grab all samples between :math:`t = 35.0` and :math:`t = 200.0`, inclusive::
    
    >>> slicing(u_nk, lower=35.0, upper=200.0)
                          0.00      0.25      0.50      0.75       1.00
    time  fep-lambda                                                   
    40.0  0.0         0.299979  2.255387  4.210794  6.166202   8.121610
    50.0  0.0         0.301968  3.209507  6.117047  9.024587  11.932126
    60.0  0.0         0.308315  2.284146  4.259977  6.235809   8.211640
    70.0  0.0         0.311610  2.773057  5.234504  7.695950  10.157397
    80.0  0.0         0.301432  1.397817  2.494203  3.590589   4.686975
    ...                    ...       ...       ...       ...        ...
    160.0 1.0         1.396968  1.122475  0.847982  0.573489   0.298995
    170.0 1.0        -1.812027 -1.283715 -0.755404 -0.227092   0.301219
    180.0 1.0         0.979355  0.810205  0.641054  0.471904   0.302754
    190.0 1.0        -2.455231 -1.766201 -1.077171 -0.388141   0.300889
    200.0 1.0         2.419113  1.890386  1.361659  0.832932   0.304205
    
    [85 rows x 5 columns]


In practice, this approach is not much different than directly using :py:property:`pandas.DataFrame.loc`::

    >>> lower, upper, step = (35.0, 200.0, 1)
    >>> u_nk.loc[lower:upper:step]

The :func:`~alchemlyb.preprocessing.subsampling.slicing` function, however, performs some additional checks,
such as ensuring there are not duplicate time values in the dataset, which can happen if data from repeat simulations were concatenated together prior to use.
It's generally advisable to use subsampling functions on data from repeat simulations with ``time`` overlap *individually*, only concatenating afterward just before use with an :ref:`estimator <estimators>`.


.. _subsampling_statinef:

Statistical Inefficiency
------------------------
The :func:`~alchemlyb.preprocessing.subsampling.statistical_inefficiency` function subsamples each timeseries in the dataset by its calculated *statistical inefficiency*, :math:`g`, defined as:

.. math::

    g \equiv (1 + 2\tau) > 1

where :math:`\tau` is the autocorrelation time of the timeseries.
:math:`g` therefore functions as the spacing between uncorrelated samples in the timeseries.
The timeseries is sliced with the :math:`\text{ceil}(g)` as its step (in the conservative case, see the API docs below).

For example, if we have::

    >>> u_nk
                            0.00      0.25      0.50       0.75       1.00
    time    fep-lambda                                                    
    0.0     0.0         0.309323  3.656838  7.004353  10.351868  13.699383
    10.0    0.0         0.308844  2.616688  4.924532   7.232376   9.540220
    20.0    0.0         0.300940  1.626739  2.952538   4.278337   5.604136
    30.0    0.0         0.309712  1.579647  2.849583   4.119518   5.389454
    40.0    0.0         0.299979  2.255387  4.210794   6.166202   8.121610
    ...                      ...       ...       ...        ...        ...
    39960.0 1.0         3.913594  3.011197  2.108801   1.206405   0.304009
    39970.0 1.0        -0.365724 -0.197390 -0.029055   0.139279   0.307614
    39980.0 1.0         1.495407  1.199280  0.903152   0.607024   0.310897
    39990.0 1.0         1.578606  1.260235  0.941863   0.623492   0.305121
    40000.0 1.0         0.715197  0.611187  0.507178   0.403169   0.299160
    
    [20005 rows x 5 columns]

We can extract uncorrelated samples for each ``fep-lambda`` with::

    >>> statistical_inefficiency(u_nk, how='right')
                            0.00      0.25      0.50       0.75       1.00
    time    fep-lambda                                                    
    0.0     0.0         0.309323  3.656838  7.004353  10.351868  13.699383
    20.0    0.0         0.300940  1.626739  2.952538   4.278337   5.604136
    40.0    0.0         0.299979  2.255387  4.210794   6.166202   8.121610
    60.0    0.0         0.308315  2.284146  4.259977   6.235809   8.211640
    80.0    0.0         0.301432  1.397817  2.494203   3.590589   4.686975
    ...                      ...       ...       ...        ...        ...
    39920.0 1.0         3.175234  2.457369  1.739504   1.021640   0.303775
    39940.0 1.0        -1.480193 -1.034104 -0.588015  -0.141925   0.304164
    39960.0 1.0         3.913594  3.011197  2.108801   1.206405   0.304009
    39980.0 1.0         1.495407  1.199280  0.903152   0.607024   0.310897
    40000.0 1.0         0.715197  0.611187  0.507178   0.403169   0.299160
    
    [12005 rows x 5 columns]

The ``how`` parameter indicates the choice of observable for performing the calculation of :math:`g`.

* For :math:`u_{nk}` datasets, the choice of ``'right'`` is recommended: the column immediately to the right of the column corresponding to the group's lambda index value is used as the observable.
* For :math:`\frac{dH}{d\lambda}` datasets, the choice of ``'sum'`` is recommended: the columns are simply summed, and the resulting :py:class:`pandas.Series` is used as the observable.

See the API documentation below on the possible values for ``how``, as well as more detailed explanations for each choice.

It is also possible to choose a specific column, or an arbitrary :py:class:`pandas.Series` (with an *exactly* matching index to the dataset), as the basis for calculating :math:`g`::

    >>> statistical_inefficiency(u_nk, column=0.75)
                            0.00      0.25      0.50       0.75       1.00
    time    fep-lambda                                                    
    0.0     0.0         0.309323  3.656838  7.004353  10.351868  13.699383
    20.0    0.0         0.300940  1.626739  2.952538   4.278337   5.604136
    40.0    0.0         0.299979  2.255387  4.210794   6.166202   8.121610
    60.0    0.0         0.308315  2.284146  4.259977   6.235809   8.211640
    80.0    0.0         0.301432  1.397817  2.494203   3.590589   4.686975
    ...                      ...       ...       ...        ...        ...
    39920.0 1.0         3.175234  2.457369  1.739504   1.021640   0.303775
    39940.0 1.0        -1.480193 -1.034104 -0.588015  -0.141925   0.304164
    39960.0 1.0         3.913594  3.011197  2.108801   1.206405   0.304009
    39980.0 1.0         1.495407  1.199280  0.903152   0.607024   0.310897
    40000.0 1.0         0.715197  0.611187  0.507178   0.403169   0.299160
    
    [12005 rows x 5 columns]

    >>> statistical_inefficiency(u_nk, column=u_nk[0.75])
                            0.00      0.25      0.50       0.75       1.00
    time    fep-lambda                                                    
    0.0     0.0         0.309323  3.656838  7.004353  10.351868  13.699383
    20.0    0.0         0.300940  1.626739  2.952538   4.278337   5.604136
    40.0    0.0         0.299979  2.255387  4.210794   6.166202   8.121610
    60.0    0.0         0.308315  2.284146  4.259977   6.235809   8.211640
    80.0    0.0         0.301432  1.397817  2.494203   3.590589   4.686975
    ...                      ...       ...       ...        ...        ...
    39920.0 1.0         3.175234  2.457369  1.739504   1.021640   0.303775
    39940.0 1.0        -1.480193 -1.034104 -0.588015  -0.141925   0.304164
    39960.0 1.0         3.913594  3.011197  2.108801   1.206405   0.304009
    39980.0 1.0         1.495407  1.199280  0.903152   0.607024   0.310897
    40000.0 1.0         0.715197  0.611187  0.507178   0.403169   0.299160
    
    [12005 rows x 5 columns]

See :py:func:`pymbar.timeseries.statisticalInefficiency` for more details; this is used internally by this subsampler.
Please reference the following if you use this function in your research:

    [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted histogram analysis method for the analysis of simulated and parallel tempering simulations. JCTC 3(1):26-41, 2007.


Equilibrium Detection
---------------------
The :func:`~alchemlyb.preprocessing.subsampling.equilibrium_detection` function subsamples each timeseries in the dataset using the equilibration detection approach developed by John Chodera (see reference below).

For each sorted timeseries in the dataset, and for each sample in each timeseries, the sample is treated as the starting point of the trajectory and the *statistical inefficiency*, :math:`g`, calculated.
Each of these starting points yields an *effective* number of uncorrelated samples, :math:`N_{\text{eff}}`, and the starting point with the greatest :math:`N_{\text{eff}}` is chosen as the start of the *equilibrated* region of the trajectory.

The sorted timeseries is then subsampled by dropping the samples prior to the chosen starting point, then slicing the remaining samples with :math:`\text{ceil}(g)` as its step (in the conservative case, see the API docs below).

For example, if we have::

    >>> u_nk
                            0.00      0.25      0.50       0.75       1.00
    time    fep-lambda                                                    
    0.0     0.0         0.309323  3.656838  7.004353  10.351868  13.699383
    10.0    0.0         0.308844  2.616688  4.924532   7.232376   9.540220
    20.0    0.0         0.300940  1.626739  2.952538   4.278337   5.604136
    30.0    0.0         0.309712  1.579647  2.849583   4.119518   5.389454
    40.0    0.0         0.299979  2.255387  4.210794   6.166202   8.121610
    ...                      ...       ...       ...        ...        ...
    39960.0 1.0         3.913594  3.011197  2.108801   1.206405   0.304009
    39970.0 1.0        -0.365724 -0.197390 -0.029055   0.139279   0.307614
    39980.0 1.0         1.495407  1.199280  0.903152   0.607024   0.310897
    39990.0 1.0         1.578606  1.260235  0.941863   0.623492   0.305121
    40000.0 1.0         0.715197  0.611187  0.507178   0.403169   0.299160
    
    [20005 rows x 5 columns]

We can extract uncorrelated samples for each ``fep-lambda`` with::

    >>> equilibrium_detection(u_nk, how='right')
                            0.00      0.25      0.50       0.75       1.00
    time    fep-lambda                                                    
    0.0     0.0         0.309323  3.656838  7.004353  10.351868  13.699383
    20.0    0.0         0.300940  1.626739  2.952538   4.278337   5.604136
    40.0    0.0         0.299979  2.255387  4.210794   6.166202   8.121610
    60.0    0.0         0.308315  2.284146  4.259977   6.235809   8.211640
    80.0    0.0         0.301432  1.397817  2.494203   3.590589   4.686975
    ...                      ...       ...       ...        ...        ...
    39820.0 1.0         5.238549  4.004569  2.770589   1.536609   0.302630
    39840.0 1.0        -1.611068 -1.133603 -0.656139  -0.178674   0.298791
    39860.0 1.0         0.052599  0.116262  0.179924   0.243587   0.307249
    39880.0 1.0         1.312812  1.060874  0.808936   0.556998   0.305060
    39900.0 1.0         6.940932  5.280870  3.620806   1.960743   0.300680
    
    [11968 rows x 5 columns]

The ``how`` parameter indicates the choice of observable for performing the calculation of :math:`g`.
See the brief recommendations for ``how`` in the :ref:`subsampling_statinef` section above, or the API documentation below on the possible values for ``how`` as well as more detailed explanations for each choice.

It is also possible to choose a specific column, or an arbitrary :py:class:`pandas.Series` (with an *exactly* matching index to the dataset), as the basis for subsampling with equilibrium detection::

    >>> equilibrium_detection(u_nk, column=0.75)
                            0.00      0.25      0.50       0.75       1.00
    time    fep-lambda                                                    
    0.0     0.0         0.309323  3.656838  7.004353  10.351868  13.699383
    20.0    0.0         0.300940  1.626739  2.952538   4.278337   5.604136
    40.0    0.0         0.299979  2.255387  4.210794   6.166202   8.121610
    60.0    0.0         0.308315  2.284146  4.259977   6.235809   8.211640
    80.0    0.0         0.301432  1.397817  2.494203   3.590589   4.686975
    ...                      ...       ...       ...        ...        ...
    39820.0 1.0         5.238549  4.004569  2.770589   1.536609   0.302630
    39840.0 1.0        -1.611068 -1.133603 -0.656139  -0.178674   0.298791
    39860.0 1.0         0.052599  0.116262  0.179924   0.243587   0.307249
    39880.0 1.0         1.312812  1.060874  0.808936   0.556998   0.305060
    39900.0 1.0         6.940932  5.280870  3.620806   1.960743   0.300680
    
    [11961 rows x 5 columns]

    >>> equilibrium_detection(u_nk, column=u_nk[0.75])
                            0.00      0.25      0.50       0.75       1.00
    time    fep-lambda                                                    
    0.0     0.0         0.309323  3.656838  7.004353  10.351868  13.699383
    20.0    0.0         0.300940  1.626739  2.952538   4.278337   5.604136
    40.0    0.0         0.299979  2.255387  4.210794   6.166202   8.121610
    60.0    0.0         0.308315  2.284146  4.259977   6.235809   8.211640
    80.0    0.0         0.301432  1.397817  2.494203   3.590589   4.686975
    ...                      ...       ...       ...        ...        ...
    39820.0 1.0         5.238549  4.004569  2.770589   1.536609   0.302630
    39840.0 1.0        -1.611068 -1.133603 -0.656139  -0.178674   0.298791
    39860.0 1.0         0.052599  0.116262  0.179924   0.243587   0.307249
    39880.0 1.0         1.312812  1.060874  0.808936   0.556998   0.305060
    39900.0 1.0         6.940932  5.280870  3.620806   1.960743   0.300680
    
    [11961 rows x 5 columns]

See :py:func:`pymbar.timeseries.detectEquilibration` for more details; this is used internally by this subsampler.
Please reference the following if you use this function in your research:

    [1] John D. Chodera. A simple method for automated equilibration detection in molecular simulations.  Journal of Chemical Theory and Computation, 12:1799, 2016.

    [2] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted histogram analysis method for the analysis of simulated and parallel tempering simulations. JCTC 3(1):26-41, 2007.


API Reference
-------------
.. autofunction:: alchemlyb.preprocessing.subsampling.slicing
.. autofunction:: alchemlyb.preprocessing.subsampling.statistical_inefficiency
.. autofunction:: alchemlyb.preprocessing.subsampling.equilibrium_detection
