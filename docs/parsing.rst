Parsing data files
==================
**alchemlyb** features parsing submodules for getting raw data from different software packages into common data structures that can be used directly by its :ref:`subsamplers <subsampling>` and :ref:`estimators <estimators>`.
Each submodule features at least two functions, namely:

``extract_dHdl``
  Extract the gradient of the Hamiltonian, :math:`\frac{dH}{d\lambda}`, for each timestep of the sampled state.
  Required input for :ref:`TI-based estimators <estimators_ti>`.

``extract_u_nk``
  Extract reduced potentials, :math:`u_{nk}`, for each timestep of the sampled state and all neighboring states.
  Required input for :ref:`FEP-based estimators <estimators_fep>`.

These functions have a consistent interface across all submodules, often taking a single file as input and any additional parameters required for giving either ``dHdl`` or ``u_nk`` in standard form.

Standard forms of raw data
--------------------------
All components of **alchemlyb** are designed to work together well with minimal work on the part of the user.
To make this possible, the library deals in a common data structure for each ``dHdl`` and ``u_nk``, and all parsers yield these quantities in these standard forms.
The layout of these data structures allow for easy stacking of samples from different simulations while retaining information on where each sample came from using e.g. :py:func:`pandas.concat`.


.. _dHdl:

``dHdl`` standard form
''''''''''''''''''''''
All parsers yielding ``dHdl`` gradients return this as a :py:class:`pandas.DataFrame` with the following structure::

                                     coul        vdw
  time  coul-lambda vdw-lambda                      
    0.0 0.0         0.0         10.264125  -0.522539
    1.0 0.0         0.0          9.214077  -2.492852
    2.0 0.0         0.0         -8.527066  -0.405814
    3.0 0.0         0.0         11.544028  -0.358754
  ..... ...         ...         .........  .........
   97.0 1.0         1.0        -10.681702 -18.603644
   98.0 1.0         1.0         29.518990  -4.955664
   99 0 1.0         1.0         -3.833667  -0.836967
  100.0 1.0         1.0        -12.835707   0.786278

This is a multi-index DataFrame, giving ``time`` for each sample as the outermost index, and the value of each :math:`\lambda` from which the sample came as subsequent indexes.
The columns of the DataFrame give the value of :math:`\frac{dH}{d\lambda}` with respect to each of these separate :math:`\lambda` parameters.

For datasets that sample with only a single :math:`\lambda` parameter, then the DataFrame will feature only a single column perhaps like::

                          fep
  time  fep-lambda                     
    0.0 0.0         10.264125
    1.0 0.0          9.214077
    2.0 0.0         -8.527066
    3.0 0.0         11.544028
  ..... ...         .........
   97.0 1.0        -10.681702
   98.0 1.0         29.518990
   99 0 1.0         -3.833667
  100.0 1.0        -12.835707


.. _u_nk:

``u_nk`` standard form
''''''''''''''''''''''
All parsers yielding ``u_nk`` reduced potentials return this as a :py:class:`pandas.DataFrame` with the following structure::

                                   (0.0, 0.0) (0.25, 0.0) (0.5, 0.0)  ...  (1.0, 1.0)
  time  coul-lambda vdw-lambda                                                
    0.0 0.0         0.0         -22144.50   -22144.24  -22143.98        -21984.81
    1.0 0.0         0.0         -21985.24   -21985.10  -21984.96        -22124.26
    2.0 0.0         0.0         -22124.58   -22124.47  -22124.37        -22230.61
    3.0 1.0         0.1         -22230.65   -22230.63  -22230.62        -22083.04
  ..... ...         ...         .........   .........  .........  ...   .........
   97.0 1.0         1.0         -22082.29   -22082.54  -22082.79        -22017.42
   98.0 1.0         1.0         -22087.57   -22087.76  -22087.94        -22135.15
   99.0 1.0         1.0         -22016.69   -22016.93  -22017.17        -22057.68
  100.0 1.0         1.0         -22137.19   -22136.51  -22135.83        -22101.26

This is a multi-index DataFrame, giving ``time`` for each sample as the outermost index, and the value of each :math:`\lambda` from which the sample came as subsequent indexes.
The columns of the DataFrame give the value of :math:`u_{nk}` for each set of :math:`\lambda` parameters values were recorded for.
Column labels are the values of the :math:`\lambda` parameters as a tuple in the same order as they appear in the multi-index.

For datasets that sample only a single :math:`\lambda` parameter, then the DataFrame will feature only a single index in addition to ``time``, with the values of :math:`\lambda` for which reduced potentials were recorded given as column labels::

                              0.0        0.25        0.5  ...         1.0
      time  fep-lambda                                               
        0.0 0.0         -22144.50   -22144.24  -22143.98        -21984.81
        1.0 0.0         -21985.24   -21985.10  -21984.96        -22124.26
        2.0 0.0         -22124.58   -22124.47  -22124.37        -22230.61
        3.0 1.0         -22230.65   -22230.63  -22230.62        -22083.04
      ..... ...         .........   .........  .........  ...   .........
       97.0 1.0         -22082.29   -22082.54  -22082.79        -22017.42
       98.0 1.0         -22087.57   -22087.76  -22087.94        -22135.15
       99.0 1.0         -22016.69   -22016.93  -22017.17        -22057.68
      100.0 1.0         -22137.19   -22136.51  -22135.83        -22101.26

A note on units
'''''''''''''''
Throughout ``alchemlyb``, energy quantities such as ``dHdl`` or ``u_nk`` are given in units of :math:`k_B T`.
Also, although parsers will extract timestamps from input data, these are taken as-is and the library does not have any awareness of units for these.
Keep this in mind when doing, e.g. :ref:`subsampling <subsampling>`.

Parsers by software package
---------------------------
**alchemlyb** tries to provide parser functions for as many simulation packages as possible.
See the documentation for the package you are using for more details on parser usage, including the assumptions parsers make and suggestions for how output data should be structured for ease of use:

.. currentmodule:: alchemlyb.parsing

.. autosummary::
    :toctree: parsing

    gmx
    amber
    
