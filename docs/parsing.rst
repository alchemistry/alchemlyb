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

This is a multi-index DataFrame, giving ``time`` for each sample as the outermost index, and the value of each :math:`\lambda` from which the sample as subsequent indexes.
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

The layout of this data structure allows for easy stacking of samples from different simulations while retaining information on where each sample came from using e.g. :py:func:`pandas.concat`.


Parsers by software package
---------------------------
.. toctree::
    :maxdepth: 1

    parsing-gmx
