.. module:: alchemlyb.parsing


.. _parsing:

Parsing data files
==================
**alchemlyb** features parsing submodules for getting raw data from different software packages into common data structures that can be used directly by its :ref:`subsamplers <subsampling>` and :ref:`estimators <estimators>`.
Each submodule features at least two functions, namely:

:func:`extract_dHdl`
  Extract the gradient of the Hamiltonian, :math:`\frac{dH}{d\lambda}`, for each timestep of the sampled state.
  Required input for :ref:`TI-based estimators <estimators_ti>`.

:func:`extract_u_nk`
  Extract reduced potentials, :math:`u_{nk}`, for each timestep of the sampled state and all neighboring states.
  Required input for :ref:`FEP-based estimators <estimators_fep>`.

:func:`extract`
  Extract both reduced potentials and the gradient of the Hamiltonian, :math:`u_{nk}` and :math:`\frac{dH}{d\lambda}`, in the form of a dictionary ``'dHdl': Series,  'u_nk': DataFrame``.
  Required input for :ref:`FEP-based estimators <estimators_fep>` and :ref:`TI-based estimators <estimators_ti>`.

These functions have a consistent interface across all submodules, often taking a single file as input and any additional parameters required for giving either ``dHdl`` or ``u_nk`` in standard form.

Standard forms of raw data
--------------------------
All components of **alchemlyb** are designed to work together well with minimal
work on the part of the user. To make this possible, the library deals in a
common data structure for each ``dHdl`` and ``u_nk``, and all parsers yield
these quantities in these standard forms.  The common data structure is a 
:class:`pandas.DataFrame`. Normally, it should be sufficient to just pass the
``dHdl`` and ``u_nk`` dataframes from one ``alchemlyb`` function to the
next. However, being a :class:`~pandas.DataFrame` provides enormous flexibility
if the data need to be reorganized or transformed because of the powerful tools 
that :mod:`pandas` makes available to manipulate these data structures. 

.. Warning::
   When alchemlyb dataframes are transformed with standard pandas functions
   (such as :func:`pandas.concat`),
   care needs to be taken to ensure that ``alchemlyb`` metadata, which are stored
   in the dataframe, are maintained and propagated during processing of 
   ``alchemlyb`` dataframes. 
   See :ref:`metadata propagation <metadata>` for how do work with dataframes
   safely in ``alchemlyb``.


The metadata (such as the unit of the energy and temperature) are stored in 
:attr:`pandas.DataFrame.attrs`, a :class:`dict`. Functions in ``alchemlyb`` are
aware of these metadata but working with the data using :mod:`pandas`
requires some care due to shortcomings in how pandas currently handles
metadata (see issue `pandas-dev/pandas#28283 <https://github.com/pandas-dev/pandas/issues/28283>`_).


Serialisation
'''''''''''''

Alchemlyb data structures (``dHdl`` and ``u_nk``) can be serialized as dataframes
and made persistent.
We use the `parquet <https://pandas.pydata.org/docs/user_guide/io.html#io-parquet>`_
format for serializing (writing) to a file and de-serializing (reading) from a 
parquet file.

For serialization we simply use the :meth:`pandas.DataFrame.to_parquet` method of
a :class:`pandas.DataFrame`. For loading alchemlyb data we provide the 
:func:`alchemlyb.parsing.parquet.extract_dHdl` and 
:func:`alchemlyb.parsing.parquet.extract_u_nk` functions as shown in the example::

    from alchemlyb.parsing.parquet import extract_dHdl, extract_u_nk
    import pandas as pd

    u_nk.to_parquet(path='u_nk.parquet', index=True)
    dHdl.to_parquet(path='dHdl.parquet', index=True)

    new_u_nk = extract_u_nk('u_nk.parquet', T=300)
    new_dHdl = extract_dHdl('dHdl.parquet', T=300)

.. Note::
    Serialization of :class:`pandas.DataFrame` to `parquet` file is only allowed
    for `pandas>=2`, whereas the deserialization is permitted for any pandas version.

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


.. _note-on-units:

A note on units
'''''''''''''''

``alchemlyb`` reads input files in native energy units and converts them to a common
unit, the energy measured in :math:`k_B T`, where :math:`k_B` is `Boltzmann's constant 
<https://physics.nist.gov/cgi-bin/cuu/Value?k>`_ and :math:`T` is the thermodynamic 
absolute temperature in Kelvin. Therefore, all parsers require specification of :math:`T`.
 
Throughout ``alchemlyb``, the metadata, such as the energy unit and temperature of 
the dataset, are stored as a dictionary in :attr:`pandas.DataFrame.attrs` metadata
attribute. The keys of the :attr:`~pandas.DataFrame.attrs`  dictionary are

``"temperature"``
   the temperature at which the simulation was performed, in Kelvin
   
``"energy_unit"``
   the unit of energy, such as "kT", "kcal/mol", "kJ/mol" (as
   defined in :mod:`~alchemlyb.postprocessors.units`)

Conversion functions in :mod:`alchemlyb.postprocessing` and elsewhere may use the 
metadata for unit conversion and other transformations.

As the following example shows, after parsing of data files, the energy unit is "kT", i.e.,
the :math:`\partial H/\partial\lambda` timeseries is measured in multiples of 
:math:`k_B T` per lambda step::

    >>> from alchemtest.gmx import load_benzene
    >>> from alchemlyb.parsing.gmx import extract_dHdl
    >>> dataset = load_benzene()
    >>> dhdl = extract_dHdl(dataset['data']['Coulomb'][0], 310)
    >>> dhdl.attrs['temperature']
    310
    >>> dhdl.attrs['energy_unit']
    'kT'

Also, although parsers will extract timestamps from input data, these are taken as-is and the library does not have any awareness of units for these.
Keep this in mind when doing, e.g. :ref:`subsampling <subsampling>`.

.. _metadata:

Metadata Propagation
''''''''''''''''''''
The metadata is stored in :attr:`pandas.DataFrame.attrs`. Though common pandas
functions can safely propagate the metadata, the metadata might get lost
during some operations such as concatenation (`pandas-dev/pandas#28283
<https://github.com/pandas-dev/pandas/issues/28283>`_). :func:`alchemlyb.concat`
is provided to replace :func:`pandas.concat` allowing the safe propagation
of metadata. ::

    >>> import alchemlyb
    >>> from alchemtest.gmx import load_benzene
    >>> from alchemlyb.parsing.gmx import extract_dHdl
    >>> dataset = load_benzene().data
    >>> dhdl_coul = alchemlyb.concat([extract_dHdl(xvg, T=300) for xvg in dataset['Coulomb']])
    >>> dhdl_coul.attrs
    {'temperature': 300, 'energy_unit': 'kT'}

.. autofunction:: alchemlyb.concat

Although all functions in **alchemlyb** will safely propagate the metadata, if
the user is interested in writing custom data manipulation functions,
a decorator :func:`alchemlyb.pass_attrs` could be used to pass the metadata
from the input data frame (first positional argument) to the output
dataframe to ensure safe propagation of metadata. ::

    >>> from alchemlyb import pass_attrs
    >>> @pass_attrs
    >>> def manipulation(dataframes, *args, **kwargs):
    >>>     return func(dataframes, *args, **kwargs)

.. autofunction:: alchemlyb.pass_attrs

Parsers by software package
---------------------------
**alchemlyb** tries to provide parser functions for as many simulation packages as possible.
See the documentation for the package you are using for more details on parser usage, including the assumptions parsers make and suggestions for how output data should be structured for ease of use:

.. currentmodule:: alchemlyb.parsing

.. autosummary::
    :toctree: parsing

    gmx
    amber
    namd
    gomc
    parquet
    lammps
    
