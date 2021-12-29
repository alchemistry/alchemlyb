.. module:: alchemlyb.postprocessors
	    
.. _postprocessing:

Tools for postprocessing
========================

Tools are available for postprocessing the dataframes.


Unit Conversion
---------------

For all of the input and output dataframes (such as ``u_nk``, ``dHdl``,
``Estimator.delta_f_``, ``Estimator.d_delta_f_``), the :ref:`metadata
<metadata>` is stored as :attr:`pandas.DataFrame.attrs`. The unit of the data
can be converted to :math:`kT`, kJ/mol or kcal/mol via the functions
:func:`~alchemlyb.postprocessors.units.to_kT`,
:func:`~alchemlyb.postprocessors.units.to_kJmol`,
:func:`~alchemlyb.postprocessors.units.to_kcalmol`.
  
    

Unit Conversion Functions
'''''''''''''''''''''''''

.. currentmodule:: alchemlyb.postprocessors

.. autosummary::
    :toctree: postprocessors

    units

Constants and auxiliary functions
---------------------------------

The postprocessing functions can make use of the following auxiliary functions,
which in turn may use constants defined :mod:`alchemlyb.postprocessors.units`.

Scientific constants
''''''''''''''''''''

Common scientific constants based on :mod:`scipy.constants` and are provided
for use across **alchemlyb**.

.. currentmodule:: alchemlyb.postprocessors.units

.. autodata:: kJ2kcal
.. autodata:: R_kJmol
	      

Unit conversion developer information
'''''''''''''''''''''''''''''''''''''

The function :func:`alchemlyb.postprocessors.units.get_unit_converter` provides
the relevant converter for unit conversion via a built-in dispatch table::

    >>> from alchemlyb.postprocessors.units import get_unit_converter
    >>> get_unit_converter('kT')
    <function to_kT>
    >>> get_unit_converter('kJ/mol')
    <function to_kJmol>
    >>> get_unit_converter('kcal/mol')
    <function to_kcalmol>

For unit conversion to work, the dataframes *must* maintain the **energy_unit**
and **temperature** metadata in :attr:`pandas.DataFrame.attrs` as described
under :ref:`note-on-units`.

When *implementing* code then ensure that the :ref:`metadata are maintained
<metadata>` by using :func:`alchemlyb.concat` in place of :func:`pandas.concat`
and use the :func:`alchemlyb.pass_attrs` decorator to copy metadata from an
input dataframe to an output dataframe.
