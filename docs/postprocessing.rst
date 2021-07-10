.. _postprocessing:

Tools for postprocessing
========================

Tools are available for postprocessing the dataframes.

Constants and auxiliary functions
---------------------------------

Scientific constants
''''''''''''''''''''
Common scientific constants based on :mod:`scipy.constants` are provided for
using across **alchemlyb**.

.. currentmodule:: alchemlyb.postprocessors.units

.. autodata:: kJ2kcal
.. autodata:: R_kJmol

Unit Conversion
'''''''''''''''

For all of the input and output dataframes (such as ``u_nk``, ``dHdl``,
``Estimator.delta_f_``, ``Estimator.d_delta_f_``), the :ref:`metadata <metadata>`
is stored as
:attr:`pandas.DataFrame.attrs`. The unit of the data could be converted to
:math:`kT`, :math:`kJ/mol` or :math:`kcal/mol` via the
:func:`~alchemlyb.postprocessors.units.to_kT`,
:func:`~alchemlyb.postprocessors.units.to_kJmol`,
:func:`~alchemlyb.postprocessors.units.to_kcalmol`. A dispatch table
(:func:`~alchemlyb.postprocessors.units.get_unit_converter`) is also to
provide the relevant converter. ::

    >>> from alchemlyb.postprocessors.units import get_unit_converter
    >>> get_unit_converter('kT')
    <function to_kT>
    >>> get_unit_converter('kJ/mol')
    <function to_kJmol>
    >>> get_unit_converter('kcal/mol')
    <function to_kcalmol>

Unit Conversion Functions
-------------------------
.. currentmodule:: alchemlyb.postprocessors

.. autosummary::
    :toctree: postprocessors

    units

