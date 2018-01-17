Amber parsing
=============
.. automodule:: alchemlyb.parsing.amber

The parsers featured in this module are constructed to properly parse Amber_ output files containing derivatives of the Hamiltonian. At the moment, only derivatives are available for TI-based estimators.

.. TODO
   
   Notes on what options need to be set in Amber to produce the required output.  See the Gromacs parser page for an example of the information that we would like to have here.


.. _`Amber`: http://ambermd.org/


API Reference
-------------
This submodule includes these parsing functions:

.. autofunction:: alchemlyb.parsing.amber.extract_dHdl
.. autofunction:: alchemlyb.parsing.amber.extract_u_nk
