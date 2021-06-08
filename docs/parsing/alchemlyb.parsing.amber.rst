Amber parsing
=============
.. automodule:: alchemlyb.parsing.amber

The parsers featured in this module are constructed to properly parse `Amber MD`_ output files containing derivatives of the Hamiltonian and FEP (BAR/MBAR) data.

.. _Amber MD: http://ambermd.org/

.. TODO
   
   Notes on what options need to be set in Amber to produce the required output.  See the Gromacs parser page for an example of the information that we would like to have here.



API Reference
-------------
This submodule includes these parsing functions:

.. autofunction:: alchemlyb.parsing.amber.extract_dHdl
.. autofunction:: alchemlyb.parsing.amber.extract_u_nk
