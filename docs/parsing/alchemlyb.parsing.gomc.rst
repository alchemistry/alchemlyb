GOMC parsing
==============
.. automodule:: alchemlyb.parsing.gomc

The parsers featured in this module are constructed to properly parse `GOMC <http://gomc.eng.wayne.edu/>`_ free energy output files, 
containing the Hamiltonian derivatives (:math:`\frac{dH}{d\lambda}`) for TI-based estimators and Hamiltonian differences (:math:`\Delta H` 
for all lambda states in the alchemical leg) for FEP-based estimators (BAR/MBAR).


API Reference
-------------
This submodule includes these parsing functions:

.. autofunction:: alchemlyb.parsing.gomc.extract_dHdl
.. autofunction:: alchemlyb.parsing.gomc.extract_u_nk
