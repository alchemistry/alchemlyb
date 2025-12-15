Convergence API Reference
=========================

The :mod:`alchemlyb.convergence.convergence` module contains building blocks that perform a specific convergence analysis. They typically operate on lists of raw data and either run estimators on these data sets to obtain free energies as a function of the amount of data or they directly assess the convergence of the raw data.

.. Note::
   Read the original literature to learn the exact meaning of parameters and how to interpret the output of the convergence analysis.


Functions in :mod:`alchemlyb.convergence`
-----------------------------------------

Important convergence functions are made available from :mod:`alchemlyb.convergence` for convenience:

.. autofunction:: alchemlyb.convergence.forward_backward_convergence
      
.. autofunction:: alchemlyb.convergence.fwdrev_cumavg_Rc

.. autofunction:: alchemlyb.convergence.A_c		  

.. autofunction:: alchemlyb.convergence.block_average


Functions in :mod:`alchemlyb.convergence.convergence`
-----------------------------------------------------

All convergence functions are located in the :mod:`alchemlyb.convergence.convergence` submodule but for convenience they are also made available from :mod:`alchemlyb.convergence`.

.. automodule:: alchemlyb.convergence.convergence
   :members:
   :show-inheritance:

