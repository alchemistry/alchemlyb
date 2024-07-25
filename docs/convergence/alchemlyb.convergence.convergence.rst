Convergence API Reference
=========================
.. automodule:: alchemlyb.convergence.convergence

The :mod:`alchemlyb.convergence.convergence` module contains building blocks that perform a specific convergence analysis. They typically operate on lists of raw data and either run estimators on these data sets to obtain free energies as a function of the amount of data or they directly assess the convergence of the raw data.

.. Note::
   Read the original literature to learn the exact meaning of parameters and how to interpret the output of the convergence analysis.


All convergence functions are located in this submodule but for convenience they are also made available from :mod:`alchemlyb.convergence`, as shown here:

.. autofunction:: alchemlyb.convergence.forward_backward_convergence
      
.. autofunction:: alchemlyb.convergence.fwdrev_cumavg_Rc

.. autofunction:: alchemlyb.convergence.A_c		  

.. autofunction:: alchemlyb.convergence.moving_average
