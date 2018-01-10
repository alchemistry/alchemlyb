Gromacs parsing
===============
.. automodule:: alchemlyb.parsing.gmx

The parsers featured in this module are constructed to properly parse XVG files containing Hamiltonian differences (for obtaining reduced potentials, :math:`u_{nk}`) and/or Hamiltonian derivatives (for obtaining gradients, :math:`\frac{dH}{d\lambda}`).
To produce such a file from an existing EDR energy file, use ``gmx energy -f <.edr> -odh dhdl.xvg`` with your installation of `Gromacs`_.

If you wish to use FEP-based estimators such as :class:`~alchemlyb.estimators.MBAR` that require reduced potentials for all lambda states in the alchemical leg, you will need to use these MDP options:

.. code-block:: none

    calc-lambda-neighbors = -1     ; calculate Delta H values for all other lambda windows
    dhdl-print-energy = potential  ; total potential energy of system included

In addition, the full set of lambda states for the alchemical leg should be explicitly specified in the ``fep-lambdas`` option (or ``coul-lambdas``, ``vdw-lambdas``, etc.), since this is what Gromacs uses to determine what lambda values to calculate :math:`\Delta H` values for.

To use TI-based estimators that require gradients, you will need to include these options:

.. code-block:: none

    dhdl-derivatives = yes         ; write derivatives of Hamiltonian with respect to lambda

Additionally, the parsers can properly parse XVG files (containing Hamiltonian differences and/or Hamiltonian derivatives) produced during expanded ensemble simulations. To produce such a file during the simulation, use ``gmx mdrun -deffnm <name> -dhdl dhdl.xvg`` with your installation of `Gromacs`_.
To run an expanded ensemble simulation you will need to use the following MDP option:

.. code-block:: none

    free_energy = expanded        ; turns on expanded ensemble simulation, lambda state becomes a dynamic variable

.. _`Gromacs`: http://www.gromacs.org/
.. _`MDP options`: 


API Reference
-------------
This submodule includes these parsing functions:

.. autofunction:: alchemlyb.parsing.gmx.extract_dHdl
.. autofunction:: alchemlyb.parsing.gmx.extract_u_nk
