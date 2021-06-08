NAMD parsing
=============
.. automodule:: alchemlyb.parsing.namd

The parsers featured in this module are constructed to properly parse `NAMD`_ ``.fepout`` output files containing derivatives of the Hamiltonian and FEP (BAR) data.
See the NAMD documentation for the `theoretical backdrop <https://www.ks.uiuc.edu/Research/namd/2.13/ug/node60.html>`_ and `implementation details <https://www.ks.uiuc.edu/Research/namd/2.13/ug/node61.html>`_.

If you wish to use BAR on FEP data, be sure to provide the ``.fepout`` file from both the forward and reverse transformations.

After calling :meth:`~alchemlyb.parsing.namd.extract_u_nk` on the forward and reverse work values, these dataframes can be combined into one:

.. code-block:: python

    # replace zeroes in initial dataframe with nan
    u_nk_fwd.replace(0, np.nan, inplace=True)
    # replace the nan values with the reverse dataframe --
    # this should not overwrite any of the fwd work values 
    u_nk_fwd[u_nk_fwd.isnull()] = u_nk_rev
    # replace remaining nan values back to zero
    u_nk_fwd.replace(np.nan, 0, inplace=True)
    # sort final dataframe by `fep-lambda` (as opposed to `timestep`)
    u_nk = u_nk_fwd.sort_index(level=u_nk_fwd.index.names[1:])

The ``fep-lambda`` index states at which lambda this particular frame was sampled, whereas the columns are the evaluations of the Hamiltonian (or the potential energy U) at other lambdas (sometimes called "foreign lambdas").

.. _`NAMD`: http://www.ks.uiuc.edu/Research/namd/


API Reference
-------------
This submodule includes these parsing functions:

.. autofunction:: alchemlyb.parsing.namd.extract_u_nk
