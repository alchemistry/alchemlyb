The base workflow
=================
.. automodule:: alchemlyb.workflows.base

The :class:`alchemlyb.workflows.base.WorkflowBase` class provides a
basic API template for the workflow development.  The workflow should
be able to run in an automatic fashion. ::

    >>> from alchemlyb.workflows.base import WorkflowBase
    >>> workflow = WorkflowBase(units='kT', software='Gromacs', T=298,
        out='./', *args, **kwargs)
    >>> workflow.run(*args, **kwargs)

Three main functions are provided such that the workflow could be run in a
step-by-step fashion. ::

    >>> from alchemlyb.workflows.base import WorkflowBase
    >>> workflow = WorkflowBase(units='kT', software='Gromacs', T=298,
        out='./', *args, **kwargs)
    >>> workflow.read(*args, **kwargs)
    >>> workflow.preprocess(*args, **kwargs)
    >>> workflow.estimate(*args, **kwargs)
    >>> workflow.check_convergence(*args, **kwargs)
    >>> workflow.plot(*args, **kwargs)


API Reference
-------------
.. currentmodule:: alchemlyb.workflows.base

.. autoclass:: alchemlyb.workflows.base.WorkflowBase
    :members:
    :inherited-members:
