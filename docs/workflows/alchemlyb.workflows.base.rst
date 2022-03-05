The base workflow
=================

The :class:`alchemlyb.workflows.base.WorkflowBase` provides a basic API
template for the workflow development.
The workflow should be able to run in an automatic fashion. ::

    >>> from alchemlyb.workflows.base import WorkflowBase
    >>> workflow = WorkflowBase()
    >>> workflow.run()

Three main functions are provided such that the workflow could be run in a
step-by-step fashion. ::

    >>> from alchemlyb.workflows.base import WorkflowBase
    >>> workflow = WorkflowBase()
    >>> workflow.read()
    >>> workflow.subsample()
    >>> workflow.estimate()
    >>> workflow.convergence()
    >>> workflow.plot()

API Reference
-------------
.. autoclass:: alchemlyb.workflows.base.WorkflowBase
    :members:
    :inherited-members:
