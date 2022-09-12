Automatic workflow
==================
Though **alchemlyb** is a library offering great flexibility in deriving free
energy estimate, it also provides workflows that provides automatic analysis
of the results and step-by-step version that allows more flexibility.

For developers, the skeleton of the workflow should follow the example in
:class:`alchemlyb.workflows.base.WorkflowBase`.

For users, **alchemlyb** offers a workflow :class:`alchemlyb.workflows.ABFE`
similar to
`Alchemical Analysis <https://github.com/MobleyLab/alchemical-analysis>`_
for doing automatic absolute binding free energy (ABFE) analysis.

.. currentmodule:: alchemlyb.workflows

.. autosummary::
    :toctree: workflows

    base
    ABFE

