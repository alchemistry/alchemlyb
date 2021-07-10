alchemlyb.postprocessors.units
==============================

.. automodule:: alchemlyb.postprocessors.units

Some examples are given here to illustrate how to use the unit converter
functions to convert units. ::

    >>> import pandas as pd
    >>> import alchemlyb
    >>> from alchemtest.gmx import load_benzene
    >>> from alchemlyb.parsing.gmx import extract_u_nk
    >>> from alchemlyb.estimators import MBAR
    >>> from alchemlyb.postprocessors.units import to_kcalmol, to_kJmol, to_kT
    >>> bz = load_benzene().data
    >>> u_nk_coul = alchemlyb.concat([extract_u_nk(xvg, T=300) for xvg in bz['Coulomb']])
    >>> mbar_coul = MBAR().fit(u_nk_coul)
    >>> mbar_coul.delta_f_
              0.00      0.25      0.50      0.75      1.00
    0.00  0.000000  1.619069  2.557990  2.986302  3.041156
    0.25 -1.619069  0.000000  0.938921  1.367232  1.422086
    0.50 -2.557990 -0.938921  0.000000  0.428311  0.483165
    0.75 -2.986302 -1.367232 -0.428311  0.000000  0.054854
    1.00 -3.041156 -1.422086 -0.483165 -0.054854  0.000000
    >>> mbar_coul.delta_f_.attrs
    {'temperature': 300, 'energy_unit': 'kT'}

The default unit is in :math:`kT`, which could be changed to
:math:`kcal/mol`. ::

    >>> delta_f_ = to_kcalmol(mbar_coul.delta_f_)
    >>> delta_f_
              0.00      0.25      0.50      0.75      1.00
    0.00  0.000000  0.965228  1.524977  1.780319  1.813021
    0.25 -0.965228  0.000000  0.559749  0.815092  0.847794
    0.50 -1.524977 -0.559749  0.000000  0.255343  0.288045
    0.75 -1.780319 -0.815092 -0.255343  0.000000  0.032702
    1.00 -1.813021 -0.847794 -0.288045 -0.032702  0.000000
    >>> delta_f_.attrs
    {'temperature': 300, 'energy_unit': 'kcal/mol'}

.. autofunction:: alchemlyb.postprocessors.units.to_kcalmol

The unit could also be changed to :math:`kJ/mol`. ::

    >>> delta_f_ = to_kJmol(delta_f_)
    >>> delta_f_
              0.00      0.25      0.50      0.75      1.00
    0.00  0.000000  4.038508  6.380495  7.448848  7.585673
    0.25 -4.038508  0.000000  2.341987  3.410341  3.547165
    0.50 -6.380495 -2.341987  0.000000  1.068354  1.205178
    0.75 -7.448848 -3.410341 -1.068354  0.000000  0.136825
    1.00 -7.585673 -3.547165 -1.205178 -0.136825  0.000000
    >>> delta_f_.attrs
    {'temperature': 300, 'energy_unit': 'kJ/mol'}

.. autofunction:: alchemlyb.postprocessors.units.to_kJmol

And change back to :math:`kT` again. ::

    >>> delta_f_ = to_kT(delta_f_)
              0.00      0.25      0.50      0.75      1.00
    0.00  0.000000  1.619069  2.557990  2.986302  3.041156
    0.25 -1.619069  0.000000  0.938921  1.367232  1.422086
    0.50 -2.557990 -0.938921  0.000000  0.428311  0.483165
    0.75 -2.986302 -1.367232 -0.428311  0.000000  0.054854
    1.00 -3.041156 -1.422086 -0.483165 -0.054854  0.000000
    >>> delta_f_.attrs
    {'temperature': 300, 'energy_unit': 'kT'}

.. autofunction:: alchemlyb.postprocessors.units.to_kT

A dispatch table approach is also provided to return the relevant converter
for every units.

.. autofunction:: alchemlyb.postprocessors.units.get_unit_converter