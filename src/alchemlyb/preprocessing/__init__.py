"""
The :mod:`alchemlyb.preprocessing` module includes methods for subsampling and
preparing data for estimators.
"""

from .subsampling import slicing, decorrelate_dhdl, decorrelate_u_nk
from .subsampling import statistical_inefficiency
from .subsampling import equilibrium_detection

__all__ = [
    'slicing',
    'statistical_inefficiency',
    'equilibrium_detection',
    'decorrelate_dhdl',
    'decorrelate_u_nk'
]
