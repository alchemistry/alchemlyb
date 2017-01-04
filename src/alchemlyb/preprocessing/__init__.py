"""
The :mod:`alchemlyb.preprocessing` module includes methods for subsampling and
preparing data for estimators.
"""

from .subsampling import slicing
from .subsampling import statistical_inefficiency
from .subsampling import equilibrium_detection

__all__ = [
    'slicing',
    'statistical_inefficiency',
    'equilibrium_detection',
]
