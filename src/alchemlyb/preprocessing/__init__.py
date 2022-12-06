"""
The :mod:`alchemlyb.preprocessing` module includes methods for subsampling and
preparing data for estimators.
"""

from .subsampling import equilibrium_detection
from .subsampling import (
    slicing,
    dhdl2series,
    u_nk2series,
    decorrelate_dhdl,
    decorrelate_u_nk,
)
from .subsampling import statistical_inefficiency

__all__ = [
    "slicing",
    "statistical_inefficiency",
    "equilibrium_detection",
    "decorrelate_dhdl",
    "decorrelate_u_nk",
    "dhdl2series",
    "u_nk2series",
]
