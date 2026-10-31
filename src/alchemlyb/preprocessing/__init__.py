"""
The :mod:`alchemlyb.preprocessing` module includes methods for subsampling and
preparing data for estimators.
"""

from .subsampling import (
    decorrelate_dhdl,
    decorrelate_u_nk,
    dhdl2series,
    equilibrium_detection,
    slicing,
    statistical_inefficiency,
    u_nk2series,
)

__all__ = [
    "decorrelate_dhdl",
    "decorrelate_u_nk",
    "dhdl2series",
    "equilibrium_detection",
    "slicing",
    "statistical_inefficiency",
    "u_nk2series",
]
