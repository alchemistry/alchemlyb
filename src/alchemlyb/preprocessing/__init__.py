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

def pass_attrs(func):
    '''Pass the attributes from the input dataframe to the output dataframe'''
    def wrapper(input_dataframe, *args,**kwargs):
        dataframe = func(input_dataframe, *args,**kwargs)
        dataframe.attrs = input_dataframe.attrs
        return dataframe
    return wrapper
