import pandas as pd
from warnings import warn
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

def pass_attrs(func):
    '''Pass the attrs from the first positional argument to the output
    dataframe.
 
 
    .. versionadded:: 0.5.0
 '''
    def wrapper(input_dataframe, *args,**kwargs):
        dataframe = func(input_dataframe, *args,**kwargs)
        dataframe.attrs = input_dataframe.attrs
        return dataframe
    return wrapper

def concat(objs, *args, **kwargs):
    '''Concatenate pandas objects while persevering the attrs.

    Concatenate pandas objects along a particular axis with optional set
    logic along the other axes. If all pandas objects have the same  attrs
    attribute, the new pandas objects would have this attrs attribute. A
    ValueError would be raised if any pandas object has a different attrs.

    Parameters
    ----------
    objs
        A sequence or mapping of Series or DataFrame objects.

    Returns
    -------
    DataFrame
        Concatenated pandas object.

    Raises
    ------
    ValueError
        If not all pandas objects have the same attrs.

    See Also
    --------
    pandas.concat
 
 
    .. versionadded:: 0.5.0'''
    warn('This method will be deprecated in 1.0.0. Use pd.concat instead.',
         DeprecationWarning,)
    return pd.concat(objs, *args, **kwargs)
