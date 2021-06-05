
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

def pass_attrs(func):
    '''Pass the attributes from the input dataframe to the output dataframe'''
    def wrapper(input_dataframe, *args,**kwargs):
        dataframe = func(input_dataframe, *args,**kwargs)
        dataframe.attrs = input_dataframe.attrs
        return dataframe
    return wrapper
