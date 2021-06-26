from functools import wraps

def _init_attrs(func):
    '''Add temperature to the parsed dataframe.

    The temperature is added to the dataframe as dataframe.attrs['temperature']
    and the energy unit is initiated as dataframe.attrs['energy_unit'] = 'kT'.
    '''
    @wraps(func)
    def wrapper(outfile, T):
        dataframe = func(outfile, T)
        if dataframe is not None:
            dataframe.attrs['temperature'] = T
            dataframe.attrs['energy_unit'] = 'kT'
        return dataframe
    return wrapper
