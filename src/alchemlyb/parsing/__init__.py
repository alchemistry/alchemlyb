from functools import wraps

def _init_attrs(func):
    '''Add temperature to the parsed dataframe.

    The temperature is added to the dataframe as dataframe.attrs['temperature']
    and the energy unit is initiated as dataframe.attrs['energy_unit'] = 'kT'.
    '''
    @wraps(func)
    def wrapper(outfile, T, *args, **kwargs):
        results = func(outfile, T, *args, **kwargs)
        if results is not None:
            if isinstance(results, tuple):
                for item in results:
                    item.attrs['temperature'] = T
                    item.attrs['energy_unit'] = 'kT'
            else:
                results.attrs['temperature'] = T
                results.attrs['energy_unit'] = 'kT'
        return results

    return wrapper
