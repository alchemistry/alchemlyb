def add_attr(func):
    '''Add temperature to the parsed dataframe.'''
    def wrapper(outfile, T):
        dataframe = func(outfile, T)
        if dataframe is not None:
            dataframe.attrs['temperature'] = T
            dataframe.attrs['energy_unit'] = 'kT'
        return dataframe
    return wrapper
