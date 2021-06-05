def add_attr(func):
    '''Pass the attributes from the input dataframe to the output dataframe'''
    def wrapper(outfile, T):
        dataframe = func(outfile, T)
        dataframe.attrs['temperature'] = T
        dataframe.attrs['energy_unit'] = 'kT'
        return dataframe
    return wrapper