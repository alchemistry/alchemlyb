from functools import wraps


def _init_attrs(func):
    """Add temperature to the parsed dataframe.

    The temperature is added to the dataframe as dataframe.attrs['temperature']
    and the energy unit is initiated as dataframe.attrs['energy_unit'] = 'kT'.
    """

    @wraps(func)
    def wrapper(outfile, T, *args, **kwargs):
        dataframe = func(outfile, T, *args, **kwargs)
        if dataframe is not None:
            dataframe.attrs["temperature"] = T
            dataframe.attrs["energy_unit"] = "kT"
        return dataframe

    return wrapper


def _init_attrs_dict(func):
    """Add temperature and energy units to the parsed dataframes.

    The temperature is added to the dataframe as dataframe.attrs['temperature']
    and the energy unit is initiated as dataframe.attrs['energy_unit'] = 'kT'.
    """

    @wraps(func)
    def wrapper(outfile, T, *args, **kwargs):
        dict_with_df = func(outfile, T, *args, **kwargs)
        for k in dict_with_df.keys():
            if dict_with_df[k] is not None:
                dict_with_df[k].attrs["temperature"] = T
                dict_with_df[k].attrs["energy_unit"] = "kT"
        return dict_with_df

    return wrapper
