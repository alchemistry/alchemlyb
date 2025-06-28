from functools import wraps
from typing import Any, Callable
import pandas as pd


def _init_attrs(
    func: Callable[..., None | pd.DataFrame],
) -> Callable[..., None | pd.DataFrame]:
    """Add temperature to the parsed dataframe.

    The temperature is added to the dataframe as dataframe.attrs['temperature']
    and the energy unit is initiated as dataframe.attrs['energy_unit'] = 'kT'.
    """

    @wraps(func)
    def wrapper(
        outfile: str, T: float, *args: Any, **kwargs: Any
    ) -> None | pd.DataFrame:
        dataframe = func(outfile, T, *args, **kwargs)
        if dataframe is not None:
            dataframe.attrs["temperature"] = T
            dataframe.attrs["energy_unit"] = "kT"
        return dataframe

    return wrapper


def _init_attrs_dict(
    func: Callable[..., dict[str, None | pd.DataFrame]],
) -> Callable[..., dict[str, None | pd.DataFrame]]:
    """Add temperature and energy units to the parsed dataframes.

    The temperature is added to the dataframe as dataframe.attrs['temperature']
    and the energy unit is initiated as dataframe.attrs['energy_unit'] = 'kT'.
    """

    @wraps(func)
    def wrapper(
        outfile: str, T: float, *args: Any, **kwargs: Any
    ) -> dict[str, None | pd.DataFrame]:
        dict_with_df = func(outfile, T, *args, **kwargs)
        for k in dict_with_df.keys():
            dataframe = dict_with_df[k]
            if dataframe is not None:
                dataframe.attrs["temperature"] = T
                dataframe.attrs["energy_unit"] = "kT"
        return dict_with_df

    return wrapper
