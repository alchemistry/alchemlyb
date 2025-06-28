"""
Unit conversion and constants
"""

from scipy.constants import R, calorie
import pandas as pd
from typing import Callable

#: conversion factor from kJ to kcal, based on :data:`scipy.constants.calorie`
#: in :mod:`scipy.constants`
kJ2kcal = 1 / calorie

#: gas constant :math:`R` in kJ/(mol K), based on :data:`scipy.constants.R`
#: in :mod:`scipy.constants`
R_kJmol = R / 1000


def to_kT(
    df: pd.DataFrame | pd.Series, T: None | float = None
) -> pd.DataFrame | pd.Series:
    """Convert the unit of a DataFrame to `kT`.

    Note that if entropy values are passed it is assumed that they are
    multiplied by the temperature, S * T.

    If temperature `T` is not provided, the DataFrame need to have attribute
    `temperature` and `energy_unit`. Otherwise, the temperature of the output
    dateframe will be set accordingly.

    Parameters
    ----------
    df : DataFrame
        DataFrame to convert unit.
    T : float
        Temperature (default: None) in Kelvin.

    Returns
    -------
    DataFrame
        `df` converted.
    """
    new_df = df.copy()
    if T is not None:
        new_df.attrs["temperature"] = T
    elif "temperature" not in df.attrs:
        raise TypeError("Attribute temperature not found in the input Dataframe.")

    if "energy_unit" not in df.attrs:
        raise TypeError("Attribute energy_unit not found in the input Dataframe.")

    if df.attrs["energy_unit"] == "kT":
        return new_df
    elif df.attrs["energy_unit"] == "kJ/mol":
        new_df /= R_kJmol * df.attrs["temperature"]
        new_df.attrs["energy_unit"] = "kT"
        return new_df  # type: ignore[no-any-return]
    elif df.attrs["energy_unit"] == "kcal/mol":
        new_df /= R_kJmol * df.attrs["temperature"] * kJ2kcal
        new_df.attrs["energy_unit"] = "kT"
        return new_df  # type: ignore[no-any-return]
    else:
        raise ValueError(
            "energy_unit {} can only be kT, kJ/mol or kcal/mol.".format(
                df.attrs["energy_unit"]
            )
        )


def to_kcalmol(
    df: pd.DataFrame | pd.Series, T: None | float = None
) -> pd.DataFrame | pd.Series:
    """Convert the unit of a DataFrame to kcal/mol.

    Note that if entropy values are passed, the result is S * T in units
    of kcal/mol.

    If temperature `T` is not provided, the DataFrame need to have attribute
    `temperature` and `energy_unit`. Otherwise, the temperature of the output
    dateframe will be set accordingly.

    Parameters
    ----------
    df : DataFrame
        DataFrame to convert unit.
    T : float
        Temperature (default: None).

    Returns
    -------
    DataFrame
        `df` converted.
    """
    kt_df = to_kT(df, T)
    kt_df *= R_kJmol * df.attrs["temperature"] * kJ2kcal
    kt_df.attrs["energy_unit"] = "kcal/mol"
    return kt_df  # type: ignore[no-any-return]


def to_kJmol(
    df: pd.DataFrame | pd.Series, T: None | float = None
) -> pd.DataFrame | pd.Series:
    """Convert the unit of a DataFrame to kJ/mol.

    Note that if entropy values are passed, the result is S * T in units
    of kJ/mol.

    If temperature `T` is not provided, the DataFrame need to have attribute
    `temperature` and `energy_unit`. Otherwise, the temperature of the output
    dateframe will be set accordingly.

    Parameters
    ----------
    df : DataFrame
        DataFrame to convert unit.
    T : float
        Temperature (default: None).

    Returns
    -------
    DataFrame
        `df` converted.
    """
    kt_df = to_kT(df, T)
    kt_df *= R_kJmol * df.attrs["temperature"]
    kt_df.attrs["energy_unit"] = "kJ/mol"
    return kt_df  # type: ignore[no-any-return]


def get_unit_converter(units: str) -> Callable:
    """Obtain the converter according to the unit string.

    If `units` is 'kT', the `to_kT` converter is returned. If `units` is
    'kJ/mol', the `to_kJmol` converter is returned. If `units` is 'kcal/mol',
    the `to_kcalmol` converter is returned.

    Parameters
    ----------
    units : str
        The unit that the function converts to.

    Returns
    -------
    func
        converter


    .. versionadded:: 0.5.0
    """
    converters = {"kT": to_kT, "kJ/mol": to_kJmol, "kcal/mol": to_kcalmol}
    try:
        convert = converters[units]
    except KeyError:
        raise ValueError(
            f"Energy unit {units} is not supported, "
            f"choose one of {list(converters.keys())}"
        )
    return convert
