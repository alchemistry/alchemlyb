"""Functions for subsampling datasets.

"""
import numpy as np
from pymbar.timeseries import statisticalInefficiency
from pymbar.timeseries import detectEquilibration


def slicing(df, lower=None, upper=None, step=None):
    """Subsample a DataFrame using simple slicing.

    Parameters
    ----------
    df : DataFrame
        DataFrame to subsample.
    lower : float
        Lower bound to slice from.
    upper : float
        Upper bound to slice to (inclusive).
    step : int
        Step between rows to slice by.

    Returns
    -------
    DataFrame
        `df` subsampled.

    """
    df = df.loc[lower:upper:step]

    # drop any rows that have missing values
    df = df.dropna()

    # subsample according to step
    #df = df.iloc[::step]

    return df


def statistical_inefficiency(df, series=None, lower=None, upper=None, step=None):
    """Subsample a DataFrame based on the calculated statistical inefficiency
    of a timeseries.

    If `series` is ``None``, then this function will behave the same as
    :func:`slicing`.

    Parameters
    ----------
    df : DataFrame
        DataFrame to subsample according statistical inefficiency of `series`.
    series : Series
        Series to use for calculating statistical inefficiency. If ``None``,
        no statistical inefficiency-based subsampling will be performed.
    lower : float
        Lower bound to pre-slice `series` data from.
    upper : float
        Upper bound to pre-slice `series` to (inclusive).
    step : int
        Step between `series` items to pre-slice by.

    Returns
    -------
    DataFrame
        `df` subsampled according to subsampled `series`.

    See Also
    --------
    pymbar.timeseries.statisticalInefficiency : detailed background

    """
    if series is not None:
        series = series.loc[lower:upper]

        # drop any rows that have missing values
        series = series.dropna()

        # subsample according to step
        series = series.iloc[::step]

        # calculate statistical inefficiency of series
        statinef  = statisticalInefficiency(series)

        # we round up
        statinef = int(np.rint(statinef))

        # subsample according to statistical inefficiency
        series = series.iloc[::statinef]

        df = df.loc[series.index]
    else:
        df = slicing(df, lower=lower, upper=upper, step=step)
    
    return df


def equilibrium_detection(df, series=None, lower=None, upper=None, step=None):
    """Subsample a DataFrame using automated equilibrium detection on a timeseries.

    If `series` is ``None``, then this function will behave the same as
    :func:`slicing`.

    Parameters
    ----------
    df : DataFrame
        DataFrame to subsample according to equilibrium detection on `series`.
    series : Series
        Series to detect equilibration on. If ``None``, no equilibrium 
        detection-based subsampling will be performed.
    lower : float
        Lower bound to pre-slice `series` data from.
    upper : float
        Upper bound to pre-slice `series` to (inclusive).
    step : int
        Step between `series` items to pre-slice by.

    Returns
    -------
    DataFrame
        `df` subsampled according to subsampled `series`.

    See Also
    --------
    pymbar.timeseries.detectEquilibration : detailed background

    """
    if series is not None:
        series = series.loc[lower:upper]

        # drop any rows that have missing values
        series = series.dropna()

        # subsample according to step
        series = series.iloc[::step]

        # calculate statistical inefficiency of series, with equilibrium detection
        t, statinef, Neff_max  = detectEquilibration(series.values)

        # we round up
        statinef = int(np.rint(statinef))

        # subsample according to statistical inefficiency
        series = series.iloc[t::statinef]

        df = df.loc[series.index]
    else:
        df = slicing(df, lower=lower, upper=upper, step=step)
    
    return df
