"""Functions for subsampling datasets.

"""
import numpy as np
from pymbar.timeseries import statisticalInefficiency
from pymbar.timeseries import detectEquilibration


def _check_multiple_times(df):
    return df.sort_index(0).reset_index(0).duplicated('time').any()


def _check_sorted(df):
    return df.reset_index(0)['time'].is_monotonic_increasing


def slicing(df, lower=None, upper=None, step=None, force=False):
    """Subsample a DataFrame using simple slicing.

    Parameters
    ----------
    df : DataFrame
        DataFrame to subsample.
    lower : float
        Lower time to slice from.
    upper : float
        Upper time to slice to (inclusive).
    step : int
        Step between rows to slice by.
    force : bool
        Ignore checks that DataFrame is in proper form for expected behavior.

    Returns
    -------
    DataFrame
        `df` subsampled.

    """
    try:
        df = df.loc[lower:upper:step]
    except KeyError:
        raise KeyError("DataFrame rows must be sorted by time, increasing.")

    if not force and _check_multiple_times(df):
        raise KeyError("Duplicate time values found; it's generally advised "
                       "to use slicing on DataFrames with unique time values "
                       "for each row. Use `force=True` to ignore this error.")

    # drop any rows that have missing values
    df = df.dropna()

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
    if _check_multiple_times(df):
        raise KeyError("Duplicate time values found; statistical inefficiency "
                       "only works on a single, contiguous, "
                       "and sorted timeseries.")

    if not _check_sorted(df):
        raise KeyError("Statistical inefficiency only works as expected if "
                       "values are sorted by time, increasing.")

    if series is not None:
        series = slicing(series, lower=lower, upper=upper, step=step)

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
    if _check_multiple_times(df):
        raise KeyError("Duplicate time values found; equilibrium detection "
                       "is only meaningful for a single, contiguous, "
                       "and sorted timeseries.")

    if not _check_sorted(df):
        raise KeyError("Equilibrium detection only works as expected if "
                       "values are sorted by time, increasing.")

    if series is not None:
        series = slicing(series, lower=lower, upper=upper, step=step)

        # calculate statistical inefficiency of series
        statinef  = statisticalInefficiency(series)

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
