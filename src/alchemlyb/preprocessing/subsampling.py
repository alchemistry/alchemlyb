"""Functions for subsampling datasets.

"""
import numpy as np
from pymbar.timeseries import (statisticalInefficiency,
                               detectEquilibration,
                               subsampleCorrelatedData, )


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


def statistical_inefficiency(df, series=None, lower=None, upper=None, step=None,
                             conservative=True):
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
    conservative : bool
        ``True`` use ``ceil(statistical_inefficiency)`` to slice the data in uniform
        intervals (the default). ``False`` will sample at non-uniform intervals to
        closely match the (fractional) statistical_inefficieny, as implemented
        in :func:`pymbar.timeseries.subsampleCorrelatedData`.

    Returns
    -------
    DataFrame
        `df` subsampled according to subsampled `series`.

    Warning
    -------
    The `series` and the data to be sliced, `df`, need to have the same number
    of elements because the statistical inefficiency is calculated based on
    the index of the series (and not an associated time). At the moment there is
    no automatic conversion from a time to an index.

    Note
    ----
    For a non-integer statistical ineffciency :math:`g`, the default value
    ``conservative=True`` will provide _fewer_ data points than allowed by
    :math:`g` and thus error estimates will be _higher_. For large numbers of
    data points and converged free energies, the choice should not make a
    difference. For small numbers of data points, ``conservative=True``
    decreases a false sense of accuracy and is deemed the more careful and
    conservative approach.

    See Also
    --------
    pymbar.timeseries.statisticalInefficiency : detailed background
    pymbar.timeseries.subsampleCorrelatedData : used for subsampling


    .. versionchanged:: 0.2.0
       The ``conservative`` keyword was added and the method is now using
       ``pymbar.timeseries.statisticalInefficiency()``; previously, the statistical
       inefficiency was _rounded_ (instead of ``ceil()``) and thus one could
       end up with correlated data.

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

        if (len(series) != len(df) or
            not all(series.reset_index()['time'] == df.reset_index()['time'])):
            raise ValueError("series and data must be sampled at the same times")

        # calculate statistical inefficiency of series (could use fft=True but needs test)
        statinef  = statisticalInefficiency(series, fast=False)

        # use the subsampleCorrelatedData function to get the subsample index
        indices = subsampleCorrelatedData(series, g=statinef,
                                          conservative=conservative)
        df = df.iloc[indices]
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
