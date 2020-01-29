"""Functions for subsampling datasets.

"""

from collections import defaultdict
import numpy as np
import pandas as pd
from pymbar import timeseries


def _check_multiple_times(df):
    return df.sort_index(0).reset_index(0).duplicated('time').any()


def slicing(df, lower=None, upper=None, step=None, force=False):
    """Subsample an alchemlyb DataFrame using slicing on the outermost index (time).

    Slicing will be performed separately on groups of rows corresponding to
    each set of lambda values present in the DataFrame's index. Each group will
    be sorted on the outermost (time) index prior to slicing.

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
    # we always start with a full index sort on the whole dataframe
    df = df.sort_index()

    index_names = list(df.index.names[1:])
    resdf = list()

    for name, group in df.groupby(level=index_names):
        group_s = group.loc[lower:upper:step]

        if not force and _check_multiple_times(group_s):
            raise KeyError("Duplicate time values found; it's generally advised "
                           "to use slicing on DataFrames with unique time values "
                           "for each row. Use `force=True` to ignore this error.")

        resdf.append(group_s)

    return pd.concat(resdf)


def statistical_inefficiency(df, column, lower=None, upper=None, step=None,
                             conservative=True, return_calculated=False, force=False):
    """Subsample an alchemlyb DataFrame based on the calculated statistical inefficiency
    of one of its columns.

    Calculation of statistical inefficiency and subsequent subsampling will be
    performed separately on groups of rows corresponding to each set of lambda
    values present in the DataFrame's index. Each group will be sorted on the
    outermost (time) index prior to any calculation.

    Parameters
    ----------
    df : DataFrame
        DataFrame to subsample according statistical inefficiency of `series`.
    column : label
        Label of column to use for calculating statistical inefficiency.
    lower : float
        Lower time to pre-slice data from.
    upper : float
        Upper time to pre-slice data to (inclusive).
    step : int
        Step between rows to pre-slice by.
    conservative : bool
        ``True`` use ``ceil(statistical_inefficiency)`` to slice the data in uniform
        intervals (the default). ``False`` will sample at non-uniform intervals to
        closely match the (fractional) statistical_inefficieny, as implemented
        in :func:`pymbar.timeseries.subsampleCorrelatedData`.
    return_calculated : bool
        ``True`` return a tuple, with the second item a dict giving, e.g. `statinef`
        for each group.
    force : bool
        Ignore checks that DataFrame is in proper form for expected behavior.

    Returns
    -------
    DataFrame
        `df` subsampled according to subsampled `column`.

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


    .. versionchanged:: 0.4.0
       The ``series`` keyword was replaced with the ``column`` keyword.
       The function no longer takes an arbitrary series as input for
       calculating statistical inefficiency.

    .. versionchanged:: 0.2.0
       The ``conservative`` keyword was added and the method is now using
       ``pymbar.timeseries.subsampleCorrelatedData()``; previously, the statistical
       inefficiency was _rounded_ (instead of ``ceil()``) and thus one could
       end up with correlated data.

    """
    # we always start with a full index sort on the whole dataframe
    df = df.sort_index()

    index_names = list(df.index.names[1:])
    resdf = list()

    if return_calculated:
        calculated = defaultdict(dict)

    for name, group in df.groupby(level=index_names):
            
        group_s = slicing(group, lower=lower, upper=upper, step=step)

        if not force and _check_multiple_times(group):
            raise KeyError("Duplicate time values found; statistical inefficiency"
                           "is only meaningful for a single, contiguous, "
                           "and sorted timeseries.")

        # calculate statistical inefficiency of column (could use fft=True but needs test)
        statinef = timeseries.statisticalInefficiency(group_s[column], fast=False)

        # use the subsampleCorrelatedData function to get the subsample index
        indices = timeseries.subsampleCorrelatedData(group_s[column], g=statinef,
                                      conservative=conservative)

        resdf.append(group_s.iloc[indices])

        if return_calculated:
            calculated['statinef'][name] = statinef
    
    if return_calculated:
        return pd.concat(resdf), calculated
    else:
        return pd.concat(resdf)


def equilibrium_detection(df, column, lower=None, upper=None, step=None,
                          conservative=True, return_calculated=False, force=False):
    """Subsample a DataFrame using automated equilibrium detection on one of
    its columns.

    Equilibrium detection and subsequent subsampling will be performed
    separately on groups of rows corresponding to each set of lambda values
    present in the DataFrame's index. Each group will be sorted on the
    outermost (time) index prior to any calculation.

    Parameters
    ----------
    df : DataFrame
        DataFrame to subsample according to equilibrium detection on `series`.
    column : label
        Label of column to use for equilibrium detection.
    lower : float
        Lower time to pre-slice data from.
    upper : float
        Upper time to pre-slice data to (inclusive).
    step : int
        Step between rows to pre-slice by.
    conservative : bool
        ``True`` use ``ceil(statistical_inefficiency)`` to slice the data in uniform
        intervals (the default). ``False`` will sample at non-uniform intervals to
        closely match the (fractional) statistical_inefficieny, as implemented
        in :func:`pymbar.timeseries.subsampleCorrelatedData`.
    return_calculated : bool
        ``True`` return a tuple, with the second item a dict giving, e.g. `statinef`
        for each group.
    force : bool
        Ignore checks that DataFrame is in proper form for expected behavior.

    Returns
    -------
    DataFrame
        `df` subsampled according to subsampled `column`.

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
    pymbar.timeseries.detectEquilibration : detailed background
    pymbar.timeseries.subsampleCorrelatedData : used for subsampling


    .. versionchanged:: 0.4.0
       The ``series`` keyword was replaced with the ``column`` keyword.
       The function no longer takes an arbitrary series as input for
       calculating statistical inefficiency.

    .. versionchanged:: 0.4.0
       The ``conservative`` keyword was added and the method is now using
       ``pymbar.timeseries.subsampleCorrelatedData()``; previously, the statistical
       inefficiency was _rounded_ (instead of ``ceil()``) and thus one could
       end up with correlated data.

    """
    # we always start with a full index sort on the whole dataframe
    df = df.sort_index()

    index_names = list(df.index.names[1:])
    resdf = list()

    if return_calculated:
        calculated = defaultdict(dict)

    for name, group in df.groupby(level=index_names):
        group_s = slicing(group, lower=lower, upper=upper, step=step)

        if not force and _check_multiple_times(group):
            raise KeyError("Duplicate time values found; equilibrium detection "
                           "is only meaningful for a single, contiguous, "
                           "and sorted timeseries.")

        # calculate statistical inefficiency of series, with equilibrium detection
        t, statinef, Neff_max  = timeseries.detectEquilibration(group_s[column])

        # only keep values past `t`
        group_s = group_s.iloc[t:]

        # use the subsampleCorrelatedData function to get the subsample index
        indices = timeseries.subsampleCorrelatedData(group_s[column], g=statinef,
                                      conservative=conservative)

        resdf.append(group_s.iloc[indices])

        if return_calculated:
            calculated['t'][name] = statinef
            calculated['statinef'][name] = statinef
            calculated['Neff_max'][name] = statinef

    if return_calculated:
        return pd.concat(resdf), calculated
    else:
        return pd.concat(resdf)
