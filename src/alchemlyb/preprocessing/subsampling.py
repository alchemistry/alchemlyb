"""Functions for subsampling datasets.

"""
import numpy as np
import pandas as pd
from pymbar.timeseries import (statisticalInefficiency,
                               detectEquilibration,
                               subsampleCorrelatedData, )

def decorrelate_u_nk(df, method='dhdl', drop_duplicates=True,
                     sort=True, **kwargs):
    """Subsample a u_nk DataFrame based on the selected method.

    The method can be either 'dhdl_all' (obtained as a sum over all energy
    components) or 'dhdl' (obtained as the energy components that are
    changing; default) or 'dE'. In the latter case the energy differences
    dE_{i,i+1} (dE_{i,i-1} for the last lambda) are used.

    This is a wrapper function around the function
    :func:`~alchemlyb.preprocessing.subsampling.statistical_inefficiency`.

    Parameters
    ----------
    df : DataFrame
        DataFrame to be subsampled according to the selected method.
    method : {'dhdl', 'dhdl_all', 'dE'}
        Method for decorrelating the data.
    drop_duplicates : bool
        Drop the duplicated lines based on time.
    sort : bool
        Sort the Dataframe based on the time column.
    **kwargs :
        Additional keyword arguments for
        :func:`~alchemlyb.preprocessing.subsampling.statistical_inefficiency`.

    Returns
    -------
    DataFrame
        `df` subsampled according to selected `method`.

    Note
    ----
    The default of ``True`` for  `drop_duplicates` and `sort` should result in robust decorrelation
    but can loose data.

    .. versionadded:: 0.6.0
    """
    kwargs['drop_duplicates'] = drop_duplicates
    kwargs['sort'] = sort

    # Check if the input is u_nk
    try:
        key = df.index.values[0][1:]
        if len(key) == 1:
            key = key[0]
        df[key]
    except KeyError:
        raise ValueError('The input should be u_nk')

    if method == 'dhdl':
        # Find the current column index
        # Select the first row and remove the first column (Time)
        key = df.index.values[0][1:]
        if len(key) > 1:
            # Multiple keys
            series = df[key]
        else:
            # Single key
            series = df[key[0]]
    elif method == 'dhdl_all':
        series = df.sum(axis=1)
    elif method == 'dE':
        # Using the same logic as alchemical-analysis
        key = df.index.values[0][1:]
        if len(key) == 1:
            # For the case where there is a single lambda
            index = df.columns.values.tolist().index(key[0])
        else:
            # For the case of more than 1 lambda
            index = df.columns.values.tolist().index(key)
            # for the state that is not the last state, take the state+1
        if index + 1 < len(df.columns):
            series = df.iloc[:, index + 1]
            # for the state that is the last state, take the state-1
        else:
            series = df.iloc[:, index - 1]
    else: # pragma: no cover
        raise ValueError(
            'Decorrelation method {} not found.'.format(method))
    return statistical_inefficiency(df, series, **kwargs)

def decorrelate_dhdl(df, drop_duplicates=True, sort=True, **kwargs):
    """Subsample a dhdl DataFrame.

    This is a wrapper function around the function
    :func:`~alchemlyb.preprocessing.subsampling.statistical_inefficiency`.

    Parameters
    ----------
    df : DataFrame
        DataFrame to subsample according to the selected method.
    drop_duplicates : bool
        Drop the duplicated lines based on time.
    sort : bool
        Sort the Dataframe based on the time column.
    **kwargs :
        Additional keyword arguments for
        :func:`~alchemlyb.preprocessing.subsampling.statistical_inefficiency`.

    Returns
    -------
    DataFrame
        `df` subsampled.

    Note
    ----
    The default of ``True`` for  `drop_duplicates` and `sort` should result in robust decorrelation
    but can loose data.


    .. versionadded:: 0.6.0
    """

    kwargs['drop_duplicates'] = drop_duplicates
    kwargs['sort'] = sort

    series = df.sum(axis=1)

    return statistical_inefficiency(df, series, **kwargs)

def _check_multiple_times(df):
    if isinstance(df, pd.Series):
        return df.sort_index(0).reset_index('time', name='').duplicated('time').any()
    else:
        return df.sort_index(0).reset_index('time').duplicated('time').any()


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
                             conservative=True, drop_duplicates=False, sort=False):
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
    drop_duplicates : bool
        Drop the duplicated lines based on time.
    sort : bool
        Sort the Dataframe based on the time column.

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

    .. versionchanged:: 0.7.0
       Fixed a bug that would effectively ignore the ``lower`` and ``step``
       keywords when returning the subsampled DataFrame object. See
       `issue #198 <https://github.com/alchemistry/alchemlyb/issues/198>`_ for 
       more details.

    """
    if _check_multiple_times(df):
        if drop_duplicates:
            if isinstance(df, pd.Series):
                # remove the duplicate based on time
                drop_duplicates_series = df.reset_index('time', name='').\
                    drop_duplicates('time')
                # Rest the time index
                lambda_names = ['time',]
                lambda_names.extend(drop_duplicates_series.index.names)
                df = drop_duplicates_series.set_index('time', append=True).\
                    reorder_levels(lambda_names)
            else:
                # remove the duplicate based on time
                drop_duplicates_df = df.reset_index('time').drop_duplicates('time')
                # Rest the time index
                lambda_names = ['time',]
                lambda_names.extend(drop_duplicates_df.index.names)
                df = drop_duplicates_df.set_index('time', append=True).\
                    reorder_levels(lambda_names)

            # Do the same withing with the series
            if series is not None:
                # remove the duplicate based on time
                drop_duplicates_series = series.reset_index('time', name='').\
                    drop_duplicates('time')
                # Rest the time index
                lambda_names = ['time',]
                lambda_names.extend(drop_duplicates_series.index.names)
                series = drop_duplicates_series.set_index('time', append=True).\
                    reorder_levels(lambda_names)

        else:
            raise KeyError("Duplicate time values found; statistical inefficiency "
                           "only works on a single, contiguous, "
                           "and sorted timeseries.")

    if not _check_sorted(df):
        if sort:
            df = df.sort_index(level='time')

            if series is not None:
                series = series.sort_index(level='time')
        else:
            raise KeyError("Statistical inefficiency only works as expected if "
                           "values are sorted by time, increasing.")

    if series is not None:
    
        if (len(series) != len(df) or
            not all(series.reset_index()['time'] == df.reset_index()['time'])):
            raise ValueError("series and data must be sampled at the same times")
               
        series = slicing(series, lower=lower, upper=upper, step=step)
        df = slicing(df, lower=lower, upper=upper, step=step)

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
