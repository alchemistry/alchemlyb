"""Functions for subsampling datasets.

"""

import warnings

import pandas as pd
from pymbar.timeseries import detect_equilibration as _detect_equilibration
from pymbar.timeseries import statistical_inefficiency as _statistical_inefficiency
from pymbar.timeseries import subsample_correlated_data as _subsample_correlated_data
from loguru import logger

from .. import pass_attrs


def decorrelate_u_nk(
    df, method="dE", drop_duplicates=True, sort=True, remove_burnin=False, **kwargs
):
    """Subsample an u_nk DataFrame based on the selected method.

    The method can be either 'all' (obtained as a sum over all energy
    components) or 'dE'. In the latter case the energy differences
    :math:`dE_{i,i+1}` (:math:`dE_{i,i-1}` for the last lambda) are used.  This
    is a wrapper function around the function
    :func:`~alchemlyb.preprocessing.subsampling.statistical_inefficiency` or
    :func:`~alchemlyb.preprocessing.subsampling.equilibrium_detection`.

    Parameters
    ----------
    df : DataFrame
        DataFrame to be subsampled according to the selected method.
    method : {'all', 'dE'}
        Method for decorrelating the data.
    drop_duplicates : bool
        Drop the duplicated lines based on time.
    sort : bool
        Sort the Dataframe based on the time column.
    remove_burnin : bool
        Whether to perform equilibrium detection (``True``) or just do
        statistical inefficiency (``False``).

        .. versionadded:: 1.0.0

    **kwargs :
        Additional keyword arguments for
        :func:`~alchemlyb.preprocessing.subsampling.statistical_inefficiency`
        or :func:`~alchemlyb.preprocessing.subsampling.equilibrium_detection`.
    Returns
    -------
    DataFrame
        `df` subsampled according to selected `method`.
    Note
    ----
    The default of ``True`` for  `drop_duplicates` and `sort` should result in robust decorrelation
    but can lose data.


    .. versionadded:: 0.6.0
    .. versionchanged:: 1.0.0
       Add the `remove_burnin` keyword to allow unequilibrated frames
       to be removed. Rename `method` value 'dhdl_all' to 'all' and
       deprecate the 'dhdl'.

    """
    kwargs["drop_duplicates"] = drop_duplicates
    kwargs["sort"] = sort

    series = u_nk2series(df, method)

    if remove_burnin:
        return equilibrium_detection(df, series, **kwargs)
    else:
        return statistical_inefficiency(df, series, **kwargs)


def decorrelate_dhdl(
    df, drop_duplicates=True, sort=True, remove_burnin=False, **kwargs
):
    """Subsample a dhdl DataFrame.
    This is a wrapper function around the function
    :func:`~alchemlyb.preprocessing.subsampling.statistical_inefficiency` and
    :func:`~alchemlyb.preprocessing.subsampling.equilibrium_detection`.

    Parameters
    ----------
    df : DataFrame
        DataFrame to subsample according to the selected method.
    drop_duplicates : bool
        Drop the duplicated lines based on time.
    sort : bool
        Sort the Dataframe based on the time column.
    remove_burnin : bool
        Whether to perform equilibrium detection (``True``) or just do
        statistical inefficiency (``False``).

        .. versionadded:: 1.0.0

    **kwargs :
        Additional keyword arguments for
        :func:`~alchemlyb.preprocessing.subsampling.statistical_inefficiency`
        or :func:`~alchemlyb.preprocessing.subsampling.equilibrium_detection`.

    Returns
    -------
    DataFrame
        `df` subsampled.

    Note
    ----
    The default of ``True`` for `drop_duplicates` and `sort` should result in
    robust decorrelation but can loose data.


    .. versionadded:: 0.6.0
    .. versionchanged:: 1.0.0
       Add the `remove_burnin` keyword to allow unequilibrated frames to be
       removed.

    """

    kwargs["drop_duplicates"] = drop_duplicates
    kwargs["sort"] = sort

    series = dhdl2series(df)

    if remove_burnin:
        return equilibrium_detection(df, series, **kwargs)
    else:
        return statistical_inefficiency(df, series, **kwargs)


@pass_attrs
def u_nk2series(df, method="dE"):
    """Convert an u_nk DataFrame into a series based on the selected method
    for subsampling.

    The method can be either 'all' (obtained as a sum over all energy
    components) or 'dE'. In the latter case the energy differences
    :math:`dE_{i,i+1}` (:math:`dE_{i,i-1}` for the last lambda) are used.

    Parameters
    ----------
    df : DataFrame
        DataFrame to be converted according to the selected method.
    method : {'all', 'dE'}
        Method for converting the data.

    Returns
    -------
    Series
        `series` to be used as input for
        :func:`~alchemlyb.preprocessing.subsampling.statistical_inefficiency`
        or
        :func:`~alchemlyb.preprocessing.subsampling.equilibrium_detection`.


    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.1
       The `dE` method computes the difference between the current lambda
       and the next lambda (previous lambda for the last window), instead
       of using the next lambda or the previous lambda for the last window.

    """

    # deprecation: remove in 3.0.0
    # (the deprecations should show up in the calling functions)
    if method == "dhdl":
        warnings.warn(
            "Method 'dhdl' has been deprecated, using 'dE' instead. "
            "'dhdl' will be removed in alchemlyb 3.0.0.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        method = "dE"
    elif method == "dhdl_all":
        warnings.warn(
            "Method 'dhdl_all' has been deprecated, using 'all' instead. "
            "'dhdl_all' will be removed in alchemlyb 3.0.0.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        method = "all"

    # Check if the input is u_nk
    try:
        key = df.index.values[0][1:]
        if len(key) == 1:
            key = key[0]
        df[key]
    except KeyError:
        raise ValueError("The input should be u_nk")

    if method == "all":
        series = df.sum(axis=1)
    elif method == "dE":
        # Using the same logic as alchemical-analysis
        key = df.index.values[0][1:]
        if len(key) == 1:
            # For the case where there is a single lambda
            index = df.columns.values.tolist().index(key[0])
        else:
            # For the case of more than 1 lambda
            index = df.columns.values.tolist().index(key)
            # for the state that is not the last state, take the state+1
        current_lambda = df.iloc[:, index]
        if index + 1 < len(df.columns):
            new_lambda = df.iloc[:, index + 1]
            # for the state that is the last state, take the state-1
        else:
            new_lambda = df.iloc[:, index - 1]
        series = new_lambda - current_lambda
    else:
        raise ValueError("Decorrelation method {} not found.".format(method))
    return series


@pass_attrs
def dhdl2series(df, method="all"):
    """Convert a dhdl DataFrame to a series for subsampling.

    The series is generated by summing over all energy components (axis 1 of
    `df`), as for ``method='all'`` in :func:`u_nk2series`. Commonly, `df` only
    contains a single energy component but in some cases (such as using a split
    protocol in GROMACS), it can contain multiple columns for different energy
    terms.

    Parameters
    ----------
    df : DataFrame
        DataFrame to subsample according to the selected method.
    method : 'all'
        Only 'all' is available; the keyword is provided for compatibility with
        :func:`u_nk2series`.

    Returns
    -------
    Series
        `series` to be used as input for
        :func:`~alchemlyb.preprocessing.subsampling.statistical_inefficiency`
        or
        :func:`~alchemlyb.preprocessing.subsampling.equilibrium_detection`.


    .. versionadded:: 1.0.0

    """
    if method != "all":
        raise ValueError("Only method='all' is supported for dhdl2series().")
    series = df.sum(axis=1)
    return series


def _check_multiple_times(df):
    if isinstance(df, pd.Series):
        return (
            df.sort_index(axis=0).reset_index("time", name="").duplicated("time").any()
        )
    else:
        return df.sort_index(axis=0).reset_index("time").duplicated("time").any()


def _check_sorted(df):
    return df.reset_index(0)["time"].is_monotonic_increasing


def _drop_duplicates(df, series=None):
    """Drop the duplication in the ``df`` which could be Dataframe or
    Series, if series is provided, format the series such that it has the
    same length as ``df``.

    Parameters
    ----------
    df : DataFrame or Series
        DataFrame or Series where duplication will be dropped.
    series : Series
        series to be formatted in the same way as df.

    Returns
    -------
    df : DataFrame or Series
        Formatted DataFrame or Series.
    series : Series
        Formatted Series.
    """
    if isinstance(df, pd.Series):
        # remove the duplicate based on time
        drop_duplicates_series = df.reset_index("time", name="").drop_duplicates("time")
        # Rest the time index
        lambda_names = [
            "time",
        ]
        lambda_names.extend(drop_duplicates_series.index.names)
        df = drop_duplicates_series.set_index("time", append=True).reorder_levels(
            lambda_names
        )
    else:
        # remove the duplicate based on time
        drop_duplicates_df = df.reset_index("time").drop_duplicates("time")
        # Rest the time index
        lambda_names = [
            "time",
        ]
        lambda_names.extend(drop_duplicates_df.index.names)
        df = drop_duplicates_df.set_index("time", append=True).reorder_levels(
            lambda_names
        )

    # Do the same withing with the series
    if series is not None:
        # remove the duplicate based on time
        drop_duplicates_series = series.reset_index("time", name="").drop_duplicates(
            "time"
        )
        # Rest the time index
        lambda_names = [
            "time",
        ]
        lambda_names.extend(drop_duplicates_series.index.names)
        series = drop_duplicates_series.set_index("time", append=True).reorder_levels(
            lambda_names
        )
    return df, series


def _sort_by_time(df, series=None):
    """Sort the ``df`` by time which could be Dataframe or
    Series, if series is provided, sort the series as well.

    Parameters
    ----------
    df : DataFrame or Series
        DataFrame or Series to be sorted by time.
    series : Series
        series to be sorted by time.

    Returns
    -------
    df : DataFrame or Series
        Formatted DataFrame or Series.
    series : Series
        Formatted Series.
    """
    df = df.sort_index(level="time")

    if series is not None:
        series = series.sort_index(level="time")
    return df, series


def _prepare_input(df, series, drop_duplicates, sort):
    """Prepare and check the input to be used for statistical_inefficiency or equilibrium_detection.

    Parameters
    ----------
    df : DataFrame or Series
        DataFrame or Series to be Prepared and checked.
    series : Series
        series to be Prepared and checked.

    Returns
    -------
    df : DataFrame or Series
        Formatted DataFrame or Series.
    series : Series
        Formatted Series.
    """
    if series is None:
        warnings.warn(
            "The series input is `None`, would not subsample according to statistical inefficiency."
        )

    elif len(df) != len(series):
        raise ValueError(
            f"The length of df ({len(df)}) should be same as the length of series ({len(series)})."
        )
    if _check_multiple_times(df):
        if drop_duplicates:
            df, series = _drop_duplicates(df, series)
        else:
            raise KeyError(
                "Duplicate time values found; statistical inefficiency "
                "only works on a single, contiguous, "
                "and sorted timeseries."
            )

    if not _check_sorted(df):
        if sort:
            df, series = _sort_by_time(df, series)
        else:
            raise KeyError(
                "Statistical inefficiency only works as expected if "
                "values are sorted by time, increasing."
            )

    if series is not None:
        if len(series) != len(df) or not all(
            series.reset_index()["time"] == df.reset_index()["time"]
        ):
            raise ValueError("series and data must be sampled at the same times")
    return df, series


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


    .. versionchanged:: 1.0.1
       The rows with NaN values are not dropped by default.
    """
    try:
        df = df.loc[lower:upper:step]
    except KeyError:
        raise KeyError("DataFrame rows must be sorted by time, increasing.")

    if not force and _check_multiple_times(df):
        raise KeyError(
            "Duplicate time values found; it's generally advised "
            "to use slicing on DataFrames with unique time values "
            "for each row. Use `force=True` to ignore this error."
        )

    return df


def statistical_inefficiency(
    df,
    series=None,
    lower=None,
    upper=None,
    step=None,
    conservative=True,
    drop_duplicates=False,
    sort=False,
):
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
        in :func:`pymbar.timeseries.subsample_correlated_data`.
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
    pymbar.timeseries.statistical_inefficiency : detailed background
    pymbar.timeseries.subsample_correlated_data : used for subsampling


    .. versionchanged:: 0.2.0
       The ``conservative`` keyword was added and the method is now using
       ``pymbar.timeseries.statistical_inefficiency()``; previously, the statistical
       inefficiency was _rounded_ (instead of ``ceil()``) and thus one could
       end up with correlated data.

    .. versionchanged:: 1.0.0
       Fixed a bug that would effectively ignore the ``lower`` and ``step``
       keywords when returning the subsampled DataFrame object. See
       `issue #198 <https://github.com/alchemistry/alchemlyb/issues/198>`_ for
       more details.

    """
    df, series = _prepare_input(df, series, drop_duplicates, sort)

    if series is not None:
        series = slicing(series, lower=lower, upper=upper, step=step)
        df = slicing(df, lower=lower, upper=upper, step=step)

        # calculate statistical inefficiency of series (could use fft=True but needs test)
        logger.debug("Running statistical inefficiency analysis.")
        statinef = _statistical_inefficiency(series)
        logger.debug("Statistical inefficiency: {:.2f}.", statinef)

        # use the subsample_correlated_data function to get the subsample index
        indices = _subsample_correlated_data(
            series, g=statinef, conservative=conservative
        )
        logger.debug("Number of uncorrelated samples: {}.", len(indices))
        df = df.iloc[indices]
    else:
        df = slicing(df, lower=lower, upper=upper, step=step)

    return df


def equilibrium_detection(
    df,
    series=None,
    lower=None,
    upper=None,
    step=None,
    drop_duplicates=False,
    sort=False,
):
    """Subsample a DataFrame using automated equilibrium detection on a timeseries.

    This function uses the :mod:`pymbar` implementation of the *simple
    automated equilibrium detection* algorithm in [Chodera2016]_.

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
    drop_duplicates : bool
        Drop the duplicated lines based on time.
    sort : bool
        Sort the Dataframe based on the time column.

    Returns
    -------
    DataFrame
        `df` subsampled according to subsampled `series`.

    Notes
    -----
    Please cite [Chodera2016]_ when you use this function in published work.

    See Also
    --------
    pymbar.timeseries.detect_equilibration : detailed background
    pymbar.timeseries.subsample_correlated_data : used for subsampling


    .. versionchanged:: 1.0.0
        Add the drop_duplicates and sort keyword to unify the behaviour between
        :func:`~alchemlyb.preprocessing.subsampling.statistical_inefficiency` or
        :func:`~alchemlyb.preprocessing.subsampling.equilibrium_detection`.

    """
    df, series = _prepare_input(df, series, drop_duplicates, sort)

    if series is not None:
        series = slicing(series, lower=lower, upper=upper, step=step)
        df = slicing(df, lower=lower, upper=upper, step=step)

        # calculate statistical inefficiency of series, with equilibrium detection
        logger.debug("Running equilibration detection.")
        t, statinef, Neff_max = _detect_equilibration(series.values)
        logger.debug("Start index: {}.", t)
        logger.debug("Statistical inefficiency: {:.2f}.", statinef)

        series_equil = series[t:]
        df_equil = df[t:]

        indices = _subsample_correlated_data(series_equil, g=statinef)
        logger.debug("Number of uncorrelated samples: {}.", len(indices))
        df = df_equil.iloc[indices]
    else:
        df = slicing(df, lower=lower, upper=upper, step=step)

    return df
