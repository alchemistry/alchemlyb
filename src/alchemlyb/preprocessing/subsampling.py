"""Functions for subsampling datasets.

"""
import random

from collections import defaultdict
import numpy as np
import pandas as pd
from pymbar import timeseries
from pymbar.utils import ParameterError


class CorrelationError(Exception):
    pass


def _check_multiple_times(data):
    return data.index.get_level_values(0).duplicated().any()


def _how_lr(name, group, direction):

    # first, get column index of column `name`, if it exists
    # if it does exist, increment or decrement position by one based on
    # `direction`
    try:
        pos = group.columns.get_loc(name)
    except KeyError:
        raise KeyError("No column with label '{}'".format(name))
    else:
        if direction == 'right':
            pos += 1
        elif direction == 'left':
            pos -= 1
        else:
            raise ValueError("`direction` must be either 'right' or 'left'")
    
    # handle cases where we are dealing with the leftmost column or rightmost
    # column
    if pos == -1:
        pos += 2
    elif pos == len(group.columns):
        pos -= 2
    elif (pos < -1) or (pos > len(group.columns)):
        raise IndexError("Position of selected column is outside of all expected bounds")

    return group[group.columns[pos]]


def _how_random(name, group, tried=None):

    candidates = set(group.columns) - (set(tried) | {name})

    if not candidates:
        raise CorrelationError("No column in the dataset could be used"
                " successfully for decorrelation")
    else:
        selection = random.choice(list(candidates))

    return group[selection], selection 


def _how_sum(name, group):
    return group.sum(axis=1)


def slicing(data, lower=None, upper=None, step=None, force=False):
    """Subsample an alchemlyb DataFrame using slicing on the outermost index (time).

    Slicing will be performed separately on groups of rows corresponding to
    each set of lambda values present in the DataFrame's index. Each group will
    be sorted on the outermost (time) index prior to slicing.

    Parameters
    ----------
    data : DataFrame
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
        `data` subsampled.

    """
    # we always start with a full index sort on the whole dataframe
    data = data.sort_index()

    index_names = list(data.index.names[1:])
    resdata = list()

    for name, group in data.groupby(level=index_names):
        group_s = group.loc[lower:upper:step]

        if not force and _check_multiple_times(group_s):
            raise KeyError("Duplicate time values found; it's generally advised "
                           "to use slicing on DataFrames with unique time values "
                           "for each row. Use `force=True` to ignore this error.")

        resdata.append(group_s)

    return pd.concat(resdata)


def _decorrelator(subsample, data, column=None, how=None, lower=None,
        upper=None, step=None, conservative=True, return_calculated=False,
        force=False, random_state=None):

    # input checks
    if column is not None:
        if isinstance(column, pd.Series):
            if not (data.index == column.index).all():
                raise ValueError("The index of `column` must match the index of `data` exactly"
                                 " to be used for decorrelation.")
        elif column in data.columns:
            pass
        else:
            raise ValueError("`column` must give either an existing column label in `data`"
                             " or a Series object with a matching index.")
    elif how is not None:
        pass
    else:
        raise ValueError("One of `column` or `how` must be set")

    np.random.seed(random_state)

    # we always start with a full index sort on the whole dataframe
    # should produce a copy
    data = data.sort_index()

    index_names = list(data.index.names[1:])
    resdata = list()

    if return_calculated:
        calculated = defaultdict(dict)

    def random_selection(name, group):
        tried = set()
        while True:
            group_c, selection = _how_random(name, group, tried=tried)
            try:
                indices, calculated = subsample(group_c, group)
            except:
                tried.add(selection)
            else:
                break

        return indices, calculated

    for name, group in data.groupby(level=index_names):

        if not force and _check_multiple_times(group):
            raise KeyError("Duplicate time values found; decorrelation "
                           "is only meaningful for a single, contiguous, "
                           "and sorted timeseries")

        if column is not None:
            if isinstance(column, pd.Series):
                group_c = column.groupby(level=index_names).get_group(name)
                indices, calc = subsample(group_c, group)
            elif column in group.columns:
                group_c = group[column]
                indices, calc = subsample(group_c, group)
        elif how is not None:
            if (how == 'right') or (how == 'left'):
                try:
                    group_c = _how_lr(name, group, how)
                except KeyError:
                    indices, calc = random_selection(name, group)
                else:
                    indices, calc = subsample(group_c, group)
            elif how == 'random':
                indices, calc = random_selection(name, group)
            elif how == 'sum':
                group_c = _how_sum(name, group)
                indices, calc = subsample(group_c, group)
            else:
                raise ValueError("`how` cannot be '{}';"
                " see docstring for available options".format(how))

        group_s = slicing(group, lower=lower, upper=upper, step=step)
        resdata.append(group_s.iloc[indices])

        if return_calculated:
            calculated[name] = calc
    
    if return_calculated:
        return pd.concat(resdata), pd.DataFrame(calculated)
    else:
        return pd.concat(resdata)


def statistical_inefficiency(data, column=None, how=None, lower=None,
        upper=None, step=None, conservative=True, return_calculated=False,
        force=False, random_state=None):
    """Subsample an alchemlyb DataFrame based on the calculated statistical
    inefficiency of one of its columns.

    Calculation of statistical inefficiency and subsequent subsampling will be
    performed separately on groups of rows corresponding to each set of lambda
    values present in the DataFrame's index. Each group will be sorted on the
    outermost (time) index prior to any calculation.

    The `how` parameter determines the observable used for calculating the
    correlations within each group of samples. The options are as follows:

        'right'
            The recommended setting for 'u_nk' datasets; the column immediately
            to the right of the column corresponding to the group's lambda
            index value is used. If there is no column to the right, then the
            column to the left is used.  If there is no column corresponding to
            the group's lambda index value, then 'random' is used (see below).
        'left'
            The opposite of the 'right' approach; the column immediately to the
            left of the column corresponding to the group's lambda index value
            is used. If there is no column to the left, then the column to the
            right is used.  If there is no column corresponding to the group's
            lambda index value, then 'random' is used for that group (see below).
        'random'
            A column is chosen uniformly at random from the set of columns
            available in the group, with the column corresponding to the
            group's lambda index value excluded, if present. If the correlation
            calculation fails, then another column is tried. This process
            continues until success or until all columns have been attempted
            without success (in which case, ``CorrelationError`` is raised).
        'sum'
            The recommended setting for 'dHdl' datasets; the columns are simply
            summed, and the resulting `Series` is used.

    Specifying the 'column' parameter overrides the behavior of 'how'. This
    allows the user to use a particular column or a specially-crafted `Series`
    for correlation calculation.

    Parameters
    ----------
    data : DataFrame
        DataFrame to subsample according statistical inefficiency of `series`.
    column : label or `pandas.Series`
        Label of column to use for calculating statistical inefficiency.
        Overrides `how`; can also take a `Series` object, but the index of the
        `Series` *must* match that of `data` exactly.
    how : {'right', 'left', 'random', 'sum'}
        The approach used to choose the observable on which correlations are
        calculated. See explanation above.
    lower : float
        Lower time to pre-slice data from.
    upper : float
        Upper time to pre-slice data to (inclusive).
    step : int
        Step between rows to pre-slice by.
    conservative : bool
        ``True`` use ``ceil(statistical_inefficiency)`` to slice the data in
        uniform intervals (the default). ``False`` will sample at non-uniform
        intervals to closely match the (fractional) statistical_inefficieny, as
        implemented in :func:`pymbar.timeseries.subsampleCorrelatedData`.
    return_calculated : bool
        ``True`` return a tuple, with the second item a dict giving, e.g. `statinef`
        for each group.
    force : bool
        Ignore checks that DataFrame is in proper form for expected behavior.
    random_state : int
        Integer between 0 and 2**32 -1 inclusive; fed to `numpy.random.seed`.
        Running this function on the same data with a specific random seed will
        produce the same result each time.

    Returns
    -------
    DataFrame
        `data` subsampled according to subsampled `column`.

    Raises
    ------
    CorrelationError
        If correlation removal fails irrecoverably.

    Note
    ----
    For a non-integer statistical ineffciency :math:`g`, the default value
    ``conservative=True`` will provide _fewer_ data points than allowed by
    :math:`g` and thus error estimates will be _higher_. For large numbers of
    data points and converged free energies, the choice should not make a
    difference. For small numbers of data points, ``conservative=True``
    decreases a false sense of accuracy and is deemed the more careful and
    conservative approach.

    References
    ----------
    [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
        histogram analysis method for the analysis of simulated and parallel tempering simulations.
        JCTC 3(1):26-41, 2007.

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
       ``pymbar.timeseries.subsampleCorrelatedData()``; previously, the
       statistical inefficiency was _rounded_ (instead of ``ceil()``) and thus
       one could end up with correlated data.

    """

    def subsample(group_c, group):
        group_cs = slicing(group_c, lower=lower, upper=upper, step=step)

        # calculate statistical inefficiency of column (could use fft=True but needs test)
        statinef = timeseries.statisticalInefficiency(group_cs, fast=False)

        # use the subsampleCorrelatedData function to get the subsample index
        indices = timeseries.subsampleCorrelatedData(group_cs, g=statinef,
                                      conservative=conservative)

        calculated = dict()
        calculated['statinef'] = statinef

        return indices, calculated

    return _decorrelator(
            subsample=subsample,
            data=data,
            column=column,
            how=how,
            lower=lower,
            upper=upper,
            step=step,
            conservative=conservative,
            return_calculated=return_calculated,
            force=force,
            random_state=random_state)
                        

def equilibrium_detection(data, column=None, how=None, lower=None,
        upper=None, step=None, conservative=True, return_calculated=False,
        force=False, random_state=None, nskip=1):
    """Subsample a DataFrame using automated equilibrium detection on one of
    its columns.

    Equilibrium detection and subsequent subsampling will be performed
    separately on groups of rows corresponding to each set of lambda values
    present in the DataFrame's index. Each group will be sorted on the
    outermost (time) index prior to any calculation.

    The `how` parameter determines the observable used for calculating the
    correlations within each group of samples. The options are as follows:

        'right'
            The recommended setting for 'u_nk' datasets; the column immediately
            to the right of the column corresponding to the group's lambda
            index value is used. If there is no column to the right, then the
            column to the left is used.  If there is no column corresponding to
            the group's lambda index value, then 'random' is used (see below).
        'left'
            The opposite of the 'right' approach; the column immediately to the
            left of the column corresponding to the group's lambda index value
            is used. If there is no column to the left, then the column to the
            right is used.  If there is no column corresponding to the group's
            lambda index value, then 'random' is used for that group (see below).
        'random'
            A column is chosen at random from the set of columns available in
            the group. If the correlation calculation fails, then another
            column is tried. This process continues until success or until all
            columns have been attempted without success.
        'sum'
            The recommended setting for 'dHdl' datasets; the columns are simply
            summed, and the resulting `Series` is used.

    Specifying the 'column' parameter overrides the behavior of 'how'. This
    allows the user to use a particular column or a specially-crafted `Series`
    for correlation calculation.

    Parameters
    ----------
    data : DataFrame
        DataFrame to subsample according to equilibrium detection on `series`.
    column : label or `pandas.Series`
        Label of column to use for calculating statistical inefficiency.
        Overrides `how`; can also take a `Series` object, but the index of the
        `Series` *must* match that of `data` exactly.
    how : {'right', 'left', 'random', 'sum'}
        The approach used to choose the observable on which correlations are
        calculated. See explanation above.
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
    random_state : int
        Integer between 0 and 2**32 -1 inclusive; fed to `numpy.random.seed`.
        Running this function on the same data with a specific random seed will
        produce the same result each time.
    nskip : int
        Number of samples to skip when walking through dataset; for long
        trajectories, can offer substantial speedups at the cost of possibly
        discarding slightly more data.

    Returns
    -------
    DataFrame
        `data` subsampled according to subsampled `column`.

    Note
    ----
    For a non-integer statistical ineffciency :math:`g`, the default value
    ``conservative=True`` will provide _fewer_ data points than allowed by
    :math:`g` and thus error estimates will be _higher_. For large numbers of
    data points and converged free energies, the choice should not make a
    difference. For small numbers of data points, ``conservative=True``
    decreases a false sense of accuracy and is deemed the more careful and
    conservative approach.

    References
    ----------
    [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
        histogram analysis method for the analysis of simulated and parallel tempering simulations.
        JCTC 3(1):26-41, 2007.
    [2] John D. Chodera. A simple method for automated
        equilibration detection in molecular simulations.
        Journal of Chemical Theory and Computation,
        12:1799, 2016.

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

    def subsample(group_c, group):
        group_cs = slicing(group_c, lower=lower, upper=upper, step=step)

        # calculate statistical inefficiency of series, with equilibrium detection
        t, statinef, Neff_max  = timeseries.detectEquilibration(group_cs, nskip=nskip)

        # only keep values past `t`
        group_cs = group_cs.iloc[t:]

        # use the subsampleCorrelatedData function to get the subsample index
        indices = timeseries.subsampleCorrelatedData(group_cs, g=statinef,
                                      conservative=conservative)

        calculated = dict()
        calculated['t'] = t
        calculated['statinef'] = statinef
        calculated['Neff_max'] = Neff_max

        return indices, calculated

    return _decorrelator(
            subsample=subsample,
            data=data,
            column=column,
            how=how,
            lower=lower,
            upper=upper,
            step=step,
            conservative=conservative,
            return_calculated=return_calculated,
            force=force,
            random_state=random_state)
