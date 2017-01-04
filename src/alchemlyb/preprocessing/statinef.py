import numpy as np
from pymbar.timeseries import statisticalInefficiency

def subsample(df, series, lower=None, upper=None, step=None):
    """Subsample a DataFrame based on the calculated statistical inefficiency of a timeseries.

    Parameters
    ----------
    df : DataFrame
        DataFrame to subsample according statistical inefficiency of `series`.
    series : Series
        Series to use for calculating statistical inefficiency.
    lower : float
        Lower bound to slice `series` data from.
    upper : float
        Upper bound to slice `series` to (inclusive).
    step : int
        Step between `series` items to slice by.

    Returns
    -------
    DataFrame
        `df`, subsampled according to subsampled `series`.

    """
    series = series.loc[lower:upper]

    # drop any rows that have missing values
    series = series.dropna()

    # subsample according to step
    series = series.iloc[::step]

    # subsample according to statistical inefficiency after equilibration detection
    # we do this after slicing by lower/upper to simulate
    # what we'd get with only this data available
    statinef  = statisticalInefficiency(series)

    # we round up
    statinef = int(np.rint(statinef))

    # subsample according to statistical inefficiency
    series = series.iloc[::statinef]

    df = df.loc[series.index]
    
    return df
