def subsample(df, lower=None, upper=None, step=None):
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
        `df`, subsampled.

    """
    df = df.loc[lower:upper]

    # drop any rows that have missing values
    df = df.dropna()

    # subsample according to step
    df = df.iloc[::step]

    return df
