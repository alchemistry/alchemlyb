"""Functions for bootstrapping from datasets.

"""
import numpy as np
import pandas as pd


def bootstrap(df, random_state=None):
    """Bootstrap sample a DataFrame to produce a new DataFrame.

    Values are sampled from `df` on a per-lambda-index basis.
    Order of samples, or blocks of samples, are not preserved.

    Parameters
    ----------
    df : DataFrame
        DataFrame to bootstrap sample from.
    random_state : int, optional
        Integer between 0 and 2**32 -1 inclusive; fed to `numpy.random.seed`.
        Running this function on the same data with a specific random seed will
        produce the same result each time.

    Returns
    -------
    DataFrame
        DataFrame with the same shape as `df`, bootstrapped from the values of `df`.
    """
    np.random.seed(random_state)

    df = df.copy()
    lambda_inds = [i for i in df.index.names if not i == 'time']
    
    res = list()
    for name, group in df.groupby(level=lambda_inds):
        res.append(_bootstrap_samples(group))
        
    return pd.concat(res)
    

def _bootstrap_samples(df):
    indices = np.random.choice(len(df), size=len(df))
    return df.iloc[indices]


def block_bootstrap(df, block_size=None, correlation_method=None):
    """Block bootstrap a DataFrame from a DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame to block bootstrap sample from.

    Returns
    -------
    DataFrame
        DataFrame with the same shape as `df`, block bootstrapped from the values of `df`.
    """
    pass

