"""Functions for bootstrapping from datasets.

"""
import numpy as np

def bootstrap(df):
    """Bootstrap sample a DataFrame to produce a new DataFrame.

    Values are sampled from `df` on a per-lambda-index basis.

    Parameters
    ----------
    df : DataFrame
        DataFrame to bootstrap sample from.

    Returns
    -------
    DataFrame
        DataFrame with the same shape as `df`, bootstrapped from the values of `df`.
    """
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

