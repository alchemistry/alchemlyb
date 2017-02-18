import numpy as np
import pandas as pd
from scipy.integrate import simps

from sklearn.base import BaseEstimator


class TI(BaseEstimator):
    """Thermodynamic integration (TI).

    Parameters
    ----------

    verbose : bool, optional
        Set to True if verbose debug output is desired.

    Attributes
    ----------

    delta_f_ : DataFrame
        The estimated dimensionless free energy difference between each state.

    """

    def __init__(self, verbose=False):
        self.verbose = verbose


    def fit(self, dHdl):
        """
        Compute free energy differences between each state by integrating
        dHdl across lambda values.

        Parameters
        ----------
        dHdl : DataFrame 
            dHdl[n,k] is the potential energy gradient with respect to lambda
            for each configuration n and lambda k.

        """

        # sort by state so that rows from same state are in contiguous blocks,
        # and adjacent states are next to each other
        dHdl = dHdl.sort_index(level=dHdl.index.names[1:])

        # obtain the mean value for each state
        means = dHdl.mean(level=dHdl.index.names[1:])
        
        # obtain vector of delta lambdas between each state
        dl = means.reset_index()[means.index.names[:]].diff().iloc[1:].values

        # apply trapezoid rule to obtain DF between each adjacent state
        deltas = (dl * (means.iloc[:-1].values + means.iloc[1:].values)/2).sum(axis=1)

        # build matrix of deltas between each state
        adelta = np.zeros((len(deltas)+1, len(deltas)+1))

        for j in range(len(deltas)):
            out = []
            for i in range(len(deltas) - j):
                out.append(deltas[i] + deltas[i+1:i+j+1].sum())

            adelta += np.diagflat(np.array(out), k=j+1)

        # yield standard delta_f_ free energies between each state
        self.delta_f_ = pd.DataFrame(adelta - adelta.T,
                                     columns=means.index.values,
                                     index=means.index.values)

        return self
        


