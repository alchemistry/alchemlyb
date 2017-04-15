import numpy as np
import pandas as pd

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

    def _partial_sum_error(self, i, dl, dHdl_var):
        """
        Calculate the error of the partial sums in the trapezoidal rule.

        Parameters
        ----------
        i : int
            the ith partial sum (starting at 0)

        dl : array-like
            The delta lambdas between states

        dHdl_var : array-like
            The variance of each lambda state
        """

        return 0.25 * (dl[i]**2) * (dHdl_var[i] * dHdl_var[i+1])

    def _trapezoidal_error(self, i, j, dl, dHdl_var):
        """Calculate the error of the trapezoidal rule integral approximation
        betweeen state i and state j
        
        Parameters
        ----------
        i : int
            index of state i (starting at 0)

        j : int
            index of state j (starting at 0)

        dl : array-like
            The delta lambdas between states

        dHdl_var : array-like
            The variance of each lambda state
        """

        return np.sum([self._partial_sum_error(x, dl, dHdl_var) for x in range(i,j+1)])

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
        variance = dHdl.std(level=dHdl.index.names[1:])**2
        
        # obtain vector of delta lambdas between each state
        dl = means.reset_index()[means.index.names[:]].diff().iloc[1:].values

        # apply trapezoid rule to obtain DF between each adjacent state
        deltas = (dl * (means.iloc[:-1].values + means.iloc[1:].values)/2).sum(axis=1)

        # build matrix of deltas between each state
        adelta = np.zeros((len(deltas)+1, len(deltas)+1))

        # build matrix of delta errors between each state
        adelta_error = np.zeros_like(adelta)

        for j in range(len(deltas)):
            out = []
            out_error = []
            for i in range(len(deltas) - j):
                out.append(deltas[i] + deltas[i+1:i+j+1].sum())
                out_error.append(self._trapezoidal_error(i, j, dl, variance))

            adelta += np.diagflat(np.array(out), k=j+1)
            adelta_error += np.diagflat(np.array(out_error), k=j+1)

        # yield standard delta_f_ free energies between each state
        self.delta_f_ = pd.DataFrame(adelta - adelta.T,
                                     columns=means.index.values,
                                     index=means.index.values)
        # yield delta_f_error_ free energy errors between each state
        self.delta_f_error_ = pd.DataFrame(adelta_error + adelta_error.T,
                                           columns=means.index.values,
                                           index=means.index.values)

        return self
