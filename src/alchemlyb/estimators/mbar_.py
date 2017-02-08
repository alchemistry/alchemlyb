import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from pymbar import MBAR as MBAR_
from pymbar.mbar import DEFAULT_SOLVER_PROTOCOL
from pymbar.mbar import DEFAULT_SUBSAMPLING_PROTOCOL


class MBAR(BaseEstimator):
    """Multi-state Bennett acceptance ratio (MBAR).

    Parameters
    ----------

    maximum_iterations : int, optional
        Set to limit the maximum number of iterations performed.

    relative_tolerance : float, optional
        Set to determine the relative tolerance convergence criteria.

    initial_f_k : np.ndarray, float, shape=(K), optional
        Set to the initial dimensionless free energies to use as a 
        guess (default None, which sets all f_k = 0).

    method : str, optional, default="hybr"
        The optimization routine to use.  This can be any of the methods
        available via scipy.optimize.minimize() or scipy.optimize.root().

    verbose : bool, optional
        Set to True if verbose debug output is desired.

    Attributes
    ----------

    delta_f_ : DataFrame
        The estimated dimensionless free energy difference between each state.

    d_delta_f_ : DataFrame
        The estimated statistical uncertainty (one standard deviation) in
        dimensionless free energy differences.

    theta_ : DataFrame
        The theta matrix.

    """

    def __init__(self, maximum_iterations=10000, relative_tolerance=1.0e-7,
                 initial_f_k=None, method='hybr', verbose=False):

        self.maximum_iterations = maximum_iterations
        self.relative_tolerance = relative_tolerance
        self.initial_f_k = initial_f_k
        self.method = (dict(method=method), )
        self.verbose = verbose

        # handle for pymbar.MBAR object
        self._mbar = None

    def fit(self, u_nk):
        """
        Compute overlap matrix of reduced potentials using multi-state
        Bennett acceptance ratio.

        Parameters
        ----------
        u_nk : DataFrame 
            u_kn[k,n] is the reduced potential energy of uncorrelated
            configuration n evaluated at state k.

        u_nk = [ u_1(x_1) u_2(x_1) u_3(x_1) . . . u_k(x_1)
                 u_1(x_2) u_2(x_2) u_3(x_2) . . . u_k(x_2)
                                .  .  .
                 u_1(x_n) u_2(x_n) u_3(x_n) . . . u_k(x_n)]

        """
        # sort by state so that rows from same state are in contiguous blocks
        u_nk = u_nk.sort_index(level=u_nk.index.names[1:])
        
        groups = u_nk.groupby(level=u_nk.index.names[1:])
        N_k = [(len(groups.get_group(i)) if i in groups.groups else 0) for i in u_nk.columns]        
        
        self._mbar = MBAR_(u_nk.T, N_k,
                           maximum_iterations=self.maximum_iterations,
                           relative_tolerance=self.relative_tolerance,
                           initial_f_k=self.initial_f_k,
                           solver_protocol=self.method,
                           verbose=self.verbose)

        self.states_ = u_nk.columns.values.tolist()
        
        return self

    @property
    def delta_f_(self):
        if self._mbar is not None:
            out = self._mbar.getFreeEnergyDifferences()[0]
            return pd.DataFrame(out, columns=self.states_, index=self.states_)

    @property
    def d_delta_f_(self):
        if self._mbar is not None:
            out = self._mbar.getFreeEnergyDifferences()[1]
            return pd.DataFrame(out, columns=self.states_, index=self.states_)

    @property
    def theta_(self):
        if self._mbar is not None:
            out = self._mbar.getFreeEnergyDifferences()[2]
            return pd.DataFrame(out, columns=self.states_, index=self.states_)
            
    def predict(self, u_ln):
        pass
