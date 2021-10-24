import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from pymbar import MBAR as MBAR_
import pymbar

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

    states_ : list
        Lambda states for which free energy differences were obtained.

    """

    def __init__(self, maximum_iterations=10000, relative_tolerance=1.0e-7,
                 initial_f_k=None, method='hybr', verbose=False):
        self.maximum_iterations = maximum_iterations
        self.relative_tolerance = relative_tolerance
        self.initial_f_k = initial_f_k
        self.method = method
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
            u_nk[n,k] is the reduced potential energy of uncorrelated
            configuration n evaluated at state k.

        """
        # sort by state so that rows from same state are in contiguous blocks
        u_nk = u_nk.sort_index(level=u_nk.index.names[1:])

        groups = u_nk.groupby(level=u_nk.index.names[1:])
        N_k = [(len(groups.get_group(i)) if i in groups.groups else 0) for i in
               u_nk.columns]
        self.states_ = u_nk.columns.values.tolist()

        # Prepare the solver_protocol as stated in https://github.com/choderalab/pymbar/issues/419#issuecomment-803714103
        solver_options = {"maximum_iterations": self.maximum_iterations,
                          "verbose": self.verbose}
        solver_protocol = {"method": self.method,
                           "options": solver_options}

        self._mbar, out = self._do_MBAR(u_nk, N_k, solver_protocol)

        free_energy_differences = [pd.DataFrame(i,
                                                columns=self.states_,
                                                index=self.states_) for i in
                                   out]

        (self.delta_f_, self.d_delta_f_, self.theta_) = free_energy_differences

        self.delta_f_.attrs = u_nk.attrs
        self.d_delta_f_.attrs = u_nk.attrs

        return self

    def predict(self, u_ln):
        pass

    def _do_MBAR(self, u_nk, N_k, solver_protocol):
        mbar = MBAR_(u_nk.T, N_k,
                     relative_tolerance=self.relative_tolerance,
                     initial_f_k=self.initial_f_k,
                     solver_protocol=(solver_protocol,))

        # set attributes
        out = mbar.getFreeEnergyDifferences(return_theta=True)
        return mbar, out

    @property
    def overlap_matrix(self):
        r"""MBAR overlap matrix.

        The estimated state overlap matrix :math:`O_{ij}` is an estimate of the probability
        of observing a sample from state :math:`i` in state :math:`j`.

        The :attr:`overlap_matrix` is computed on-the-fly. Assign it to a variable if
        you plan to re-use it.

        See Also
        ---------
        pymbar.mbar.MBAR.computeOverlap
        """
        return self._mbar.computeOverlap()['matrix']


class AutoMBAR(MBAR):
    """A more robust version of Multi-state Bennett acceptance ratio (MBAR).

    Given that there isn't a single *method* that would allow MBAR to converge
    at every single cases, the AutoMBAR class is created to iteratively try
    all the available methods to obtain the converged estimate. The fastest
    method *hybr* will be tried first, followed by the most stable method
    *adaptive*. If *adaptive* finds it difficult to converge, *BFGS* will be
    used as last resort. Though *BFGS* is not as stable as *adaptive*,
    it has been shown that it could solve some cases where *adaptive* cannot.

    This class is created to facilitate the high-throughput pipeline with
    the intention of minimising the chance that MBAR solver will not converge.

    Note
    ----
    Unlike :class:`~alchemlyb.estimators.MBAR`, the method of solving the MBAR
    is determined by the class.


    .. versionadded:: 0.6.0
    """
    def __init__(self, maximum_iterations=10000, relative_tolerance=1.0e-7,
                 initial_f_k=None, verbose=False):
        super().__init__(maximum_iterations=maximum_iterations,
                         relative_tolerance=relative_tolerance,
                         initial_f_k=initial_f_k,
                         verbose=verbose, method=None)

    def _do_MBAR(self, u_nk, N_k, solver_protocol):
        # Try the fastest method first
        try:
            solver_protocol["method"] = 'hybr'
            mbar, out = super()._do_MBAR(u_nk, N_k, solver_protocol)
        except pymbar.utils.ParameterError:
            try:
                solver_protocol["method"] = 'adaptive'
                mbar, out = super()._do_MBAR(u_nk, N_k, solver_protocol)
            except pymbar.utils.ParameterError:
                solver_protocol["method"] = 'BFGS'
                mbar, out = super()._do_MBAR(u_nk, N_k, solver_protocol)
        return mbar, out
