import numpy as np
import pandas as pd
from pymbar.other_estimators import bar as BAR_
from sklearn.base import BaseEstimator

from .base import _EstimatorMixOut


class BAR(BaseEstimator, _EstimatorMixOut):
    """Bennett acceptance ratio (BAR).

    Parameters
    ----------

    maximum_iterations : int, optional
        Set to limit the maximum number of iterations performed.

    relative_tolerance : float, optional
        Set to determine the relative tolerance convergence criteria.

    method : str, optional, default='false-position'
        choice of method to solve BAR nonlinear equations,
        one of 'self-consistent-iteration' or 'false-position' (default: 'false-position')

    verbose : bool, optional
        Set to True if verbose debug output is desired.

    Attributes
    ----------

    delta_f_ : DataFrame
        The estimated dimensionless free energy difference between each state.

    d_delta_f_ : DataFrame
        The estimated statistical uncertainty (one standard deviation) in
        dimensionless free energy differences.

    states_ : list
        Lambda states for which free energy differences were obtained.

    Notes
    -----
    See [Bennett1976]_ for details of the derivation and cite the paper
    (together with [Shirts2008]_ for the Python implementation in
    :mod:`pymbar`) when using BAR in published work.

    When possible, use MBAR instead of BAR as it makes better use of the
    available data.

    See Also
    --------
    MBAR


    .. versionchanged:: 1.0.0
       `delta_f_`, `d_delta_f_`, `states_` are view of the original object.
    .. versionchanged:: 2.4.0
       Added assessment of lambda states represented in the indices of u_nk
       to provide meaningful errors to ensure proper use.

    """

    def __init__(
        self,
        maximum_iterations=10000,
        relative_tolerance=1.0e-7,
        method="false-position",
        verbose=False,
    ):
        self.maximum_iterations = maximum_iterations
        self.relative_tolerance = relative_tolerance
        self.method = method
        self.verbose = verbose

        # handle for pymbar.BAR object
        self._bar = None

    def fit(self, u_nk):
        """
        Compute overlap matrix of reduced potentials using
        Bennett acceptance ratio.

        Parameters
        ----------
        u_nk : DataFrame
            u_nk[n,k] is the reduced potential energy of uncorrelated
            configuration n evaluated at state k.

        """
        # sort by state so that rows from same state are in contiguous blocks
        u_nk = u_nk.sort_index(level=u_nk.index.names[1:])

        # get a list of the lambda states that are sampled
        self._states_ = u_nk.columns.values.tolist()

        # group u_nk by lambda states
        groups = u_nk.groupby(level=u_nk.index.names[1:])
        N_k = [
            (
                len(groups.get_group(i if isinstance(i, tuple) else (i,)))
                if i in groups.groups
                else 0
            )
            for i in u_nk.columns
        ]

        # Pull lambda states from indices
        states = list(set(x[1:] if len(x[1:]) > 1 else x[1] for x in u_nk.index))
        for state in states:
            if state not in self._states_:
                raise ValueError(
                    f"Indexed lambda state, {state}, is not represented in u_nk columns:"
                    f" {self._states_}"
                )
        states.sort(key=lambda x: self._states_.index(x))

        # Now get free energy differences and their uncertainties for each step
        deltas = np.array([])
        d_deltas = np.array([])
        for k in range(len(N_k) - 1):
            if N_k[k] == 0 or N_k[k + 1] == 0:
                continue

            # get us from lambda step k
            uk = groups.get_group(
                self._states_[k]
                if isinstance(self._states_[k], tuple)
                else (self._states_[k],)
            )
            # get w_F
            w_f = uk.iloc[:, k + 1] - uk.iloc[:, k]

            # get us from lambda step k+1
            uk1 = groups.get_group(
                self._states_[k + 1]
                if isinstance(self._states_[k + 1], tuple)
                else (self._states_[k + 1],)
            )

            # get w_R
            w_r = uk1.iloc[:, k] - uk1.iloc[:, k + 1]

            # now determine df and ddf using pymbar.BAR
            out = BAR_(
                w_f,
                w_r,
                method=self.method,
                maximum_iterations=self.maximum_iterations,
                relative_tolerance=self.relative_tolerance,
                verbose=self.verbose,
            )

            df, ddf = out["Delta_f"], out["dDelta_f"]
            deltas = np.append(deltas, df)
            d_deltas = np.append(d_deltas, ddf**2)

        if len(deltas) == 0 and len(states) > 1:
            raise ValueError(
                "u_nk does not contain energies computed between any adjacent states.\n"
                "To compute the free energy with BAR, ensure that values in u_nk exist"
                f" for the columns:\n{states}."
            )

        # build matrix of deltas between each state
        adelta = np.zeros((len(deltas) + 1, len(deltas) + 1))
        ad_delta = np.zeros_like(adelta)

        for j in range(len(deltas)):
            out = []
            dout = []
            for i in range(len(deltas) - j):
                out.append(deltas[i : i + j + 1].sum())

                # See https://github.com/alchemistry/alchemlyb/pull/60#issuecomment-430720742
                # Error estimate generated by BAR ARE correlated

                # Use the BAR uncertainties between two neighbour states
                if j == 0:
                    dout.append(d_deltas[i : i + j + 1].sum())
                # Other uncertainties are unknown at this point
                else:
                    dout.append(np.nan)

            adelta += np.diagflat(np.array(out), k=j + 1)
            ad_delta += np.diagflat(np.array(dout), k=j + 1)

        # yield standard delta_f_ free energies between each state
        self._delta_f_ = pd.DataFrame(adelta - adelta.T, columns=states, index=states)

        # yield standard deviation d_delta_f_ between each state
        self._d_delta_f_ = pd.DataFrame(
            np.sqrt(ad_delta + ad_delta.T), columns=states, index=states
        )
        self._delta_f_.attrs = u_nk.attrs
        self._d_delta_f_.attrs = u_nk.attrs

        return self
