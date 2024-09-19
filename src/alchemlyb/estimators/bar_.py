import numpy as np
import pandas as pd
import pymbar
from pymbar.other_estimators import bar as BAR_
from sklearn.base import BaseEstimator

from .. import concat
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

    n_bootstraps : int, optional
        Whether to use bootstrap to estimate uncertainty. `0` means use analytic error
        estimation. 50~200 is a reasonable range to do bootstrap. Currently used when
        ``use_mbar is True``.

    verbose : bool, optional
        Set to True if verbose debug output is desired.

    Attributes
    ----------

    delta_f_ : DataFrame
        The estimated dimensionless free energy difference between each state.

    d_delta_f_ : DataFrame
        The estimated statistical uncertainty (one standard deviation) in
        dimensionless free energy differences.

    delta_h_ : DataFrame
        The estimated dimensionless enthalpy difference between each state.

    d_delta_h_ : DataFrame
        The estimated statistical uncertainty (one standard deviation) in
        dimensionless enthalpy differences.

    delta_s_ : DataFrame, optional
        The estimated dimensionless entropy difference between each state.

    d_delta_s_ : DataFrame
        The estimated statistical uncertainty (one standard deviation) in
        dimensionless entropy differences.

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
       Added computation of enthalpy and entropy with 2-state MBAR

    """

    def __init__(
        self,
        maximum_iterations=10000,
        relative_tolerance=1.0e-7,
        method="false-position",
        n_bootstraps=0,
        verbose=False,
    ):

        self.maximum_iterations = maximum_iterations
        self.relative_tolerance = relative_tolerance
        self.method = method
        self.n_bootstraps = n_bootstraps
        self.verbose = verbose

        # handle for pymbar.BAR object
        self._bar = None

    def fit(self, u_nk, use_mbar=False, compute_entropy_enthalpy=False):
        """
        Compute overlap matrix of reduced potentials using
        Bennett acceptance ratio.

        Sets the attributes: delta_f_ and d_delta_f_

        Parameters
        ----------
        u_nk : DataFrame
            u_nk[n,k] is the reduced potential energy of uncorrelated
            configuration n evaluated at state k.

        use_mbar : bool, optional, default=False
            Use 2-state MBAR instead of BAR. This will allow for the
            calculation of enthalpic and entropic contributions.

        compute_entropy_enthalpy : bool, optional, default=False
            Compute entropy and enthalpy from 2-state MBAR. Note
            that ``use_mbar`` must be ``True``.

        """

        if compute_entropy_enthalpy and not use_mbar:
            raise ValueError(
                "Cannot compute the enthalpy and entropy with BAR, set use_mbar=True."
            )

        # sort by state so that rows from same state are in contiguous blocks
        u_nk = u_nk.sort_index(level=u_nk.index.names[1:])

        # get a list of the lambda states that are sampled
        self._states_ = u_nk.columns.values.tolist()

        # group u_nk by lambda states
        groups = u_nk.groupby(level=u_nk.index.names[1:])
        N_k = [
            (len(groups.get_group(i)) if i in groups.groups else 0)
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
        if compute_entropy_enthalpy:
            deltas_h = np.array([])
            d_deltas_h = np.array([])
            deltas_s = np.array([])
            d_deltas_s = np.array([])
        for k in range(len(N_k) - 1):
            if N_k[k] == 0 or N_k[k + 1] == 0:
                continue

            # get us from lambda step k
            uk = groups.get_group(self._states_[k])
            # get w_F
            w_f = uk.iloc[:, k + 1] - uk.iloc[:, k]

            # get us from lambda step k+1
            uk1 = groups.get_group(self._states_[k + 1])
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
            if use_mbar:  # now determine df and ddf using pymbar.MBAR
                tmp_u_nk = concat(
                    [
                        groups.get_group(
                            self._states_[k]
                            if isinstance(self._states_[k], tuple)
                            else (self._states_[k],)
                        ),
                            groups.get_group(
                                self._states_[k + 1]
                                if isinstance(self._states_[k + 1], tuple)
                                else (self._states_[k + 1],)
                        ),
                    ]
                )
                columns = tmp_u_nk.columns.tolist()
                tmp_u_nk = tmp_u_nk.drop(
                    [x for x in columns if x not in columns[k : k + 2]], axis=1
                )
                mbar = pymbar.MBAR(
                    tmp_u_nk.T,
                    N_k[k : k + 2],
                    maximum_iterations=self.maximum_iterations,
                    relative_tolerance=self.relative_tolerance,
                    verbose=self.verbose,
                    initial_f_k=[0, out["Delta_f"]],
                    solver_protocol=self.method,
                    n_bootstraps=self.n_bootstraps,
                )
                uncertainty_method = None if self.n_bootstraps == 0 else "bootstrap"
                if compute_entropy_enthalpy:
                    out = mbar.compute_entropy_and_enthalpy(
                        uncertainty_method=uncertainty_method
                    )
                else:
                    out = mbar.compute_free_energy_differences(
                        uncertainty_method=uncertainty_method
                    )
                out = {key: val[0, 1] for key, val in out.items()}

            df, ddf = out["Delta_f"], out["dDelta_f"]
            deltas = np.append(deltas, df)
            d_deltas = np.append(d_deltas, ddf**2)
            if compute_entropy_enthalpy:
                dh, ddh = out["Delta_u"], out["dDelta_u"]
                deltas_h = np.append(deltas_h, dh)
                d_deltas_h = np.append(d_deltas_h, ddh**2)

                ds, dds = out["Delta_s"], out["dDelta_s"]
                deltas_s = np.append(deltas_s, ds)
                d_deltas_s = np.append(d_deltas_s, dds**2)

        if len(deltas) == 0 and len(states) > 1:
            raise ValueError(
                "u_nk does not contain energies computed between any adjacent states.\n"
                "To compute the free energy with BAR, ensure that values in u_nk exist"
                f" for the columns:\n{states}."
            )

        # build matrix of deltas between each state
        lx = len(deltas)
        adelta = np.zeros((lx + 1, lx + 1))
        ad_delta = np.zeros_like(adelta)
        if compute_entropy_enthalpy:
            adelta_h, ad_delta_h = np.zeros_like(adelta), np.zeros_like(adelta)
            adelta_s, ad_delta_s = np.zeros_like(adelta), np.zeros_like(adelta)

        for j in range(lx):
            out_f, dout_f = np.empty(lx - j), np.empty(lx - j)
            if compute_entropy_enthalpy:
                out_h, dout_h = np.empty(lx - j), np.empty(lx - j)
                out_s, dout_s = np.empty(lx - j), np.empty(lx - j)
            for i in range(lx - j):
                out_f[i] = deltas[i : i + j + 1].sum()

                # See https://github.com/alchemistry/alchemlyb/pull/60#issuecomment-430720742
                # Error estimate generated by BAR ARE correlated

                # Use the BAR uncertainties between two neighbour states
                # Other uncertainties are unknown at this point
                dout_f[i] = d_deltas[i : i + j + 1].sum() if j == 0 else np.nan

                if compute_entropy_enthalpy:
                    out_h[i] = deltas_h[i : i + j + 1].sum()
                    out_s[i] = deltas_s[i : i + j + 1].sum()

                    # Use the BAR uncertainties between two neighbour states
                    # Other uncertainties are unknown at this point
                    dout_h[i] = d_deltas_h[i : i + j + 1].sum() if j == 0 else np.nan
                    dout_s[i] = d_deltas_s[i : i + j + 1].sum() if j == 0 else np.nan

            adelta += np.diagflat(out_f, k=j + 1)
            ad_delta += np.diagflat(dout_f, k=j + 1)
            if compute_entropy_enthalpy:
                adelta_h += np.diagflat(out_h, k=j + 1)
                ad_delta_h += np.diagflat(dout_h, k=j + 1)
                adelta_s += np.diagflat(out_s, k=j + 1)
                ad_delta_s += np.diagflat(dout_s, k=j + 1)

        # yield standard delta_f_ free energies between each state
        self._delta_f_ = pd.DataFrame(adelta - adelta.T, columns=states, index=states)
        if compute_entropy_enthalpy:
            self._delta_h_ = pd.DataFrame(
                adelta_h - adelta_h.T, columns=states, index=states
            )
            self._delta_s_ = pd.DataFrame(
                adelta_s - adelta_s.T, columns=states, index=states
            )

        # yield standard deviation d_delta_f_ between each state
        self._d_delta_f_ = pd.DataFrame(
            np.sqrt(ad_delta + ad_delta.T), columns=states, index=states
        )
        if compute_entropy_enthalpy:
            self._d_delta_h_ = pd.DataFrame(
                np.sqrt(ad_delta_h + ad_delta_h.T), columns=states, index=states
            )
            self._d_delta_s_ = pd.DataFrame(
                np.sqrt(ad_delta_s + ad_delta_s.T), columns=states, index=states
            )

        self._delta_f_.attrs = u_nk.attrs
        self._d_delta_f_.attrs = u_nk.attrs
        if compute_entropy_enthalpy:
            self._delta_h_.attrs = u_nk.attrs
            self._d_delta_h_.attrs = u_nk.attrs
            self._delta_s_.attrs = u_nk.attrs
            self._d_delta_s_.attrs = u_nk.attrs

        return self
