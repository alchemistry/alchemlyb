from __future__ import annotations
from typing import Literal

import numpy as np
import pandas as pd
import pymbar
import numpy.typing as npt
from sklearn.base import BaseEstimator

from . import BAR
from .base import _EstimatorMixOut


class MBAR(BaseEstimator, _EstimatorMixOut):
    r"""Multi-state Bennett acceptance ratio (MBAR).

    Parameters
    ----------

    maximum_iterations : int, optional
        Set to limit the maximum number of iterations performed.

    relative_tolerance : float, optional
        Set to determine the relative tolerance convergence criteria.

    initial_f_k : np.ndarray, float, shape=(K), optional or String `BAR`
        When `isinstance(initial_f_k, np.ndarray)`, `initial_f_k` will be used as
        initial guess for MBAR estimator. initial_f_k should be dimensionless
        free energies.
        When `initial_f_k` is ``None``, ``initial_f_k`` will be set to 0.
        When `initial_f_k` is set to "BAR", a BAR calculation will be done and
        the result is used as the initial guess (default).

        .. versionchanged:: 2.3.0
           The new default is now "BAR" as it provides a substantial speedup
           over the previous default `None`.


    method : str, optional, default="robust"
        The optimization routine to use.  This can be any of the methods
        available via :func:`scipy.optimize.minimize` or
        :func:`scipy.optimize.root`.

    n_bootstraps : int, optional
        Whether to use bootstrap to estimate uncertainty. `0` means use analytic error
        estimation. 50~200 is a reasonable range to do bootstrap.

    verbose : bool, optional
        Set to ``True`` if verbose debug output from :mod:`pymbar` is desired.

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

    delta_sT_ : DataFrame, optional
        The estimated dimensionless entropy difference between each state.

    d_delta_sT_ : DataFrame
        The estimated statistical uncertainty (one standard deviation) in
        dimensionless entropy differences.

    theta_ : DataFrame
        The theta matrix.

    states_ : list
        Lambda states for which free energy differences were obtained.

    Notes
    -----
    See [Shirts2008]_ for details of the derivation and cite the
    paper when using MBAR in published work.

    See Also
    --------
    pymbar.mbar.MBAR


    .. versionchanged:: 1.0.0
       `delta_f_`, `d_delta_f_`, `states_` are view of the original object.
    .. versionchanged:: 2.0.0
        default value for `method` was changed from "hybr" to "robust"
    .. versionchanged:: 2.1.0
        `n_bootstraps` option added.
    .. versionchanged:: 2.4.0
        Handle initial estimate, initial_f_k, from bar in the instance
        that not all lambda states represented as column headers are
        represented in the indices of u_nk.
    .. versionchanged:: 2.5.0
        Added computation of enthalpy and entropy

    """

    def __init__(
        self,
        maximum_iterations: int = 10000,
        relative_tolerance: float = 1.0e-7,
        initial_f_k: np.ndarray | Literal["BAR"] | None = "BAR",
        method: str = "robust",
        n_bootstraps: int = 0,
        verbose: bool = False,
    ) -> None:
        self.maximum_iterations = maximum_iterations
        self.relative_tolerance = relative_tolerance
        if isinstance(initial_f_k, str) and initial_f_k != "BAR":
            raise ValueError(
                f"Only `BAR` is supported as string input to `initial_f_k`. Got ({initial_f_k})."
            )
        else:
            self.initial_f_k = initial_f_k
        self.method = method
        self.verbose = verbose
        self.n_bootstraps = n_bootstraps

        # handle for pymbar.MBAR object
        self._mbar = None

    def fit(self, u_nk: pd.DataFrame, compute_entropy_enthalpy: bool = False) -> "MBAR":
        """
        Compute overlap matrix of reduced potentials using multi-state
        Bennett acceptance ratio.

        Parameters
        ----------
        u_nk : DataFrame
            ``u_nk[n, k]`` is the reduced potential energy of uncorrelated
            configuration ``n`` evaluated at state ``k``.

        compute_entropy_enthalpy : bool, optional, default=False
            Compute entropy and enthalpy.

        """
        # sort by state so that rows from same state are in contiguous blocks
        u_nk = u_nk.sort_index(level=u_nk.index.names[1:])  # type: ignore[arg-type]

        groups = u_nk.groupby(level=u_nk.index.names[1:])
        N_k = [
            (
                len(groups.get_group(i if isinstance(i, tuple) else (i,)))  # type: ignore[unreachable]
                if i in groups.groups
                else 0
            )
            for i in u_nk.columns
        ]
        self._states_ = u_nk.columns.values.tolist()

        if isinstance(self.initial_f_k, str) and self.initial_f_k == "BAR":
            bar = BAR(
                maximum_iterations=self.maximum_iterations,
                relative_tolerance=self.relative_tolerance,
                verbose=self.verbose,
            )
            bar.fit(u_nk)
            initial_f_k = bar.delta_f_.iloc[0, :]  # type: ignore[union-attr]
            if len(bar.delta_f_.iloc[0, :]) != len(self._states_):  # type: ignore[union-attr]
                states = [
                    x
                    for i, x in enumerate(self._states_[:-1])
                    if N_k[i] > 0 and N_k[i + 1] > 0
                ]
                initial_f_k = pd.Series(
                    [
                        initial_f_k.loc(x) if x in states else np.nan
                        for x in self._states_  # type: ignore
                    ],
                    index=self._states_,  # type: ignore[arg-type]
                    dtype=float,
                )
        else:
            initial_f_k = self.initial_f_k

        self._mbar = pymbar.MBAR(
            u_nk.T,
            N_k,
            maximum_iterations=self.maximum_iterations,
            relative_tolerance=self.relative_tolerance,
            verbose=self.verbose,
            initial_f_k=initial_f_k,
            solver_protocol=self.method,
            n_bootstraps=self.n_bootstraps,
        )

        uncertainty_method = None if self.n_bootstraps == 0 else "bootstrap"
        out = self._mbar.compute_free_energy_differences(  # type: ignore[attr-defined]
            return_theta=True, uncertainty_method=uncertainty_method
        )
        if compute_entropy_enthalpy:
            out.update(
                self._mbar.compute_entropy_and_enthalpy(  # type: ignore[attr-defined]
                    uncertainty_method=uncertainty_method
                )
            )

        self._delta_f_ = pd.DataFrame(
            out["Delta_f"],
            columns=self._states_,  # type: ignore[arg-type]
            index=self._states_,  # type: ignore[arg-type]
        )
        self._d_delta_f_ = pd.DataFrame(
            out["dDelta_f"],
            columns=self._states_,  # type: ignore[arg-type]
            index=self._states_,  # type: ignore[arg-type]
        )
        self.theta_ = pd.DataFrame(
            out["Theta"],
            columns=self._states_,  # type: ignore[arg-type]
            index=self._states_,  # type: ignore[arg-type]
        )
        if compute_entropy_enthalpy:
            self._delta_h_ = pd.DataFrame(
                out["Delta_u"],
                columns=self._states_,  # type: ignore[arg-type]
                index=self._states_,  # type: ignore[arg-type]
            )
            self._d_delta_h_ = pd.DataFrame(
                out["dDelta_u"],
                columns=self._states_,  # type: ignore[arg-type]
                index=self._states_,  # type: ignore[arg-type]
            )
            self._delta_sT_ = pd.DataFrame(
                out["Delta_s"],
                columns=self._states_,  # type: ignore[arg-type]
                index=self._states_,  # type: ignore[arg-type]
            )
            self._d_delta_sT_ = pd.DataFrame(
                out["dDelta_s"],
                columns=self._states_,  # type: ignore[arg-type]
                index=self._states_,  # type: ignore[arg-type]
            )

        self._delta_f_.attrs = u_nk.attrs
        self._d_delta_f_.attrs = u_nk.attrs
        if compute_entropy_enthalpy:
            self._delta_h_.attrs = u_nk.attrs  # type: ignore[union-attr]
            self._d_delta_h_.attrs = u_nk.attrs  # type: ignore[union-attr]
            self._delta_sT_.attrs = u_nk.attrs  # type: ignore[union-attr]
            self._d_delta_sT_.attrs = u_nk.attrs  # type: ignore[union-attr]

        return self

    @property
    def overlap_matrix(self) -> npt.NDArray:
        r"""MBAR overlap matrix.

        The estimated state overlap matrix :math:`O_{ij}` is an estimate of the probability
        of observing a sample from state :math:`i` in state :math:`j`.

        The :attr:`overlap_matrix` is computed on-the-fly. Assign it to a variable if
        you plan to re-use it.

        See Also
        ---------
        pymbar.mbar.MBAR.computeOverlap
        """
        return self._mbar.compute_overlap()["matrix"]  # type: ignore[attr-defined, no-any-return]
