import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .base import _EstimatorMixOut


class TI(BaseEstimator, _EstimatorMixOut):
    """Thermodynamic integration (TI).

    Parameters
    ----------

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

    dhdl : DataFrame
        The estimated dhdl of each state.


    .. versionchanged:: 1.0.0
       `delta_f_`, `d_delta_f_`, `states_` are view of the original object.

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

        # obtain the mean and variance of the mean for each state
        # variance calculation assumes no correlation between points
        # used to calculate mean
        means = dHdl.groupby(level=dHdl.index.names[1:]).mean()
        variances = np.square(dHdl.groupby(level=dHdl.index.names[1:]).sem())

        # get the lambda names
        l_types = dHdl.index.names[1:]

        # obtain vector of delta lambdas between each state
        # Fix issue #148, where for pandas == 1.3.0
        # dl = means.reset_index()[list(means.index.names[:])].diff().iloc[1:].values
        dl = means.reset_index()[means.index.names[:]].diff().iloc[1:].values

        # apply trapezoid rule to obtain DF between each adjacent state
        deltas = (dl * (means.iloc[:-1].values + means.iloc[1:].values) / 2).sum(axis=1)

        # build matrix of deltas between each state
        adelta = np.zeros((len(deltas) + 1, len(deltas) + 1))
        ad_delta = np.zeros_like(adelta)

        for j in range(len(deltas)):
            out = []
            dout = []
            for i in range(len(deltas) - j):
                out.append(deltas[i] + deltas[i + 1 : i + j + 1].sum())

                # Define additional zero lambda
                a = [0.0] * len(l_types)

                # Define dl series' with additional zero lambda on the left and right
                dll = np.insert(dl[i : i + j + 1], 0, [a], axis=0)
                dlr = np.append(dl[i : i + j + 1], [a], axis=0)

                # Get a series of the form: x1, x1 + x2, ..., x(n-1) + x(n), x(n)
                dllr = dll + dlr

                # Append deviation of free energy difference between state i and i+j+1
                dout.append(
                    (dllr**2 * variances.iloc[i : i + j + 2].values / 4)
                    .sum(axis=1)
                    .sum()
                )
            adelta += np.diagflat(np.array(out), k=j + 1)
            ad_delta += np.diagflat(np.array(dout), k=j + 1)

        # yield standard delta_f_ free energies between each state
        self._delta_f_ = pd.DataFrame(
            adelta - adelta.T, columns=means.index.values, index=means.index.values
        )
        self.dhdl = means

        # yield standard deviation d_delta_f_ between each state
        self._d_delta_f_ = pd.DataFrame(
            np.sqrt(ad_delta + ad_delta.T),
            columns=variances.index.values,
            index=variances.index.values,
        )

        self._states_ = means.index.values.tolist()

        self._delta_f_.attrs = dHdl.attrs
        self._d_delta_f_.attrs = dHdl.attrs
        self.dhdl.attrs = dHdl.attrs

        return self

    def separate_dhdl(self):
        """
        For transitions with multiple lambda, the attr:`dhdl` would return
        a :class:`~pandas.DataFrame` which gives the dHdl for all the lambda
        states, regardless of whether it is perturbed or not. This function
        creates a list of :class:`pandas.Series` for each lambda, where each
        :class:`pandas.Series` describes the potential energy gradient for the
        lambdas state that is perturbed.

        Returns
        ----------
        dHdl_list : list
            A list of :class:`pandas.Series` such that ``dHdl_list[k]`` is the
            potential energy gradient with respect to lambda for each
            configuration that lambda k is perturbed.
        """
        if len(self.dhdl.index.names) == 1:
            name = self.dhdl.columns[0]
            return [
                self.dhdl[name],
            ]
        dhdl_list = []
        # get the lambda names
        l_types = self.dhdl.index.names
        # obtain bool of changed lambdas between each state
        # Fix issue #148, where for pandas == 1.3.0
        # lambdas = self.dhdl.reset_index()[list(l_types)]
        lambdas = self.dhdl.reset_index()[l_types]
        diff = lambdas.diff().to_numpy(dtype="bool")
        # diff will give the first row as NaN so need to fix that
        diff[0, :] = diff[1, :]
        # Make sure that the start point is set to true as well
        diff[:-1, :] = diff[:-1, :] | diff[1:, :]
        for i in range(len(l_types)):
            if any(diff[:, i]):
                new = self.dhdl.iloc[diff[:, i], i]
                # drop all other index
                for l_type in l_types:
                    if l_type != l_types[i]:
                        new = new.reset_index(l_type, drop=True)
                new.attrs = self.dhdl.attrs
                dhdl_list.append(new)
        return dhdl_list
