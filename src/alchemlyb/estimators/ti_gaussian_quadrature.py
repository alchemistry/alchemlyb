import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .base import _EstimatorMixOut


class TI_gq(BaseEstimator, _EstimatorMixOut):
    """Thermodynamic integration (TI) with gaussian quadrature estimation.

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
        Lambda states for which free energy estimation were obtained.

    dhdl : DataFrame
        The estimated dhdl of each state.


    .. versionchanged:: 2.0.0
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

        # extract the lambda values used in the simulations
        lambdas = means.reset_index()[means.index.names[:]].iloc[:].values.sum(axis=1)
        num_lambdas = len(lambdas)
        # suggested lambda values corresponding weights 
        special_points = {1: [[0.5], [1.0]], 2: [[0.21132, 0.78867], [0.5, 0.5]],
                          3: [[0.1127, 0.5, 0.88729], [0.27777, 0.44444, 0.27777]],
                          5: [[0.04691, 0.23076, 0.5, 0.76923, 0.95308], [0.11846, 0.23931, 0.28444, 0.23931, 0.11846]], 
                          7: [[0.02544, 0.12923, 0.29707, 0.5, 0.70292, 0.87076, 0.97455], 
                              [0.06474, 0.13985, 0.19091, 0.20897, 0.19091, 0.13985, 0.06474]],
                          9: [[0.01592, 0.08198, 0.19331, 0.33787, 0.5, 0.66213, 0.80669, 0.91802, 0.98408],
                              [0.04064, 0.09032, 0.13031, 0.15617, 0.16512, 0.15617, 0.13031, 0.09032, 0.04064]],
                         12: [[0.00922, 0.04794, 0.11505, 0.20634, 0.31608, 0.43738, 0.56262, 0.68392, 0.79366, 0.88495, 0.95206, 0.99078],
                              [0.02359, 0.05347, 0.08004, 0.10158, 0.11675, 0.12457, 0.12457, 0.11675, 0.10158, 0.08004, 0.05347, 0.02359]]
                         }
        if (num_lambdas not in special_points) or (not np.allclose(lambdas, special_points[num_lambdas][0], rtol=0.1)):
            raise ValueError('The lambda values for gaussian quadrature are not supported, please use trapezoid rule instead.') 
        
        weights = special_points[num_lambdas][1]
        mean_values = means.values.sum(axis=1)
        variance_values = variances.values.sum(axis=1)

        # apply gaussian quadrature multiplication at each lambda state 
        deltas = weights * mean_values
        d_deltas_squared = np.square(weights) * variance_values
        # build matrix of deltas between each state
        adelta = np.zeros((len(deltas), len(deltas)))
        ad_delta = np.zeros_like(adelta)

        for j in range(len(deltas)):
            out = []
            dout = []
            for i in range(len(deltas) - j):
                # Append cumulative free energy value from state i to i+j
                out.append(deltas[i] + deltas[i + 1 : i + j + 1].sum())
                # Append cumulative squared deviation of free energy from state i to i+j
                dout.append(d_deltas_squared[i] + d_deltas_squared[i + 1 : i + j + 1].sum())
            
            adelta += np.diagflat(np.array(out), k=j)
            ad_delta += np.diagflat(np.array(dout), k=j)

        # yield standard delta_f_ cumulative free energies from one state to another
        self._delta_f_ = pd.DataFrame(
            adelta, columns=means.index.values, index=means.index.values
        )
        self.dhdl = means
        # yield standard deviation d_delta_f_ between each state
        self._d_delta_f_ = pd.DataFrame(
            np.sqrt(ad_delta),
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
                for l in l_types:
                    if l != l_types[i]:
                        new = new.reset_index(l, drop=True)
                new.attrs = self.dhdl.attrs
                dhdl_list.append(new)
        return dhdl_list
