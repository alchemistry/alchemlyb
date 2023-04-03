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

    _gq_delta_f_ : DataFrame, optional
        The estimated dimensionless cumulative sum of free energy up to each state 
        with gaussian quadrature.

    _gq_d_delta_f_ : DataFrame, optional
        The estimated statistical uncertainty (one standard deviation) in 
        the cumulative sum of free energy up to each state with gaussian quadrature

    .. versionchanged:: 1.0.0
       `delta_f_`, `d_delta_f_`, `states_` are view of the original object.

    """

    def __init__(self, verbose=False, gauss_qua=False):
        self.verbose = verbose
        self.gauss_qua = gauss_qua

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

        if self.gauss_qua:
            self.gaussian_quadrature(dHdl, means=means, variances=variances)

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

    def gaussian_quadrature(self, dHdl, means, variances):
        """
        numerically estimate the free energy using gaussian quadrature with 
        lambda values suggested in Amber manual

        Parameters
        ----------
        dHdl : DataFrame
        means : DataFrame
        variances: DataFrame
        """
        # extract the lambda values used in the simulations
        lambdas = means.reset_index()[means.index.names[:]].iloc[:].values.sum(axis=1)
        num_lambdas = len(lambdas)
        # check if the lambda values used match the common ones suggested in Amber manual and assign cooresponding weights
        if num_lambdas==1:
            try:
                reference_lambdas = np.round([0.5], 4)
                assert (lambdas == reference_lambdas).all()
                weights = np.array([1.0])
            except:
                print("Error: the lambda values for gaussian quadrature are not supported yet, please use trapezoidal rule instead")
                weights = None

        elif num_lambdas==2:
            try:
                reference_lambdas = np.round([0.21132, 0.78867], 4)
                assert (lambdas == reference_lambdas).all()
                weights = np.array([0.5, 0.5])
            except:
                print("Error: the lambda values for gaussian quadrature are not supported yet, please use trapezoidal rule instead")
                weights = None

        elif num_lambdas==3:
            try:
                reference_lambdas = np.round([0.1127, 0.5, 0.88729], 4)
                assert (lambdas == reference_lambdas).all()
                weights = np.array([0.27777, 0.44444, 0.27777])
            except:
                print("Error: the lambda values for gaussian quadrature are not supported yet, please use trapezoidal rule instead")
                weights = None

        elif num_lambdas==5:
            try:
                reference_lambdas = np.round([0.04691, 0.23076, 0.5, 0.76923, 0.95308], 4)
                assert (lambdas == reference_lambdas).all()
                weights = np.array([0.11846, 0.23931, 0.28444, 0.23931, 0.11846])
            except:
                print("Error: the lambda values for gaussian quadrature are not supported yet, please use trapezoid rule instead")
                weights = None

        elif num_lambdas==7:
            try:
                reference_lambdas = np.round([0.02544, 0.12923, 0.29707, 0.5, 0.70292, 0.87076, 0.97455], 4)
                assert (lambdas == reference_lambdas).all()
                weights = np.array([0.06474, 0.13985, 0.19091, 0.20897, 0.19091, 0.13985, 0.06474])
            except:
                print("Error: the lambda values for gaussian quadrature are not supported yet, please use trapezoid rule instead")
                weights = None

        elif num_lambdas==9:
            try:
                reference_lambdas = np.round([0.01592, 0.08198, 0.19331, 0.33787, 0.5,
                                               0.66213, 0.80669, 0.91802, 0.98408], 4)
                assert (lambdas == reference_lambdas).all()
                weights = np.array([0.04064, 0.09032, 0.13031, 0.15617, 0.16512, 0.15617, 0.13031, 0.09032, 0.04064])
            except:
                print("Error: the lambda values for gaussian quadrature are not supported yet, please use trapezoid rule instead")
                weights = None

        elif num_lambdas==12:
            try:
                reference_lambdas = np.round([0.00922, 0.04794, 0.11505, 0.20634, 0.31608, 0.43738, 
                                              0.56262, 0.68392, 0.79366, 0.88495, 0.95206, 0.99078], 4)
                assert (lambdas == reference_lambdas).all()
                weights = np.array([0.02359, 0.05347, 0.08004, 0.10158, 0.11675, 0.12457,
                                     0.12457, 0.11675, 0.10158, 0.08004, 0.05347, 0.02359])
            except:
                print("Error: the lambda values for gaussian quadrature are not supported yet, please use trapezoid rule instead")
                weights = None

        else:
            print("Error: the lambda values for gaussian quadrature are not supported yet, please use trapezoid rule instead")
            weights = None
        # perform gassian quadrature to esstimate the thermaldynamic integration
        if weights is not None:
            mean_values = means.values.sum(axis=1)
            variance_values = variances.values.sum(axis=1)
            gq_deltas = weights * mean_values
            gq_d_deltas_squared = np.square(weights) * variance_values
            gq_adelta = np.zeros((len(gq_deltas), len(gq_deltas)))
            gq_ad_delta = np.zeros_like(gq_adelta)
            for j in range(len(gq_deltas)):
                out = []
                dout = []
                for i in range(len(gq_deltas) - j):
                    # Append cumulative free energy value from state i to i+j+1
                    out.append(gq_deltas[i] + gq_deltas[i + 1 : i + j + 1].sum())
                    # Append cumulative deviation of free energy from state i to i+j+1
                    dout.append(gq_d_deltas_squared[i] + gq_d_deltas_squared[i + 1 : i + j + 1].sum())
            
                gq_adelta += np.diagflat(np.array(out), k=j)
                gq_ad_delta += np.diagflat(np.array(dout), k=j)

            self._gq_delta_f_ = pd.DataFrame(
                gq_adelta, columns=means.index.values, index=means.index.values
            )
            self._gq_d_delta_f_ = pd.DataFrame(
                np.sqrt(gq_ad_delta),
                columns=variances.index.values,
                index=variances.index.values,
            )
            self._gq_delta_f_.attrs = dHdl.attrs
            self._gq_d_delta_f_.attrs = dHdl.attrs

        else:
            self._gq_delta_f_, self._gq_d_delta_f_ = None, None

        return self





    