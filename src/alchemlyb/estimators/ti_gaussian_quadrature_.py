import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from .. import concat
from .base import _EstimatorMixOut


class TI_GQ(BaseEstimator, _EstimatorMixOut):
    """Thermodynamic integration (TI) with gaussian quadrature estimation.
       When the simulations are performed at certain gaussian quadrature
       points (lambdas), the free energy can be estimated using gaussian 
       quadrature as an alternative to the trapezoidal rule.

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


    .. versionadded:: 2.0.0
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
        self.means = dHdl.groupby(level=dHdl.index.names[1:]).mean()
        self.variances = np.square(dHdl.groupby(level=dHdl.index.names[1:]).sem())
        # suggested lambda values and corresponding weights 
        special_points = self.get_quadrature_points()
        weights = []
        # check if the lambdas in the simulations match the suggested values
        lambda_list, means_list, variances_list = self.separate_dhdl()
        for lambdas in lambda_list:
            num_lambdas = len(lambdas)
            if (num_lambdas not in special_points) or (not np.allclose(lambdas, special_points[num_lambdas][0], rtol=0.1)):
                raise ValueError('The lambda values for gaussian quadrature are not supported, please use trapezoid rule instead.')
            weights.extend(special_points[num_lambdas][1])
        # means_new and variances_new are similar to means and variances, but with only values relevant to each lambda type (for multi-lambda situation)
        means_new = concat(means_list)
        mean_values = means_new.to_numpy()
        variances_new = concat(variances_list)
        variance_values = variances_new.to_numpy()

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
            adelta, columns=means_new.index.values, index=means_new.index.values
        )
        
        # yield standard deviation d_delta_f_ between each state
        self._d_delta_f_ = pd.DataFrame(
            np.sqrt(ad_delta),
            columns=variances_new.index.values,
            index=variances_new.index.values,
        )
        self.dhdl = self.means
        self.dhdl.attrs = dHdl.attrs
        self._states_ = means_new.index.values.tolist()
        self._delta_f_.attrs = dHdl.attrs
        self._d_delta_f_.attrs = dHdl.attrs

        return self

    def separate_dhdl(self):
        """
        For transitions with multiple lambda, the attr:`dhdl` would return
        a :class:`~pandas.DataFrame` which gives the dHdl for all the lambda
        states, regardless of whether it is perturbed or not. This function
        creates 3 lists of :class:`numpy.array`, :class:`numpy.array` and :class:`pandas.Series` 
        for each lambda, where the lists describe the lambda values, masks of potential gaussian 
        quadrature points, and potential energy gradient for the lambdas state that is perturbed.

        Returns
        ----------
        lambda_list : list
            A list of :class:`numpy.array` such that ``lambda_list[k]`` is the
            lambda values with respect to each type of lambda.
        mask_list : list
            A list of :class:`numpy.array` such that ``lambda_list[k]`` is the
            lambda mask with respect to each type of lambda. The lambdas values
            between 0.0 and 1.0 are marked as True as potential gaussian quadrature
            points. 
        dHdl_list : list
            A list of :class:`pandas.Series` such that ``dHdl_list[k]`` is the
            potential energy gradient with respect to lambda for each
            configuration that lambda k is perturbed.
        """
        lambda_list = []
        dhdl_list = []
        variance_list = []
        # get the lambda names
        l_types = self.means.index.names
        # get the lambda vaules
        lambdas = self.means.reset_index()[self.means.index.names].values

        if len(self.means.index.names) == 1:
            name = self.means.columns[0]
            lambda_list.append(self.means.index)
            dhdl_list.append(self.means[name])
            variance_list.append(self.variances[name])

        else:
            # simultanouesly scaling of multiple lambda types are not supported
            if (((0.0 < lambdas) & (lambdas < 1.0)).sum(axis=1) > 1.0).any():
                raise ValueError('The lambda values for gaussian quadrature are not supported, please use trapezoid rule instead.')
            for i in range(len(l_types)):
                # obtain the lambda points between 0.0 and 1.0
                l_masks = (0.0 < lambdas[:, i]) & (lambdas[:, i] < 1.0)
                if not l_masks.any():
                    continue
                new_means = self.means.iloc[l_masks, i]
                new_variances = self.variances.iloc[l_masks, i]
                for l in l_types:
                    if l != l_types[i]:
                        new_means = new_means.reset_index(l, drop=True)
                        new_variances = new_variances.reset_index(l, drop=True)
                new_means.attrs = self.means.attrs
                new_variances.attrs = self.variances.attrs
                lambda_list.append(new_means.index)
                dhdl_list.append(new_means)
                variance_list.append(new_variances)

        return lambda_list, dhdl_list, variance_list
    
    def get_quadrature_points(self):
        """
        Obtain suggested gaussian quadrature points (lambda values) and 
        corresponding weights

        Returns
        ----------
        quadrature_points: dict
            A dictionary of :class:`list` such that the keys are the number of
            lambda points and the values are the lists of lambdas and weights
        """
        quadrature_points = {1: [[0.5], 
                                 [1.0]], 
                             2: [[0.21132, 0.78867], 
                                 [0.5, 0.5]],
                             3: [[0.1127, 0.5, 0.88729], 
                                 [0.27777, 0.44444, 0.27777]],
                             4: [[0.06943, 0.33001, 0.66999, 0.93057], 
                                 [0.17393, 0.32607, 0.32607, 0.17393]],
                             5: [[0.04691, 0.23076, 0.5, 0.76923, 0.95308], 
                                 [0.11846, 0.23931, 0.28444, 0.23931, 0.11846]], 
                             6: [[0.03377, 0.1694 , 0.38069, 0.61931, 0.8306 , 0.96623], 
                                 [0.08566, 0.18038, 0.23396, 0.23396, 0.18038, 0.08566]],
                             7: [[0.02544, 0.12923, 0.29707, 0.5, 0.70292, 0.87076, 0.97455], 
                                 [0.06474, 0.13985, 0.19091, 0.20897, 0.19091, 0.13985, 0.06474]],
                             8: [[0.01986, 0.10167, 0.23723, 0.40828, 0.59172, 0.76277, 0.89833, 0.98014],
                                 [0.05061, 0.11119, 0.15685, 0.18134, 0.18134, 0.15685, 0.11119, 0.05061]],
                             9: [[0.01592, 0.08198, 0.19331, 0.33787, 0.5, 0.66213, 0.80669, 0.91802, 0.98408],
                                 [0.04064, 0.09032, 0.13031, 0.15617, 0.16512, 0.15617, 0.13031, 0.09032, 0.04064]],
                            10: [[0.01305, 0.06747, 0.1603, 0.2833, 0.42556, 0.57444, 0.7167, 0.8397, 0.93253, 0.98695],
                                 [0.03334, 0.07473, 0.10954, 0.13463, 0.14776, 0.14776, 0.13463, 0.10954, 0.07473, 0.03334]],
                            11: [[0.01089, 0.05647, 0.13492, 0.24045, 0.36523, 0.5, 0.63477, 0.75955, 0.86508, 0.94353, 0.98911],
                                 [0.02783, 0.06279, 0.09315, 0.1166, 0.1314, 0.13646, 0.1314, 0.1166, 0.09315, 0.06279, 0.02783]],
                            12: [[0.00922, 0.04794, 0.11505, 0.20634, 0.31608, 0.43738, 0.56262, 0.68392, 0.79366, 0.88495, 0.95206, 0.99078],
                                 [0.02359, 0.05347, 0.08004, 0.10158, 0.11675, 0.12457, 0.12457, 0.11675, 0.10158, 0.08004, 0.05347, 0.02359]],
                            13: [[0.00791, 0.0412, 0.09921, 0.17883, 0.27575, 0.38477, 0.5, 0.61523, 0.72425, 0.82117, 0.90079, 0.9588, 0.99209],
                                 [0.02024, 0.04606, 0.06944, 0.08907, 0.10391, 0.11314, 0.11628, 0.11314, 0.10391, 0.08907, 0.06944, 0.04606, 0.02024]],
                            14: [[0.00686, 0.03578, 0.0864, 0.15635, 0.24238, 0.34044, 0.44597, 0.55403, 0.65956, 0.75762, 0.84365, 0.9136, 0.96422, 0.99314],
                                 [0.01756, 0.04008, 0.06076, 0.0786, 0.09277, 0.1026, 0.10763, 0.10763, 0.1026, 0.09277, 0.0786, 0.06076, 0.04008, 0.01756]],
                            15: [[0.006, 0.03136, 0.0759, 0.13779, 0.21451, 0.30292, 0.3994 , 0.5, 0.6006, 0.69708, 0.78549, 0.86221, 0.9241, 0.96864, 0.994],
                                 [0.01538, 0.03518, 0.05358, 0.06979, 0.08313, 0.09308, 0.09922, 0.10129, 0.09922, 0.09308, 0.08313, 0.06979, 0.05358, 0.03518, 0.01538]],
                            16: [[0.0053, 0.02771, 0.06718, 0.1223, 0.19106, 0.27099, 0.3592, 0.45249, 0.54751, 0.6408, 0.72901, 0.80894, 0.8777, 0.93282, 0.97229, 0.9947],
                                 [0.01358, 0.03113, 0.04758, 0.06231, 0.0748, 0.08458, 0.0913 , 0.09473, 0.09473, 0.0913, 0.08458, 0.0748, 0.06231, 0.04758, 0.03113, 0.01358]]
                         }
        return quadrature_points
    