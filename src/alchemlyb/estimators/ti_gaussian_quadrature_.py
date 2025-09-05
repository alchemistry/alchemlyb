import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import numpy.typing as npt
from .. import concat
from .base import _EstimatorMixOut


class TI_GQ(BaseEstimator, _EstimatorMixOut):
    """Thermodynamic integration (TI) with gaussian quadrature estimation.

    Parameters
    ----------

    verbose : bool, optional
        Set to True if verbose debug output is desired.

    Attributes
    ----------

    delta_f_ : DataFrame
        The estimated cumulative free energy from one state to another.

    d_delta_f_ : DataFrame
        The estimated statistical uncertainty (one standard deviation) in
        dimensionless cumulative free energies.

    states_ : list
        Lambda states for which free energy estimation were obtained.

    dhdl : DataFrame
        The estimated dhdl of each state.


    .. versionadded:: 2.1.0

    """

    special_points = {
        1: {"lambdas": [0.5], "weights": [1.0]},
        2: {"lambdas": [0.21132, 0.78867], "weights": [0.5, 0.5]},
        3: {"lambdas": [0.1127, 0.5, 0.88729], "weights": [0.27777, 0.44444, 0.27777]},
        4: {
            "lambdas": [0.06943, 0.33001, 0.66999, 0.93057],
            "weights": [0.17393, 0.32607, 0.32607, 0.17393],
        },
        5: {
            "lambdas": [0.04691, 0.23076, 0.5, 0.76923, 0.95308],
            "weights": [0.11846, 0.23931, 0.28444, 0.23931, 0.11846],
        },
        6: {
            "lambdas": [0.03377, 0.1694, 0.38069, 0.61931, 0.8306, 0.96623],
            "weights": [0.08566, 0.18038, 0.23396, 0.23396, 0.18038, 0.08566],
        },
        7: {
            "lambdas": [0.02544, 0.12923, 0.29707, 0.5, 0.70292, 0.87076, 0.97455],
            "weights": [0.06474, 0.13985, 0.19091, 0.20897, 0.19091, 0.13985, 0.06474],
        },
        8: {
            "lambdas": [
                0.01986,
                0.10167,
                0.23723,
                0.40828,
                0.59172,
                0.76277,
                0.89833,
                0.98014,
            ],
            "weights": [
                0.05061,
                0.11119,
                0.15685,
                0.18134,
                0.18134,
                0.15685,
                0.11119,
                0.05061,
            ],
        },
        9: {
            "lambdas": [
                0.01592,
                0.08198,
                0.19331,
                0.33787,
                0.5,
                0.66213,
                0.80669,
                0.91802,
                0.98408,
            ],
            "weights": [
                0.04064,
                0.09032,
                0.13031,
                0.15617,
                0.16512,
                0.15617,
                0.13031,
                0.09032,
                0.04064,
            ],
        },
        10: {
            "lambdas": [
                0.01305,
                0.06747,
                0.1603,
                0.2833,
                0.42556,
                0.57444,
                0.7167,
                0.8397,
                0.93253,
                0.98695,
            ],
            "weights": [
                0.03334,
                0.07473,
                0.10954,
                0.13463,
                0.14776,
                0.14776,
                0.13463,
                0.10954,
                0.07473,
                0.03334,
            ],
        },
        11: {
            "lambdas": [
                0.01089,
                0.05647,
                0.13492,
                0.24045,
                0.36523,
                0.5,
                0.63477,
                0.75955,
                0.86508,
                0.94353,
                0.98911,
            ],
            "weights": [
                0.02783,
                0.06279,
                0.09315,
                0.1166,
                0.1314,
                0.13646,
                0.1314,
                0.1166,
                0.09315,
                0.06279,
                0.02783,
            ],
        },
        12: {
            "lambdas": [
                0.00922,
                0.04794,
                0.11505,
                0.20634,
                0.31608,
                0.43738,
                0.56262,
                0.68392,
                0.79366,
                0.88495,
                0.95206,
                0.99078,
            ],
            "weights": [
                0.02359,
                0.05347,
                0.08004,
                0.10158,
                0.11675,
                0.12457,
                0.12457,
                0.11675,
                0.10158,
                0.08004,
                0.05347,
                0.02359,
            ],
        },
        13: {
            "lambdas": [
                0.00791,
                0.0412,
                0.09921,
                0.17883,
                0.27575,
                0.38477,
                0.5,
                0.61523,
                0.72425,
                0.82117,
                0.90079,
                0.9588,
                0.99209,
            ],
            "weights": [
                0.02024,
                0.04606,
                0.06944,
                0.08907,
                0.10391,
                0.11314,
                0.11628,
                0.11314,
                0.10391,
                0.08907,
                0.06944,
                0.04606,
                0.02024,
            ],
        },
        14: {
            "lambdas": [
                0.00686,
                0.03578,
                0.0864,
                0.15635,
                0.24238,
                0.34044,
                0.44597,
                0.55403,
                0.65956,
                0.75762,
                0.84365,
                0.9136,
                0.96422,
                0.99314,
            ],
            "weights": [
                0.01756,
                0.04008,
                0.06076,
                0.0786,
                0.09277,
                0.1026,
                0.10763,
                0.10763,
                0.1026,
                0.09277,
                0.0786,
                0.06076,
                0.04008,
                0.01756,
            ],
        },
        15: {
            "lambdas": [
                0.006,
                0.03136,
                0.0759,
                0.13779,
                0.21451,
                0.30292,
                0.3994,
                0.5,
                0.6006,
                0.69708,
                0.78549,
                0.86221,
                0.9241,
                0.96864,
                0.994,
            ],
            "weights": [
                0.01538,
                0.03518,
                0.05358,
                0.06979,
                0.08313,
                0.09308,
                0.09922,
                0.10129,
                0.09922,
                0.09308,
                0.08313,
                0.06979,
                0.05358,
                0.03518,
                0.01538,
            ],
        },
        16: {
            "lambdas": [
                0.0053,
                0.02771,
                0.06718,
                0.1223,
                0.19106,
                0.27099,
                0.3592,
                0.45249,
                0.54751,
                0.6408,
                0.72901,
                0.80894,
                0.8777,
                0.93282,
                0.97229,
                0.9947,
            ],
            "weights": [
                0.01358,
                0.03113,
                0.04758,
                0.06231,
                0.0748,
                0.08458,
                0.0913,
                0.09473,
                0.09473,
                0.0913,
                0.08458,
                0.0748,
                0.06231,
                0.04758,
                0.03113,
                0.01358,
            ],
        },
    }

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def fit(self, dHdl: pd.DataFrame) -> "TI_GQ":
        """
        Compute cumulative free energy from one state to another by integrating
        dHdl across lambda values.

        Parameters
        ----------
        dHdl : DataFrame
            dHdl[n,k] is the potential energy gradient with respect to lambda
            for each configuration n and lambda k.

        """

        # sort by state so that rows from same state are in contiguous blocks,
        # and adjacent states are next to each other
        dHdl = dHdl.sort_index(level=dHdl.index.names[1:])  # type: ignore[arg-type]

        # obtain the mean and variance of the mean for each state
        # variance calculation assumes no correlation between points
        # used to calculate mean
        means = dHdl.groupby(level=dHdl.index.names[1:]).mean()
        variances = np.square(dHdl.groupby(level=dHdl.index.names[1:]).sem())

        weights = []
        # check if the lambdas in the simulations match the suggested values
        lambda_list, means_list, variances_list, index_list = (
            self.separate_mean_variance(means, variances)  # type: ignore[arg-type]
        )
        for lambdas in lambda_list:
            num_lambdas = len(lambdas)
            if num_lambdas not in self.special_points:
                raise ValueError(
                    f"TI_GQ only supports a set number of lambda windows ({list(self.special_points.keys())}) currently, \
                                 but {num_lambdas} lambda windows are given."
                )
            suggested_lambdas = self.special_points[num_lambdas]["lambdas"]
            if not np.allclose(lambdas, suggested_lambdas, rtol=0.1):
                raise ValueError(
                    f"lambda values, {suggested_lambdas}, are expected, but {lambdas} are given. Please use trapezoidal rule instead."
                )
            weights.extend(self.special_points[num_lambdas]["weights"])
        # means_new and variances_new are similar to means and variances, but with only values relevant to each lambda type (for multi-lambda situation)
        means_new = concat(means_list)
        mean_values = means_new.to_numpy()
        variances_new = concat(variances_list)
        variance_values = variances_new.to_numpy()

        # apply gaussian quadrature multiplication at each lambda state
        deltas = weights * mean_values
        deltas = np.insert(deltas, 0, [0.0], axis=0)  # type: ignore[assignment]
        deltas = np.append(deltas, [0.0], axis=0)  # type: ignore[assignment]
        d_deltas_squared = np.square(weights) * variance_values
        d_deltas_squared = np.insert(d_deltas_squared, 0, [0.0], axis=0)
        d_deltas_squared = np.append(d_deltas_squared, [0.0], axis=0)
        # build matrix of deltas between each state
        adelta = np.zeros((len(deltas), len(deltas)))
        ad_delta = np.zeros_like(adelta)

        for j in range(len(deltas)):
            out = []
            dout = []
            for i in range(len(deltas) - j):
                # Append cumulative free energy value from state i to i+j
                out.append(deltas[i] + deltas[i + 1 : i + j + 1].sum())  # type: ignore[attr-defined]
                # Append cumulative squared deviation of free energy from state i to i+j
                dout.append(
                    d_deltas_squared[i] + d_deltas_squared[i + 1 : i + j + 1].sum()
                )

            adelta += np.diagflat(np.array(out), k=j)
            ad_delta += np.diagflat(np.array(dout), k=j)

        adelta = adelta - adelta.T  # type: ignore[assignment]
        ad_delta = (ad_delta + ad_delta.T) - 2 * np.diagflat(d_deltas_squared)
        # yield standard delta_f_ cumulative free energies from one state to another
        self._delta_f_ = pd.DataFrame(adelta, columns=index_list, index=index_list)

        # yield standard deviation d_delta_f_ between each state
        self._d_delta_f_ = pd.DataFrame(
            np.sqrt(ad_delta),
            columns=index_list,
            index=index_list,
        )

        self.dhdl = means
        self.dhdl.attrs = dHdl.attrs
        self._states_ = means_new.index.values.tolist()
        self._delta_f_.attrs = dHdl.attrs
        self._d_delta_f_.attrs = dHdl.attrs

        return self

    @staticmethod
    def separate_mean_variance(
        means: pd.DataFrame, variances: pd.DataFrame
    ) -> tuple[
        list[pd.Index],
        list[pd.Series],
        list[pd.Series],
        list[float | tuple[float, ...]],
    ]:
        """
        For transitions with multiple lambda, the attr:`dhdl` would return
        a :class:`~pandas.DataFrame` which gives the dHdl for all the lambda
        states, regardless of whether it is perturbed or not. This function
        creates 3 lists of :class:`numpy.array`, :class:`pandas.Series` and
        :class:`pandas.Series` for each lambda, where the lists describe
        the lambda values, potential energy gradient and variance values for
        the lambdas state that is perturbed.

        Parameters
        ----------
        means: DataFrame
            means is the average potential energy gradient at each lambda.
        variances: DataFrame
            variances is variance of the potential energy gradient at each lambda.

        Returns
        ----------
        lambda_list : list
            A list of :class:`numpy.array` such that ``lambda_list[k]`` is the
            lambda values with respect to each type of lambda.
        dhdl_list : list
            A list of :class:`pandas.Series` such that ``dHdl_list[k]`` is the
            potential energy gradient with respect to lambda for each
            configuration that lambda k is perturbed.
        variance_list : list
            A list of :class:`pandas.Series` such that ``variance_list[k]`` is the
            variance of the potential energy gradient with respect to lambda for each
            configuration that lambda k is perturbed.
        index_list : list
            A list of :class:`float` or :class:`tuple` such that each :class:`float`
            or :class:`tuple` is the index of the final `delta_f_` and `d_delta_f_`
        """
        lambda_list: list[npt.NDArray] = []
        dhdl_list: list[pd.Series] = []
        variance_list: list[pd.Series] = []
        index_list: list[float | tuple[float]] = []
        # get the lambda names
        l_types = means.index.names
        # get the lambda vaules
        lambdas = means.reset_index()[means.index.names].values

        for i in range(len(l_types)):
            # obtain the lambda points between 0.0 and 1.0
            l_masks = (0.0 < lambdas[:, i]) & (lambdas[:, i] < 1.0)
            if not l_masks.any():
                continue
            new_means = means.iloc[l_masks, i]
            new_variances = variances.iloc[l_masks, i]
            index_list.extend(new_means.index)
            # for multi-lambda case, extract the relevant column
            for l_type in l_types:
                if l_type != l_types[i]:
                    new_means = new_means.reset_index(l_type, drop=True)
                    new_variances = new_variances.reset_index(l_type, drop=True)
            new_means.attrs = means.attrs
            new_variances.attrs = variances.attrs
            lambda_list.append(new_means.index)  # type: ignore[arg-type]
            dhdl_list.append(new_means)
            variance_list.append(new_variances)

        # add two end states at all lambda zeros and ones
        if len(l_types) == 1:
            index_list = [0.0] + index_list + [1.0]
        else:
            index_list = (
                [tuple([0.0] * len(l_types))]  # type: ignore[assignment]
                + index_list
                + [tuple([1.0] * len(l_types))]
            )

        return lambda_list, dhdl_list, variance_list, index_list  # type: ignore[return-value]
