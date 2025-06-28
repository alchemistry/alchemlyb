import pandas as pd


class _EstimatorMixOut:
    """This class creates view for the attributes: states_, delta_f_, d_delta_f_,
    delta_h_, d_delta_h_, delta_sT_, d_delta_sT_ for the estimator class to consume.
    """

    _d_delta_f_: pd.DataFrame | None = None
    _delta_f_: pd.DataFrame | None = None
    _states_: pd.DataFrame | None = None
    _d_delta_h_: pd.DataFrame | None = None
    _delta_h_: pd.DataFrame | None = None
    _d_delta_sT_: pd.DataFrame | None = None
    _delta_sT_: pd.DataFrame | None = None

    @property
    def d_delta_f_(self) -> pd.DataFrame | None:
        return self._d_delta_f_

    @property
    def delta_f_(self) -> pd.DataFrame | None:
        return self._delta_f_

    @property
    def d_delta_h_(self) -> pd.DataFrame | None:
        return self._d_delta_h_

    @property
    def delta_h_(self) -> pd.DataFrame | None:
        return self._delta_h_

    @property
    def d_delta_sT_(self) -> pd.DataFrame | None:
        return self._d_delta_sT_

    @property
    def delta_sT_(self) -> pd.DataFrame | None:
        return self._delta_sT_

    @property
    def states_(self) -> pd.DataFrame | None:
        return self._states_
