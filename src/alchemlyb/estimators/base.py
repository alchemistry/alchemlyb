class _EstimatorMixOut:
    """This class creates view for the d_delta_f_, delta_f_, states_ for the
    estimator class to consume."""

    _d_delta_f_ = None
    _delta_f_ = None
    _states_ = None

    @property
    def d_delta_f_(self):
        return self._d_delta_f_

    @property
    def delta_f_(self):
        return self._delta_f_

    @property
    def states_(self):
        return self._states_
