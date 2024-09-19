class _EstimatorMixOut:
    """This class creates view for the d_delta_f_, delta_f_, states_ for the
    estimator class to consume."""

    _d_delta_f_ = None
    _delta_f_ = None
    _states_ = None
    _d_delta_h_ = None
    _delta_h_ = None
    _d_delta_s_ = None
    _delta_s_ = None

    @property
    def d_delta_f_(self):
        return self._d_delta_f_

    @property
    def delta_f_(self):
        return self._delta_f_

    @property
    def d_delta_h_(self):
        return self._d_delta_h_

    @property
    def delta_h_(self):
        return self._delta_h_

    @property
    def d_delta_s_(self):
        return self._d_delta_s_

    @property
    def delta_s_(self):
        return self._delta_s_

    @property
    def states_(self):
        return self._states_
