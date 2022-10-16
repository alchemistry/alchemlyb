class EstimatorMixOut():
    '''This class creates view for the d_delta_f_, delta_f_, states_ for the
    estimator class to consume.'''
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

from .mbar_ import MBAR, AutoMBAR
from .bar_ import BAR
from .ti_ import TI

FEP_ESTIMATORS = [MBAR.__name__, AutoMBAR.__name__, BAR.__name__]
TI_ESTIMATORS = [TI.__name__]