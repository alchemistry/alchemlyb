from .bar_ import BAR
from .mbar_ import MBAR
from .ti_ import TI
from .ti_gaussian_quadrature import TI_gq

FEP_ESTIMATORS = [MBAR.__name__, BAR.__name__]
TI_ESTIMATORS = [TI.__name__, TI_gq.__name__]
