"""Functions for outputting dF states.

To assess the quality of the free energy estimation, The dF between adjacent
lambda states can be plotted to assess the quality of the estimation.

The code for producing the dF states plot is modified based on
`Alchemical Analysis <https://github.com/MobleyLab/alchemical-analysis>`_.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties as FP

from ..estimators import TI, BAR, MBAR
from ..postprocessors.units import get_unit_converter


def plot_dF_state(
    estimator, units=None,
):
    """Return the dhdl of TI.

    Parameters
    ----------
    estimator : :class:`~alchemlyb.estimators`
        A :class:`~alchemlyb.estimators`, where the
        dhdl value will be taken from. For more than one estimators
        with more than one alchemical transformation, a list format
        is used.
    units : str
        The unit of the estimate. The default is `None`, which is to use the
        unit in the input. Setting this will change the output unit.

    Returns
    -------
    :class:`pandas.DataFrame`
        The DataFrame with estimate data. ::

               dFE            dFE_Error
            0-1  3.016442       0.052748
            1-2  3.078106       0.037170
            2-3  3.072561       0.030186
            3-4  3.048325       0.026070
            4-5  3.049769       0.023359
            5-6  3.034078       0.021260
            6-7  3.043274       0.019642
            7-8  3.035460       0.018340
            8-9  3.042032       0.017319
            9-10 3.044149       0.016405

    .. versionadded:: 1.0.0
    """
    try:
        len(estimator)
        
    except TypeError:
        estimator = [
            estimator,
        ]

    if units is None:
        units = estimator[0].delta_f_.attrs["energy_unit"]

    # Get the dF
    convert = get_unit_converter(units)
    dF = []
    error = []
    indices = []
    ind = 0
    for j, dhdl in enumerate(estimator):
        for i in range(len(dhdl.delta_f_) - 1):
            indices.append(f"{ind}-{ind}")
            ind += 1
            dF.append(convert(dhdl.delta_f_).iloc[i, i + 1])
            error.append(convert(dhdl.d_delta_f_).iloc[i, i + 1])
            
    dF_state = pd.DataFrame(
        {
            "dstate": indices,
            "dFE": dF,
            "dFE_Error": error,
        }
    )
    dF_state.set_index(["dstate"], inplace=True)
    
    return dF_state

def plot_FE_state(
    estimator, units=None,
):
    """Return free energy trend over the states

    Parameters
    ----------
    estimator : :class:`~alchemlyb.estimators`
        A :class:`~alchemlyb.estimators`, where the
        dhdl value will be taken from. For more than one estimators
        with more than one alchemical transformation, a list format
        is used.
    units : str
        The unit of the estimate. The default is `None`, which is to use the
        unit in the input. Setting this will change the output unit.

    Returns
    -------
    :class:`pandas.DataFrame`
        The DataFrame with estimate data. ::

            lam   FE            FE_Error
            0.0  3.016442       0.052748
            0.1  3.078106       0.037170
            0.2  3.072561       0.030186
            0.3  3.048325       0.026070
            0.4  3.049769       0.023359
            0.5  3.034078       0.021260
            0.6  3.043274       0.019642
            0.7  3.035460       0.018340
            0.8  3.042032       0.017319
            0.9  3.044149       0.016405
            1.0  3.046149       0.015405

    .. versionadded:: 1.0.0
    """
    
    try:
        len(estimator)
        
    except TypeError:
        estimator = [
            estimator,
        ]

    if units is None:
        units = estimator[0].delta_f_.attrs["energy_unit"]

    # Get the dF
    convert = get_unit_converter(units)
    F_list = []
    error = []
    indices = []
    ind = 0
    for dhdl in estimator:
        columns = dhdl.delta_f_.columns
        for i in range(len(dhdl.delta_f_)):
            indices.append(columns[i])
            ind += 1
            F_list.append(convert(dhdl.delta_f_).iloc[i])
            error.append(convert(dhdl.d_delta_f_).iloc[i])
            
    F_state = pd.DataFrame(
        {
            "lambda": indices,
            "FE": F_list,
            "FE_Error": error,
        }
    )
    F_state.set_index(["lambda"], inplace=True)
    
    return F_state