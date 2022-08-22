import logging
from warnings import warn

import pandas as pd
import numpy as np

from ..estimators import BAR, TI, FEP_ESTIMATORS, TI_ESTIMATORS
from ..estimators import AutoMBAR as MBAR
from .. import concat


def forward_backward_convergence(df_list, estimator='MBAR', num=10, **kwargs):
    '''Forward and backward convergence of the free energy estimate.

    Generate the free energy estimate as a function of time in both directions,
    with the specified number of equally spaced points in the time. For
    example, setting `num` to 10 would give the forward convergence which is
    the free energy estimate from the first 10%, 20%, 30%, ... of the data. The
    Backward would give the estimate from the last 10%, 20%, 30%, ... of the
    data.

    Parameters
    ----------
    df_list : list
        List of DataFrame of either dHdl or u_nk.
    estimator : {'MBAR', 'BAR', 'TI'}
        Name of the estimators.
    num : int
        The number of time points.
    kwargs : dict
        Keyword arguments to be passed to the estimator.

    Returns
    -------
    DataFrame
        The DataFrame with convergence data. ::

                Forward  Forward_Error  Backward  Backward_Error  data_fraction
            0  3.016442       0.052748  3.065176        0.051036            0.1
            1  3.078106       0.037170  3.078567        0.036640            0.2
            2  3.072561       0.030186  3.047357        0.029775            0.3
            3  3.048325       0.026070  3.057527        0.025743            0.4
            4  3.049769       0.023359  3.037454        0.023001            0.5
            5  3.034078       0.021260  3.040484        0.021075            0.6
            6  3.043274       0.019642  3.032495        0.019517            0.7
            7  3.035460       0.018340  3.036670        0.018261            0.8
            8  3.042032       0.017319  3.046597        0.017233            0.9
            9  3.044149       0.016405  3.044385        0.016402            1.0


    Note
    -----
    :class:`~alchemlyb.estimators.AutoMBAR` is used when ``estimator='MBAR'``,
    supply ``method`` keyword to restore the behavior of
    :class:`~alchemlyb.estimators.MBAR`.
    (:code:`forward_backward_convergence(u_nk, 'MBAR', num=2, method='adaptive')`)


    .. versionadded:: 0.6.0
    .. versionchanged:: 1.0.0
        The ``estimator`` accepts uppercase input.

    '''
    logger = logging.getLogger('alchemlyb.convergence.'
                               'forward_backward_convergence')
    logger.info('Start convergence analysis.')
    logger.info('Check data availability.')
    if estimator.upper() != estimator:
        warn("Using lower-case strings for the 'estimator' kwarg in "
             "convergence.forward_backward_convergence() is deprecated in "
             "1.0.0 and only upper case will be accepted in 2.0.0",
            DeprecationWarning)
        estimator = estimator.upper()

    if estimator not in (FEP_ESTIMATORS + TI_ESTIMATORS):
        msg = f"Estimator {estimator} is not available in {FEP_ESTIMATORS + TI_ESTIMATORS}."
        logger.error(msg)
        raise ValueError(msg)
    else:
        # select estimator class by name
        estimator_fit = globals()[estimator](**kwargs).fit
        logger.info(f'Use {estimator} estimator for convergence analysis.')

    logger.info('Begin forward analysis')
    forward_list = []
    forward_error_list = []
    for i in range(1, num + 1):
        logger.info('Forward analysis: {:.2f}%'.format(100 * i / num))
        sample = []
        for data in df_list:
            sample.append(data[:len(data) // num * i])
        sample = concat(sample)
        result = estimator_fit(sample)
        forward_list.append(result.delta_f_.iloc[0, -1])
        if estimator.lower() == 'bar':
            error = np.sqrt(sum(
                [result.d_delta_f_.iloc[i, i + 1] ** 2
                 for i in range(len(result.d_delta_f_) - 1)]))
            forward_error_list.append(error)
        else:
            forward_error_list.append(result.d_delta_f_.iloc[0, -1])
        logger.info('{:.2f} +/- {:.2f} kT'.format(forward_list[-1],
                                                  forward_error_list[-1]))

    logger.info('Begin backward analysis')
    backward_list = []
    backward_error_list = []
    for i in range(1, num + 1):
        logger.info('Backward analysis: {:.2f}%'.format(100 * i / num))
        sample = []
        for data in df_list:
            sample.append(data[-len(data) // num * i:])
        sample = concat(sample)
        result = estimator_fit(sample)
        backward_list.append(result.delta_f_.iloc[0, -1])
        if estimator.lower() == 'bar':
            error = np.sqrt(sum(
                [result.d_delta_f_.iloc[i, i + 1] ** 2
                 for i in range(len(result.d_delta_f_) - 1)]))
            backward_error_list.append(error)
        else:
            backward_error_list.append(result.d_delta_f_.iloc[0, -1])
        logger.info('{:.2f} +/- {:.2f} kT'.format(backward_list[-1],
                                                  backward_error_list[-1]))

    convergence = pd.DataFrame(
        {'Forward': forward_list,
         'Forward_Error': forward_error_list,
         'Backward': backward_list,
         'Backward_Error': backward_error_list,
         'data_fraction': [i / num for i in range(1, num + 1)]})
    convergence.attrs = df_list[0].attrs
    return convergence
