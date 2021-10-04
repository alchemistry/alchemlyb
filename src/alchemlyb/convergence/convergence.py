import pandas as pd
import logging
import numpy as np

from ..estimators import MBAR, BAR, TI
from .. import concat


def forward_backward_convergence(df_list, estimator='mbar', num=10):
    ''' The forward and backward convergence of the free energy estimate.

    Generate the free energy change as a function of time in both
    directions, with the specified number of points in the time.

    Parameters
    ----------
    df_list : list
        List of DataFrame of either dHdl or u_nk.
    estimator : {'mbar', 'bar', 'ti'}
        Name of the estimators.
    num : int
        The number of time points.

    Returns
    -------
    DataFrame
        The DataFrame with convergence data. ::

                         Forward  Forward_Error  Backward  Backward_Error
            t_fraction
            1/10        3.067943       0.070175  3.111035        0.067088
            2/10        3.122223       0.049303  3.126450        0.048173
            3/10        3.117742       0.039916  3.094115        0.039099
            4/10        3.091870       0.034389  3.101558        0.033783
            5/10        3.093778       0.030814  3.082714        0.030148
            6/10        3.079128       0.027999  3.085972        0.027652
            7/10        3.086951       0.025847  3.077004        0.025610
            8/10        3.079147       0.024122  3.081519        0.023968
            9/10        3.086575       0.022778  3.090475        0.022633
            10/10       3.088821       0.021573  3.089027        0.021568


    .. versionadded:: 0.6.0
    '''
    logger = logging.getLogger('alchemlyb.postprocessors.'
                               'forward_backward_convergence')
    logger.info('Start convergence analysis.')
    logger.info('Check data availability.')

    if estimator.lower() == 'mbar':
        logger.info('Use MBAR estimator for convergence analysis.')
        estimator_fit = MBAR().fit
    elif estimator.lower() == 'bar':
        logger.info('Use BAR estimator for convergence analysis.')
        estimator_fit = BAR().fit
    elif estimator.lower() == 'ti':
        logger.info('Use TI estimator for convergence analysis.')
        estimator_fit = TI().fit
    else:  # pragma: no cover
        logger.warning(
            '{} is not a valid estimator.'.format(estimator))

    logger.info('Begin forward analysis')
    forward_list = []
    forward_error_list = []
    for i in range(1, num + 1):
        logger.info('Forward analysis: {:.2f}%'.format(i / num))
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
        logger.info('Backward analysis: {:.2f}%'.format(i / num))
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
         'Backward_Error': backward_error_list},
        index=['{}/{}'.format(i, num) for i in range(1, num + 1)])
    convergence.index.name = 't_fraction'
    convergence.attrs = df_list[0].attrs
    return convergence
