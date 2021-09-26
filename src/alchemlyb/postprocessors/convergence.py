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
               Forward       F. Error       Backward       B. Error
            0      33.988935        0.334676       35.666128        0.324426
            1      35.075489        0.232150       35.382850        0.230944
            2      34.919988        0.190424       35.156028        0.189489
            3      34.929927        0.165316       35.242255        0.164400
            4      34.957007        0.147852       35.247704        0.147191
            5      35.003660        0.134952       35.214658        0.134458
            6      35.070199        0.124956       35.178422        0.124664
            7      35.019853        0.116970       35.096870        0.116783
            8      35.035123        0.110147       35.225907        0.109742
            9      35.113417        0.104280       35.113417        0.104280

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
         'F. Error': forward_error_list,
         'Backward': backward_list,
         'B. Error': backward_error_list})
    convergence.attrs = df_list[0].attrs
    return convergence
