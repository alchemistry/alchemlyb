"""Functions for assessing convergence of free energy estimates and raw data."""

import logging
from warnings import warn

import pandas as pd
import numpy as np

from ..estimators import BAR, TI, FEP_ESTIMATORS, TI_ESTIMATORS
from ..estimators import AutoMBAR as MBAR
from .. import concat
from ..postprocessors.units import to_kT


def forward_backward_convergence(df_list, estimator='MBAR', num=10, **kwargs):
    '''Forward and backward convergence of the free energy estimate.

    Generate the free energy estimate as a function of time in both directions,
    with the specified number of equally spaced points in the time
    [Klimovich2015]_. For example, setting `num` to 10 would give the forward
    convergence which is the free energy estimate from the first 10%, 20%, 30%,
    ... of the data. The Backward would give the estimate from the last 10%,
    20%, 30%, ... of the data.

    Parameters
    ----------
    df_list : list
        List of DataFrame of either dHdl or u_nk.
    estimator : {'MBAR', 'BAR', 'TI'}
        Name of the estimators.
        See the important note below on the use of "MBAR".

        .. deprecated:: 1.0.0
           Lower case input is also accepted until release 2.0.0.
    num : int
        The number of time points.
    kwargs : dict
        Keyword arguments to be passed to the estimator.

    Returns
    -------
    :class:`pandas.DataFrame`
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
       The default for using ``estimator='MBAR'`` was changed from
       :class:`~alchemlyb.estimators.MBAR` to :class:`~alchemlyb.estimators.AutoMBAR`.

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

def _cummean(vals, out_length):
    '''The cumulative mean of an array.

    This function computes the cumulative mean and shapes the result to the
    desired length.

    Parameters
    ----------
    vals : numpy.array
        The one-dimensional input array.
    out_length : int
        The length of the output array.

    Returns
    -------
    numpy.array
        The one-dimensional input array with length of total.

    Note
    ----
    If the length of the input series is smaller than the ``out_length``, the
    length of the output array is the same as the input series.


    .. versionadded:: 1.0.0

    '''
    in_length = len(vals)
    if in_length < out_length:
        out_length = in_length
    block = in_length // out_length
    reshape = vals[: block*out_length].reshape(block, out_length)
    mean = np.mean(reshape, axis=0)
    result = np.cumsum(mean) / np.arange(1, out_length+1)
    return result

def fwdrev_cumavg_Rc(series, precision=0.01, tol=2):
    '''Generate the convergence criteria :math:`R_c` for a single simulation.

    The input will be :class:`pandas.Series` generated by
    :func:`~alchemlyb.preprocessing.subsampling.decorrelate_u_nk` or
    :func:`~alchemlyb.preprocessing.subsampling.decorrelate_dhdl`.

    The output will be the float :math:`R_c` [Fan2020]_ [Fan2021]_ and a
    :class:`pandas.DataFrame` with the forward and backward cumulative average
    at `precision` fractional increments, as described below.

    :math:`R_c = 0` indicates that the system is well equilibrated right from
    the beginning while :math:`R_c = 1` signifies that the whole trajectory is
    not equilibrated.

    Parameters
    ----------
    series : pandas.Series
        The input energy array.
    precision : float
        The precision of the output :math:`R_c`. To speed the calculation up, the data
        has been block-averaged before doing the calculation, the size of the
        block is controlled by the desired precision.
    tol : float
        Tolerance (or convergence threshold :math:`\epsilon` in [Fan2021]_)
        in :math:`kT`.

    Returns
    -------
    float
        Convergence time fraction :math:`R_c` [Fan2021]_
    :class:`pandas.DataFrame`
        The DataFrame with moving average. ::

                Forward  Backward  data_fraction
            0  3.016442  3.065176            0.1
            1  3.078106  3.078567            0.2
            2  3.072561  3.047357            0.3
            3  3.048325  3.057527            0.4
            4  3.049769  3.037454            0.5
            5  3.034078  3.040484            0.6
            6  3.043274  3.032495            0.7
            7  3.035460  3.036670            0.8
            8  3.042032  3.046597            0.9
            9  3.044149  3.044385            1.0

    Notes
    -----
    This function computes :math:`R_c` from `equation 16`_ from [Fan2021]_.
    The code is modified based on Shujie Fan's (@VOD555) work. Zhiyi Wu
    (@xiki-tempula) improved the performance of the original algorithm.

    Please cite [Fan2021]_ when using this function.

    See also
    --------
    A_c


    .. versionadded:: 1.0.0

    .. _`equation 16`:
       https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8397498/#FD16

    '''
    series = to_kT(series)
    array = series.to_numpy()
    out_length = int(1 / precision)
    g_forward = _cummean(array, out_length)
    g_backward = _cummean(array[::-1], out_length)
    length = len(g_forward)

    convergence = pd.DataFrame(
        {'Forward': g_forward,
         'Backward': g_backward,
         'data_fraction': [i / length for i in range(1, length + 1)]})
    convergence.attrs = series.attrs

    # Final value
    g = g_forward[-1]
    conv = np.logical_and(np.abs(g_forward - g) < tol, np.abs(g_backward - g) < tol)
    for i in range(out_length):
        if all(conv[i:]):
            return i / length, convergence
    else:
        # This branch exists as we are truncating the dataset to speed the
        # calculation up. For example if the dataset has 21 points and the
        # precision is 0.05. The last point of g_forward is computed using
        # data[0:20] while the last point of g_backward is computed using
        # data[1:21]. Thus, the last point of g_forward and g_backward are not
        # the same as this branch will be triggered.
        return 1.0, convergence

def A_c(series_list, precision=0.01, tol=2):
    '''Generate the ensemble convergence criteria :math:`A_c` for a set of simulations.

    The input is a :class:`list` of :class:`pandas.Series` generated by
    :func:`~alchemlyb.preprocessing.subsampling.decorrelate_u_nk` or
    :func:`~alchemlyb.preprocessing.subsampling.decorrelate_dhdl`.

    The output will the float :math:`A_c` [Fan2020]_ [Fan2021]_. :math:`A_c` is
    a number between 0 and 1 that can be interpreted as the ratio of the total
    equilibrated simulation time to the whole simulation time for a full set of
    simulations. :math:`A_c = 1` means that all simulation time frames in all
    windows can be considered equilibrated, while :math:`A_c = 0` indicates
    that nothing is equilibrated.

    Parameters
    ----------
    series_list : list
        A list of :class:`pandas.Series` energy array.
    precision : float
        The precision of the output :math:`A_c`. To speed the calculation up, the data
        has been block-averaged before doing the calculation, the size of the
        block is controlled by the desired precision.
    tol : float
        Tolerance (or convergence threshold :math:`\epsilon` in [Fan2021]_)
        in :math:`kT`.

    Returns
    -------
    float
        The area :math:`A_c` under curve for convergence time fraction.

    Notes
    -----
    This function computes :math:`A_c` from `equation 18`_ from [Fan2021]_.

    Please cite [Fan2021]_ when using this function.

    See also
    --------
    fwdrev_cumavg_Rc


    .. versionadded:: 1.0.0

    .. _`equation 18`:
       https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8397498/#FD18

    '''
    logger = logging.getLogger('alchemlyb.convergence.A_c')
    n_R_c = len(series_list)
    R_c_list = [fwdrev_cumavg_Rc(series, precision, tol)[0] for series in series_list]
    logger.info(f'R_c list: {R_c_list}')
    # Integrate the R_c_list <= R_c over the range of 0 to 1
    array_01 = np.hstack((R_c_list, [0, 1]))
    sorted_array = np.sort(np.unique(array_01))
    result = 0
    for i, element in enumerate(sorted_array[::-1]):
        if i == 0:
            continue
        else:
            d_R_c = sorted_array[-i] - sorted_array[-i-1]
            result += d_R_c * sum(R_c_list <= element) / n_R_c
    return result
