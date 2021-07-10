from os.path import join
from glob import glob
import pandas as pd
import numpy as np
import logging

from ..parsing import gmx
from ..preprocessing.subsampling import statistical_inefficiency
from ..estimators import MBAR, BAR, TI
from ..visualisation import (plot_mbar_overlap_matrix, plot_ti_dhdl,
                             plot_dF_state, plot_convergence)
from ..postprocessors.units import get_unit_converter
from .. import concat


class ABFE():
    '''Alchemical Analysis style automatic workflow.

    Parameters
    ----------
    units : str
        The unit used for printing and plotting results. {'kcal/mol', 'kJ/mol',
        'kT'}
    software : str
        The software used for generating input. {'Gromacs', }
    dir : str
        Directory in which data files are stored. Default: './'.
    prefix : str
        Prefix for datafile sets. Default: 'dhdl'.
    suffix : str
        Suffix for datafile sets. Default: 'xvg'.
    T : float
        Temperature in K. Default: 298.
    skiptime : float
        Discard data prior to this specified time as 'equilibration' data. Units
        picoseconds. Default: 0.
    uncorr : str
        The observable to be used for the autocorrelation analysis; 'dhdl'
        (obtained as a sum over those energy components that are changing).
        Default: `dhdl`
    threshold : int
        Proceed with correlated samples if the number of uncorrelated samples is
        found to be less than this number. If 0 is given, the time series
        analysis will not be performed at all. Default: 50.
    methods : str
        A list of the methods to esitimate the free energy with. Default: None.
    out : str
        Directory in which the output files produced by this script will be
        stored. Default: './'.
    resultfilename : str
        custom defined result filename. Default: None. (not writing the result)
    overlap : str
        The filename for the plot of overlap matrix. Default: None. (not
        plotting).
    breakdown : bool
        Plot the free energy differences evaluated for each pair of adjacent
        states for all methods, including the dH/dlambda curve for TI. Default:
        None. (not plotting).
    forwrev : int
        Plot the free energy change as a function of time in both directions,
        with the specified number of points in the time plot. The number of time
        points (an integer) must be provided. Default: None. (not doing
        convergence analysis).
    log : str
        The filename of the log file. Default: 'result.log'

    Attributes
    ----------
    logger : Logger
        The logging object.
    file_list : list
        The list of filenames sorted by the lambda state.
    u_nk_list : list
        The list of u_nk read from the files.
    dHdl_list : list
        The list of dHdl read from the files.
    '''
    def __init__(self, units='kcal/mol', software='Gromacs', dir='./',
                 prefix='dhdl', suffix='xvg', T=298, skiptime=0, uncorr=None,
                 threshold=50, methods=None, out='./', resultfilename=None,
                 overlap=None, breakdown=None, forwrev=None,
                 log='result.log'):

        logging.basicConfig(filename=log, level=logging.INFO)
        self.logger = logging.getLogger('alchemlyb.workflows.ABFE')
        self.logger.info('Initialise Alchemlyb ABFE Workflow')

        self.logger.info('Set temperature to {} K.'.format(T))
        self.T = T
        self.out = out

        self.update_units(units)

        self.logger.info('Finding files with prefix: {}, suffix: {} under '
                         'directory {} produced by {}'.format(prefix, suffix,
                                                              dir, software))
        file_list = glob(join(dir, prefix + '*' + suffix))

        self.logger.info('Found {} xvg files.'.format(len(file_list)))
        self.logger.info('Unsorted file list: \n{}'.format('\n'.join(
            file_list)))

        if software.lower() == 'gromacs':
            self.logger.info('Using {} parser to read the data.'.format(
                software))
            extract_u_nk = gmx.extract_u_nk
            extract_dHdl = gmx.extract_dHdl
        else: # pragma: no cover
            raise NameError('{} parser not found.'.format(software))

        u_nk_list = []
        dHdl_list = []
        for xvg in file_list:
            try:
                u_nk = extract_u_nk(xvg, T=T)
                self.logger.info(
                    'Reading {} lines of u_nk from {}'.format(len(u_nk), xvg))
                u_nk_list.append(u_nk)
            except: # pragma: no cover
                self.logger.warning(
                    'Error reading read u_nk from {}.'.format(xvg))

            try:
                dhdl = extract_dHdl(xvg, T=T)
                self.logger.info(
                    'Reading {} lines of dhdl from {}'.format(len(dhdl), xvg))
                dHdl_list.append(dhdl)
            except: # pragma: no cover
                self.logger.warning(
                    'Error reading read dhdl from {}.'.format(xvg))

        # Sort the files according to the state
        if len(u_nk_list) > 0:
            self.logger.info('Sort files according to the u_nk.')
            column_names = u_nk_list[0].columns.values.tolist()
            index_list = sorted(range(len(file_list)),
                key=lambda x:column_names.index(
                    u_nk_list[x].reset_index('time').index.values[0]))
        else:
            self.logger.info('Sort files according to the dHdl.')
            column_names = sorted([dHdl.reset_index('time').index.values[0]
                                   for dHdl in dHdl_list])
            index_list = sorted(range(len(file_list)),
                key=lambda x:column_names.index(
                    dHdl_list[x].reset_index('time').index.values[0]))

        self.file_list = [file_list[i] for i in index_list]
        self.logger.info('Sorted file list: \n{}'.format('\n'.join(
            self.file_list)))
        self.u_nk_list = [u_nk_list[i] for i in index_list]
        self.dHdl_list = [dHdl_list[i] for i in index_list]

        if uncorr is not None:
            self.preprocess(skiptime=skiptime, uncorr=uncorr,
                               threshold=threshold)
        if methods is not None:
            self.estimate(methods)

        if resultfilename is not None:
            self.write(resultfilename=resultfilename)

        if overlap is not None:
            self.plot_overlap_matrix(overlap)

        if breakdown:
            self.plot_ti_dhdl()
            self.plot_dF_state()
            self.plot_dF_state(dF_state='dF_state_long.pdf',
                               orientation='landscape')

        if forwrev is not None:
            self.check_convergence(forwrev, estimator='mbar', dF_t='dF_t.pdf')

    def update_units(self, units):
        '''Update the plot and text output to the selected unit.

        Parameters
        ----------
        units : str
            The unit used for printing and plotting results. {'kcal/mol',
            'kJ/mol', 'kT'}

        Attributes
        ----------
        scaling_factor : float
            The scaling factor to change the unit from kT to the selected unit.

        Note
        ----
        The internal representations are all in kT. This function only changes
        the unit when outputting text file or plotting the results.
        '''
        if units is not None:
            self.logger.info('Set unit to {}.'.format(units))
            self.units = units
        else: # pragma: no cover
            pass

    def preprocess(self, skiptime=0, uncorr='dhdl', threshold=50):
        '''Preprocess the data by removing the equilibration time and
        decorrelate the date.

        Parameters
        ----------
        skiptime : float
            Discard data prior to this specified time as 'equilibration' data.
            Units picoseconds. Default: 0.
        uncorr : str
            The observable to be used for the autocorrelation analysis; 'dhdl'
            (obtained as a sum over those energy components that are changing).
            Default: `dhdl`
        threshold : int
            Proceed with correlated samples if the number of uncorrelated
            samples is found to be less than this number. If 0 is given, the
            time series analysis will not be performed at all. Default: 50.

        Attributes
        ----------
        u_nk_sample_list : list
            The list of u_nk after decorrelation.
        dHdl_sample_list : list
            The list of dHdl after decorrelation.
        '''
        self.logger.info('Start preprocessing with skiptime of {} '
                         'uncorrelation method of {} and '
                         'threshold of {}'.format(skiptime, uncorr, threshold))
        if len(self.u_nk_list) > 0:
            self.logger.info(
                'Processing the u_nk data set with skiptime of {}.'.format(
                    skiptime))

            self.u_nk_sample_list = []
            for index, u_nk in enumerate(self.u_nk_list):
                # Find the starting frame

                u_nk = u_nk[u_nk.index.get_level_values('time') >= skiptime]
                if uncorr == 'dhdl':
                    # Find the current column index
                    # Select the first row and remove the first column (Time)
                    key = u_nk.index.values[0][1:]
                    if len(key) > 1:
                        # Multiple keys
                        col = u_nk[key]
                    else:
                        # Single key
                        col = u_nk[key[0]]
                    subsample = statistical_inefficiency(u_nk, col, sort=True,
                                                         drop_duplicates=True)
                else: # pragma: no cover
                    # The dhdl_all and dE will be implemented here when #48 is
                    # merged
                    raise NameError(
                        'Decorrelation method {} not found.'.format(uncorr))

                if len(subsample) < threshold:
                    self.logger.warning('Number of u_nk {} for state {} is '
                                        'less than the threshold {}.'.format(
                        len(subsample), index, threshold))
                    self.logger.info('Take all the u_nk for state {}.'.format(index))
                    self.u_nk_sample_list.append(u_nk)
                else:
                    self.logger.info('Take {} uncorrelated u_nk for state '
                                     '{}.'.format(len(subsample), index))
                    self.u_nk_sample_list.append(subsample)
        else: # pragma: no cover
            self.logger.info('No u_nk data being subsampled')

        if len(self.dHdl_list) > 0:
            self.dHdl_sample_list = []
            for index, dHdl in enumerate(self.dHdl_list):
                dHdl = dHdl[dHdl.index.get_level_values('time') >= skiptime]
                subsample = statistical_inefficiency(dHdl, dHdl.sum(axis=1))
                if len(subsample) < threshold:
                    self.logger.warning('Number of dHdl {} for state {} is '
                                        'less than the threshold {}.'.format(
                        len(subsample), index, threshold))
                    self.logger.info('Take all the dHdl for state {}.'.format(index))
                    self.dHdl_sample_list.append(dHdl)
                else:
                    self.logger.info('Take {} uncorrelated dHdl for state '
                                     '{}.'.format(len(subsample), index))
                    self.dHdl_sample_list.append(subsample)
        else: # pragma: no cover
            self.logger.info('No dHdl data being subsampled')

    def estimate(self, methods=('mbar', 'bar', 'ti')):
        '''Estimate the free energy using the selected estimator.

        Parameters
        ----------
        methods : str
            A list of the methods to esitimate the free energy with. Default:
            ['TI', 'BAR', 'MBAR'].

        Attributes
        ----------
        estimator : dict
            The dictionary of estimators. The key for MBAR is 'mbar', for BAR is
            'bar' and for TI is 'ti'.
        '''
        # Make estimators into a tuple
        if isinstance(methods, str):
            methods = (methods, )

        self.logger.info(
            'Start running estimator: {}.'.format(','.join(methods)))
        self.estimator = {}
        # Use unprocessed data if preprocess is not performed.
        if 'ti' in methods:
            try:
                dHdl = concat(self.dHdl_sample_list)
            except (AttributeError, ValueError):
                dHdl = concat(self.dHdl_list)
                self.logger.warning('dHdl has not been preprocessed.')
            self.logger.info(
                'A total {} lines of dHdl is used.'.format(len(dHdl)))

        if 'bar' in methods or 'mbar' in methods:
            try:
                u_nk = concat(self.u_nk_sample_list)
            except (AttributeError, ValueError):
                u_nk = concat(self.u_nk_list)
                self.logger.warning('u_nk has not been preprocessed.')
            self.logger.info(
                'A total {} lines of u_nk is used.'.format(len(u_nk)))

        for estimator in methods:
            if estimator.lower() == 'mbar' and len(u_nk) > 0:
                self.logger.info('Run MBAR estimator.')
                self.estimator['mbar'] = MBAR().fit(u_nk)
            elif estimator.lower() == 'bar' and len(u_nk) > 0:
                self.logger.info('Run BAR estimator.')
                self.estimator['bar'] = BAR().fit(u_nk)
            elif estimator.lower() == 'ti' and len(dHdl) > 0:
                self.logger.info('Run TI estimator.')
                self.estimator['ti'] = TI().fit(dHdl)
            elif estimator.lower() == 'mbar' or estimator.lower() == 'bar': # pragma: no cover
                self.logger.warning('MBAR or BAR estimator require u_nk')
            else: # pragma: no cover
                self.logger.warning(
                    '{} is not a valid estimator.'.format(estimator))

    def write(self, resultfilename='result.out'):
        '''Write the result into a text file.

        Parameters
        ----------
        resultfilename : str
            A list of the methods to esitimate the free energy with. Default:
            ['TI', 'BAR', 'MBAR'].
        '''

        # Write estimate
        self.logger.info('Write the estimate as txt file to {} under {} '
                         'with unit {}.'.format(
            resultfilename, self.out, self.units))
        # Make the header name
        self.logger.info('Write the header names.')
        result_out = [['------------', ],
                      ['   States   ', ],
                      ['------------', ],]
        eitimator_names = list(self.estimator.keys())
        num_states = len(self.estimator[eitimator_names[0]].states_)
        for i in range(num_states - 1):
            result_out.append([str(i).rjust(4) + ' -- ' + str(i+1).ljust(4), ])
        result_out.append(['------------', ])
        try:
            u_nk = self.u_nk_list[0]
            stages = u_nk.reset_index('time').index.names
            self.logger.info('use the stage name from u_nk')
        except:
            try:
                dHdl = self.dHdl_list[0]
                stages = dHdl.reset_index('time').index.names
                self.logger.info('use the stage name from dHdl')
            except: # pragma: no cover
                stages = []
                self.logger.warning('No stage name found in dHdl or u_nk')
        for stage in stages:
            result_out.append([stage.split('-')[0][:9].rjust(9)+':  ', ])
        result_out.append(['TOTAL'.rjust(9) + ':  ', ])

        converter = get_unit_converter(self.units)
        for estimator_name, estimator in self.estimator.items():
            self.logger.info('write the result from estimator {}'.format(
                estimator_name))

            # Do the unit conversion
            delta_f_ = converter(estimator.delta_f_)
            d_delta_f_ = converter(estimator.d_delta_f_)
            # Write the estimator header
            result_out[0].append('---------------------')
            result_out[1].append('{} ({}) '.format(
                estimator_name.upper(), self.units).rjust(21))
            result_out[2].append('---------------------')
            for index in range(1, num_states):
                result_out[2+index].append('{:.3f}  +-  {:.3f}'.format(
                    delta_f_.iloc[index-1, index],
                    d_delta_f_.iloc[index-1, index]
                ).rjust(21))

            result_out[2+num_states].append('---------------------')

            self.logger.info('write the staged result from estimator {}'.format(
                estimator_name))
            for index, stage in enumerate(stages):
                if len(stages) == 1:
                    start = 0
                    end = len(estimator.states_) - 1
                else:
                    # Get the start and the end of the state
                    lambda_min = min(
                        [state[index] for state in estimator.states_])
                    lambda_max = max(
                        [state[index] for state in estimator.states_])
                    if lambda_min == lambda_max:
                        # Deal with the case where a certain lambda is used but
                        # not perturbed
                        start = 0
                        end = 0
                    else:
                        states = [state[index] for state in estimator.states_]
                        start = list(reversed(states)).index(lambda_min)
                        start = num_states - start - 1
                        end = states.index(lambda_max)
                self.logger.info(
                    'Stage {} is from state {} to state {}.'.format(
                        stage, start, end))
                result = delta_f_.iloc[start, end]
                if estimator_name != 'bar':
                    error = d_delta_f_.iloc[start, end]
                else:
                    error = np.sqrt(sum(
                        [d_delta_f_.iloc[start, start+1]**2
                         for i in range(start, end + 1)]))
                result_out[3 + num_states + index].append(
                    '{:.3f}  +-  {:.3f}'.format(result, error,).rjust(21))

            # Total result
            result = delta_f_.iloc[0, -1]
            if estimator_name != 'bar':
                error = d_delta_f_.iloc[0, -1]
            else:
                error = np.sqrt(sum(
                    [d_delta_f_.iloc[i, i + 1] ** 2
                     for i in range(num_states - 1)]))
            result_out[3 + num_states + len(stages)].append(
                '{:.3f}  +-  {:.3f}'.format(result, error, ).rjust(21))
        self.logger.info('Write results:\n'+
                         '\n'.join([' '.join(line) for line in result_out]))
        with open(join(self.out, resultfilename), 'w') as f:
            f.write('\n'.join([' '.join(line) for line in result_out]))

    def plot_overlap_matrix(self, overlap='O_MBAR.pdf', ax=None):
        '''Plot the overlap matrix for MBAR estimator using
        :func:`~alchemlyb.visualisation.plot_mbar_overlap_matrix`.

        Parameters
        ----------
        overlap : str
            The filename for the plot of overlap matrix. Default: 'O_MBAR.pdf'
        ax : matplotlib.axes.Axes
            Matplotlib axes object where the plot will be drawn on. If ax=None,
            a new axes will be generated.

        Returns
        -------
        matplotlib.axes.Axes
            An axes with the overlap matrix drawn.
        '''
        self.logger.info('Plot overlap matrix.')
        if 'mbar' in self.estimator:
            ax = plot_mbar_overlap_matrix(self.estimator['mbar'].overlap_matrix,
                                          ax=ax)
            ax.figure.savefig(join(self.out, overlap))
            self.logger.info('Plot overlap matrix to {} under {}.'
                             ''.format(self.out, overlap))
            return ax
        else: # pragma: no cover
            self.logger.warning('MBAR estimator not found. '
                                'Overlap matrix not plotted.')

    def plot_ti_dhdl(self, dhdl_TI='dhdl_TI.pdf', labels=None, colors=None,
                     ax=None):
        '''Plot the dHdl for TI estimator using
        :func:`~alchemlyb.visualisation.plot_ti_dhdl`.

        Parameters
        ----------
        dhdl_TI : str
            The filename for the plot of TI dHdl. Default: 'dhdl_TI.pdf'
        labels : List
            list of labels for labelling all the alchemical transformations.
        colors : List
            list of colors for plotting all the alchemical transformations.
            Default: ['r', 'g', '#7F38EC', '#9F000F', 'b', 'y']
        ax : matplotlib.axes.Axes
            Matplotlib axes object where the plot will be drawn on. If ax=None,
            a new axes will be generated.

        Returns
        -------
        matplotlib.axes.Axes
            An axes with the TI dhdl drawn.
        '''
        self.logger.info('Plot TI dHdl.')
        if 'ti' in self.estimator:
            ax = plot_ti_dhdl(self.estimator['ti'], units=self.units,
                              labels=labels, colors=colors, ax=ax)
            ax.figure.savefig(join(self.out, dhdl_TI))
            self.logger.info('Plot TI dHdl to {} under {}.'
                             ''.format(dhdl_TI, self.out))

    def plot_dF_state(self, dF_state='dF_state.pdf', labels=None, colors=None,
                      orientation='portrait', nb=10):
        '''Plot the dF states using
        :func:`~alchemlyb.visualisation.plot_dF_state`.

        Parameters
        ----------
        dF_state : str
            The filename for the plot of dF states. Default: 'dF_state.pdf'
        labels : List
            list of labels for labelling different estimators.
        colors : List
            list of colors for plotting different estimators.
        orientation : string
            The orientation of the figure. Can be `portrait` or `landscape`
        nb : int
            Maximum number of dF states in one row in the `portrait` mode

        Returns
        -------
        matplotlib.figure.Figure
            An Figure with the dF states drawn.
        '''
        self.logger.info('Plot dF states.')
        fig = plot_dF_state(self.estimator.values(), labels=labels, colors=colors,
                            units=self.units,
                            orientation=orientation, nb=nb)
        fig.savefig(join(self.out, dF_state))
        self.logger.info('Plot dF state to {} under {}.'
                         ''.format(dF_state, self.out))

    def check_convergence(self, forwrev, estimator='mbar', dF_t='dF_t.pdf',
                     ax=None):
        '''Compute the forward and backward convergence and plotted with
        :func:`~alchemlyb.visualisation.plot_convergence`.

        Parameters
        ----------
        forwrev : int
            Plot the free energy change as a function of time in both
            directions, with the specified number of points in the time plot.
            The number of time points (an integer) must be provided.
        estimator : str
            The estimator used for convergence analysis. Default: 'mbar'
        dF_t : str
            The filename for the plot of convergence. Default: 'dF_t.pdf'
        ax : matplotlib.axes.Axes
            Matplotlib axes object where the plot will be drawn on. If ax=None,
            a new axes will be generated.

        Attributes
        ----------
        convergence : DataFrame
            The DataFrame with convergence data. ::

                   Forward (kT)  F. Error (kT)  Backward (kT)  B. Error (kT)
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

        Returns
        -------
        matplotlib.axes.Axes
            An axes with the convergence drawn.
        '''
        self.logger.info('Start convergence analysis.')
        self.logger.info('Check data availability.')

        try:
            dHdl_list = self.dHdl_sample_list
            self.logger.info('Subsampled dHdl is available.')
        except AttributeError:
            try:
                dHdl_list = self.dHdl_list
                self.logger.info('Subsampled dHdl not available, '
                                 'use original data instead.')
            except AttributeError: # pragma: no cover
                self.logger.warning('dHdl is not available.')

        try:
            u_nk_list = self.u_nk_sample_list
            self.logger.info('Subsampled u_nk is available.')
        except AttributeError:
            try:
                u_nk_list = self.u_nk_list
                self.logger.info('Subsampled u_nk not available, '
                                 'use original data instead.')
            except AttributeError: # pragma: no cover
                self.logger.warning('u_nk is not available.')

        if estimator.lower() == 'mbar':
            self.logger.info('Use MBAR estimator for convergence analysis.')
            estimator_fit = MBAR().fit
        elif estimator.lower() == 'bar':
            self.logger.info('Use BAR estimator for convergence analysis.')
            estimator_fit = BAR().fit
        elif estimator.lower() == 'ti':
            self.logger.info('Use TI estimator for convergence analysis.')
            estimator_fit = TI().fit
        else: # pragma: no cover
            self.logger.warning(
                '{} is not a valid estimator.'.format(estimator))

        converter = get_unit_converter(self.units)

        self.logger.info('Begin forward analysis')
        forward_list = []
        forward_error_list = []
        for i in range(1, forwrev + 1):
            self.logger.info('Forward analysis: {:.2f}%'.format(i / forwrev))
            sample = []
            if estimator.lower() in ['mbar', 'bar']:
                for data in u_nk_list:
                    sample.append(data[:len(data) // forwrev * i])
            elif estimator.lower() == 'ti':
                for data in dHdl_list:
                    sample.append(data[:len(data) // forwrev * i])
            else:  # pragma: no cover
                raise NameError(
                    '{} is not a valid estimator.'.format(estimator))
            sample = concat(sample)
            result = estimator_fit(sample)
            forward_list.append(converter(result.delta_f_).iloc[0, -1])
            if estimator.lower() == 'bar':
                error = np.sqrt(sum(
                    [converter(result.d_delta_f_).iloc[i, i + 1] ** 2
                     for i in range(len(result.d_delta_f_) - 1)]))
                forward_error_list.append(error)
            else:
                forward_error_list.append(converter(result.d_delta_f_).iloc[
                                                        0, -1])
            self.logger.info('{:.2f} +/- {:.2f} kT'.format(forward_list[-1],
                                                        forward_error_list[-1]))

        self.logger.info('Begin backward analysis')
        backward_list = []
        backward_error_list = []
        for i in range(1, forwrev + 1):
            self.logger.info('Backward analysis: {:.2f}%'.format(i / forwrev))
            sample = []
            if estimator.lower() in ['mbar', 'bar']:
                for data in u_nk_list:
                    sample.append(data[-len(data) // forwrev * i:])
            elif estimator.lower() == 'ti':
                for data in dHdl_list:
                    sample.append(data[-len(data) // forwrev * i:])
            else:  # pragma: no cover
                raise NameError(
                    '{} is not a valid estimator.'.format(estimator))
            sample = concat(sample)
            result = estimator_fit(sample)
            backward_list.append(converter(result.delta_f_).iloc[0, -1])
            if estimator.lower() == 'bar':
                error = np.sqrt(sum(
                    [converter(result.d_delta_f_).iloc[i, i + 1] ** 2
                     for i in range(len(result.d_delta_f_) - 1)]))
                backward_error_list.append(error)
            else:
                backward_error_list.append(converter(
                    result.d_delta_f_).iloc[0, -1])
            self.logger.info('{:.2f} +/- {:.2f} kT'.format(backward_list[-1],
                                                        backward_error_list[-1]))

        convergence = pd.DataFrame(
            {'Forward ({})'.format(self.units): forward_list,
             'F. Error ({})'.format(self.units): forward_error_list,
             'Backward ({})'.format(self.units): backward_list,
             'B. Error ({})'.format(self.units): backward_error_list})

        self.convergence = convergence
        self.logger.info('Plot convergence analysis to {} under {}.'
                         ''.format(dF_t, self.out))

        ax = plot_convergence(np.array(forward_list),
                              np.array(forward_error_list),
                              np.array(backward_list),
                              np.array(backward_error_list),
                              units=self.units, ax=ax)
        ax.figure.savefig(join(self.out, dF_t))
        return ax
