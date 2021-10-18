import os
from os.path import join
from glob import glob
import pandas as pd
import numpy as np
import logging

from ..parsing import gmx
from ..preprocessing.subsampling import decorrelate_dhdl, decorrelate_u_nk
from ..estimators import MBAR, BAR, TI
from ..visualisation import (plot_mbar_overlap_matrix, plot_ti_dhdl,
                             plot_dF_state, plot_convergence)
from ..postprocessors.units import get_unit_converter
from ..convergence import forward_backward_convergence
from .. import concat
from .. import __version__


class ABFE():
    '''Alchemical Analysis style automatic workflow.

    Parameters
    ----------
    units : str
        The unit used for printing and plotting results. {'kcal/mol', 'kJ/mol',
        'kT'}. Default: 'kT'.
    software : str
        The software used for generating input. {'Gromacs', }
    dir : str
        Directory in which data files are stored. Default: os.path.curdir.
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
        stored. Default: os.path.curdir.
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
        The filename of the log file. The workflow logs under
        alchemlyb.workflows.ABFE. Default:
        'result.log'

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
    def __init__(self, units='kT', software='Gromacs', dir=os.path.curdir,
                 prefix='dhdl', suffix='xvg', T=298, skiptime=0, uncorr=None,
                 threshold=50, methods=None, out=os.path.curdir,
                 overlap=None, breakdown=None, forwrev=None,
                 log='result.log'):

        logging.basicConfig(filename=log, level=logging.INFO)
        self.logger = logging.getLogger('alchemlyb.workflows.ABFE')
        self.logger.info('Initialise Alchemlyb ABFE Workflow')
        self.logger.info('Alchemlyb Version: {}'.format(__version__))

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

        if overlap is not None:
            ax = self.plot_overlap_matrix(overlap)
            plt.close(ax.figure)

        if breakdown:
            ax = self.plot_ti_dhdl()
            plt.close(ax.figure)
            fig = self.plot_dF_state()
            plt.close(fig)
            fig = self.plot_dF_state(dF_state='dF_state_long.pdf',
                                    orientation='landscape')
            plt.close(fig)

        if forwrev is not None:
            ax = self.check_convergence(forwrev, estimator='mbar', dF_t='dF_t.pdf')
            plt.close(ax.figure)


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
                subsample = decorrelate_u_nk(u_nk, uncorr)

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
                subsample = decorrelate_dhdl(dHdl)
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

    def generate_result(self):
        '''Summarise the result into a dataframe.

        Returns
        -------
        DataFrame
            The DataFrame with convergence data. ::

                                      MBAR  MBAR_Error        BAR  BAR_Error         TI  TI_Error
                States 0 -- 1     0.065967    0.001293   0.066544   0.001661   0.066663  0.001675
                       1 -- 2     0.089774    0.001398   0.089303   0.002101   0.089566  0.002144
                       2 -- 3     0.132036    0.001638   0.132687   0.002990   0.133292  0.003055
                       3 -- 4     0.116494    0.001213   0.116348   0.002691   0.116845  0.002750
                       4 -- 5     0.105251    0.000980   0.106344   0.002337   0.106603  0.002362
                       5 -- 6     0.349320    0.002781   0.343399   0.006839   0.350568  0.007393
                       6 -- 7     0.402346    0.002767   0.391368   0.006641   0.395754  0.006961
                       7 -- 8     0.322284    0.002058   0.319395   0.005333   0.321542  0.005434
                       8 -- 9     0.434999    0.002683   0.425680   0.006823   0.430251  0.007155
                       9 -- 10    0.355672    0.002219   0.350564   0.005472   0.352745  0.005591
                       10 -- 11   3.574227    0.008744   3.513595   0.018711   3.514790  0.018078
                       11 -- 12   2.896685    0.009905   2.821760   0.017844   2.823210  0.018088
                       12 -- 13   2.223769    0.011229   2.188885   0.018438   2.189784  0.018478
                       13 -- 14   1.520978    0.012526   1.493598   0.019155   1.490070  0.019288
                       14 -- 15   0.911279    0.009527   0.894878   0.015023   0.896010  0.015140
                       15 -- 16   0.892365    0.010558   0.886706   0.015260   0.884698  0.015392
                       16 -- 17   1.737971    0.025315   1.720643   0.031416   1.741028  0.030624
                       17 -- 18   1.790706    0.025560   1.788112   0.029435   1.801695  0.029244
                       18 -- 19   1.998635    0.023340   2.007404   0.027447   2.019213  0.027096
                       19 -- 20   2.263475    0.020286   2.265322   0.025023   2.282040  0.024566
                       20 -- 21   2.565680    0.016695   2.561324   0.023611   2.552977  0.023753
                       21 -- 22   1.384094    0.007553   1.385837   0.011672   1.381999  0.011991
                       22 -- 23   1.428567    0.007504   1.422689   0.012524   1.416010  0.013012
                       23 -- 24   1.440581    0.008059   1.412517   0.013125   1.408267  0.013539
                       24 -- 25   1.411329    0.009022   1.419167   0.013356   1.411446  0.013795
                       25 -- 26   1.340320    0.010167   1.360679   0.015213   1.356953  0.015260
                       26 -- 27   1.243745    0.011239   1.245873   0.015711   1.248959  0.015762
                       27 -- 28   1.128429    0.012859   1.124554   0.016999   1.121892  0.016962
                       28 -- 29   1.010313    0.016442   1.005444   0.017692   1.019747  0.017257
                Stages coul      10.215658    0.033903  10.017838   0.041839  10.017854  0.048744
                       vdw       22.547489    0.098699  22.501150   0.060092  22.542936  0.106723
                       bonded     2.374144    0.014995   2.341631   0.005507   2.363828  0.021078
                       TOTAL     35.137291    0.103580  34.860619   0.087022  34.924618  0.119206

        '''

        # Write estimate
        self.logger.info('Summarise the estimate into a dataframe.')
        # Make the header name
        self.logger.info('Generate the row names.')
        eitimator_names = list(self.estimator.keys())
        num_states = len(self.estimator[eitimator_names[0]].states_)
        data_dict = {'name': [],
                     'state': []}
        for i in range(num_states - 1):
            data_dict['name'].append(str(i) + ' -- ' + str(i+1))
            data_dict['state'].append('States')

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
            data_dict['name'].append(stage.split('-')[0])
            data_dict['state'].append('Stages')
        data_dict['name'].append('TOTAL')
        data_dict['state'].append('Stages')

        col_names = []
        for estimator_name, estimator in self.estimator.items():
            self.logger.info('Read the results from estimator {}'.format(
                estimator_name))

            # Do the unit conversion
            delta_f_ = estimator.delta_f_
            d_delta_f_ = estimator.d_delta_f_
            # Write the estimator header

            col_names.append(estimator_name.upper())
            col_names.append(estimator_name.upper() + '_Error')
            data_dict[estimator_name.upper()] = []
            data_dict[estimator_name.upper() + '_Error'] = []
            for index in range(1, num_states):
                data_dict[estimator_name.upper()].append(
                    delta_f_.iloc[index-1, index])
                data_dict[estimator_name.upper() + '_Error'].append(
                    d_delta_f_.iloc[index - 1, index])

            self.logger.info('Generate the staged result from estimator {'
                             '}'.format(
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
                data_dict[estimator_name.upper()].append(result)
                data_dict[estimator_name.upper() + '_Error'].append(error)

            # Total result
            result = delta_f_.iloc[0, -1]
            if estimator_name != 'bar':
                error = d_delta_f_.iloc[0, -1]
            else:
                error = np.sqrt(sum(
                    [d_delta_f_.iloc[i, i + 1] ** 2
                     for i in range(num_states - 1)]))
            data_dict[estimator_name.upper()].append(result)
            data_dict[estimator_name.upper() + '_Error'].append(error)
        summary = pd.DataFrame.from_dict(data_dict)

        summary = summary.set_index(['state', 'name'])
        # Make sure that the columns are in the right order
        summary = summary[col_names]
        # Remove the name of the index column to make it prettier
        summary.index.names = [None, None]

        summary.attrs = estimator.delta_f_.attrs
        converter = get_unit_converter(self.units)
        summary = converter(summary)
        self.summary = summary
        self.logger.info('Write results:\n{}'.format(summary.to_string()))
        return summary

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
            return ax

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
        return fig

    def check_convergence(self, forwrev, estimator='mbar', dF_t='dF_t.pdf',
                     ax=None):
        '''Compute the forward and backward convergence using
        :func:`~alchemlyb.convergence.forward_backward_convergence`and
        plotted with
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

        Returns
        -------
        matplotlib.axes.Axes
            An axes with the convergence drawn.
        '''
        self.logger.info('Start convergence analysis.')
        self.logger.info('Check data availability.')

        if estimator.lower() in ['mbar', 'bar']:
            try:
                u_nk_list = self.u_nk_sample_list
                self.logger.info('Subsampled u_nk is available.')
            except AttributeError:
                try:
                    u_nk_list = self.u_nk_list
                    self.logger.info('Subsampled u_nk not available, '
                                     'use original data instead.')
                except AttributeError:  # pragma: no cover
                    self.logger.warning('u_nk is not available.')
            convergence = forward_backward_convergence(u_nk_list,
                                                       estimator=estimator,
                                                       num=forwrev)
        else:
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
            convergence = forward_backward_convergence(dHdl_list,
                                                       estimator=estimator,
                                                       num=forwrev)

        self.convergence = get_unit_converter(self.units)(convergence)

        self.logger.info('Plot convergence analysis to {} under {}.'
                         ''.format(dF_t, self.out))

        ax = plot_convergence(self.convergence,
                              units=self.units, ax=ax)
        ax.figure.savefig(join(self.out, dF_t))
        return ax
