import os
from os.path import join
from glob import glob
import pandas as pd
import numpy as np
import scipy
import logging

from ..parsing import gmx, amber, namd, gomc
from ..preprocessing.subsampling import statistical_inefficiency
from ..estimators import MBAR, BAR, TI
from ..visualisation import (plot_mbar_overlap_matrix, plot_ti_dhdl,
                             plot_dF_state, plot_convergence)


class ABFE():
    def __init__(self, units='kcal/mol', software='Gromacs', dir='./',
                 prefix='dhdl', suffix='xvg', T=298, skiptime=None, uncorr=None,
                 threshold=50, estimator=None, out='./', resultfilename=None,
                 overlap=None, breakdown=None, forwrev=None,
                 log='result.log'):
        logging.basicConfig(filename=log, level=logging.INFO)
        self.logger = logging.getLogger('alchemlyb.workflows.ABFE')
        self.logger.info('Initialise Alchemlyb ABFE Workflow')

        self.logger.info('Set temperature to {} K.'.format(T))
        self.T = T
        self.out = out

        self._update_units(units)

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
        elif software.lower() == 'amber':
            self.logger.info('Using {} parser to read the data.'.format(
                software))
            extract_u_nk = amber.extract_u_nk
            extract_dHdl = amber.extract_dHdl
        elif software.lower() == 'namd':
            self.logger.info('Using {} parser to read the data.'.format(
                software))
            extract_u_nk = namd.extract_u_nk
            self.logger.warning('No dHdl reader available for NAMD.')
        elif software.lower() == 'gomc':
            self.logger.info('Using {} parser to read the data.'.format(
                software))
            extract_u_nk = gomc.extract_u_nk
            extract_dHdl = gomc.extract_dHdl
        else:
            raise NameError('{} parser not found.'.format(software))

        u_nk_list = []
        dHdl_list = []
        for xvg in file_list:
            try:
                u_nk = extract_u_nk(xvg, T=T)
                self.logger.info(
                    'Reading {} lines of u_nk from {}'.format(len(u_nk), xvg))
                u_nk_list.append(u_nk)
            except:
                self.logger.warning(
                    'Error reading read u_nk from {}.'.format(xvg))

            try:
                dhdl = extract_dHdl(xvg, T=T)
                self.logger.info(
                    'Reading {} lines of dhdl from {}'.format(len(dhdl), xvg))
                dHdl_list.append(dhdl)
            except:
                self.logger.warning(
                    'Error reading read dhdl from {}.'.format(xvg))

        # # Sort the files according to the state
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

        if skiptime is not None and uncorr is not None:
            self.preprocess(skiptime=skiptime, uncorr=uncorr,
                               threshold=threshold)
        if estimator is not None:
            self.estimate(estimator)

        if resultfilename is not None:
            self.write(estimator, resultfilename=resultfilename, units=units)

        if overlap is not None:
            self.plot_overlap_matrix(overlap)

        if breakdown:
            self.plot_ti_dhdl()
            self.plot_dF_state()
            self.plot_dF_state(dF_state='dF_state_long.pdf',
                               orientation='landscape')


    def _update_units(self, units):
        if units is not None:
            self.logger.info('Set unit to {}.'.format(units))
            if units == 'kBT':
                self.scaling_factor = 1
            elif units == 'kJ/mol':
                self.scaling_factor = scipy.constants.k * self.T * scipy.constants.N_A / \
                              1000
            elif units == 'kcal/mol':
                kJ2kcal = 0.239006
                self.scaling_factor = scipy.constants.k * self.T * scipy.constants.N_A / \
                              1000 * kJ2kcal
            else:
                raise NameError('{} is not a valid unit.'.format(units))
            self.units = units

    def preprocess(self, skiptime=0, uncorr='dhdl', threshold=50):
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

                u_nk = u_nk[u_nk.index.get_level_values('time')>skiptime]
                if uncorr == 'dhdl':
                    # Find the current column index
                    # Select the first row and remove the first column (Time)
                    key = u_nk.index.values[0][1:]
                    col = u_nk[key]
                    subsample = statistical_inefficiency(u_nk, u_nk[key])
                elif uncorr == 'dhdl_all':
                    subsample = statistical_inefficiency(u_nk, u_nk.sum(axis=1))
                elif uncorr == 'dE':
                    # Using the same logic as alchemical-analysis
                    key = u_nk.index.values[0][1:]
                    index = u_nk.columns.values.tolist().index(key)
                    # for the state that is not the last state, take the state+1
                    if index + 1 < len(u_nk.columns):
                        subsample = statistical_inefficiency(
                            u_nk, u_nk.iloc[:, index + 1])
                    # for the state that is the last state, take the state-1
                    else:
                        subsample = statistical_inefficiency(
                            u_nk, u_nk.iloc[:, index - 1])
                else:
                    raise NameError(
                        'Decorrelation method {} not found.'.format(uncorr))

                if len(subsample) < threshold:
                    self.logger.warning('Number of u_nk {} for state {} is '
                                        'less than the threshold {}.'.format(
                        len(subsample), index, threshold))
                    self.logger.info('Take all the u_nk for state {}.'.format(index))
                    self.u_nk_sample_list.append(subsample)
                else:
                    self.logger.info('Take {} uncorrelated u_nk for state '
                                     '{}.'.format(len(subsample), index))
                    self.u_nk_sample_list.append(u_nk)

            self.dHdl_sample_list = []
            for index, dHdl in enumerate(self.dHdl_list):
                dHdl = dHdl[dHdl.index.get_level_values('time') > skiptime]
                subsample = statistical_inefficiency(dHdl, dHdl.sum(axis=1))
                if len(subsample) < threshold:
                    self.logger.warning('Number of dHdl {} for state {} is '
                                        'less than the threshold {}.'.format(
                        len(subsample), index, threshold))
                    self.logger.info('Take all the dHdl for state {}.'.format(index))
                    self.dHdl_sample_list.append(subsample)
                else:
                    self.logger.info('Take {} uncorrelated dHdl for state '
                                     '{}.'.format(len(subsample), index))
                    self.dHdl_sample_list.append(dHdl)

    def estimate(self, estimators=('mbar', 'bar', 'ti')):
        # Make estimators into a tuple
        if isinstance(estimators, str):
            estimators = (estimators, )

        self.logger.info(
            'Start running estimator: {}.'.format(','.join(estimators)))
        self.estimator = {}
        # Use unprocessed data if preprocess is not performed.
        try:
            dHdl = pd.concat(self.dHdl_sample_list)
        except AttributeError:
            dHdl = pd.concat(self.dHdl_list)
            self.logger.warning('dHdl has not been preprocessed.')
        self.logger.info(
            'A total {} lines of dHdl is used.'.format(len(dHdl)))

        try:
            u_nk = pd.concat(self.u_nk_sample_list)
        except AttributeError:
            u_nk = pd.concat(self.u_nk_list)
            self.logger.warning('u_nk has not been preprocessed.')
        self.logger.info(
            'A total {} lines of u_nk is used.'.format(len(u_nk)))

        for estimator in estimators:
            if estimator.lower() == 'mbar' and len(u_nk) > 0:
                self.logger.info('Run MBAR estimator.')
                self.estimator['mbar'] = MBAR().fit(u_nk)
            elif estimator.lower() == 'bar' and len(u_nk) > 0:
                self.logger.info('Run BAR estimator.')
                self.estimator['bar'] = BAR().fit(u_nk)
            elif estimator.lower() == 'ti' and len(dHdl) > 0:
                self.logger.info('Run TI estimator.')
                self.estimator['ti'] = TI().fit(dHdl)
            elif estimator.lower() == 'mbar' or estimator.lower() == 'bar':
                self.logger.warning('MBAR or BAR estimator require u_nk')
            else:
                self.logger.warning(
                    '{} is not a valid estimator.'.format(estimator))

    def write(self, resultfilename='result.out', units=None):
        self._update_units(units)

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
            except:
                stages = []
                self.logger.warning('No stage name found in dHdl or u_nk')
        for stage in stages:
            result_out.append([stage.split('-')[0][:9].rjust(9)+':  ', ])
        result_out.append(['TOTAL'.rjust(9) + ':  ', ])

        for estimator_name, estimator in self.estimator.items():
            self.logger.info('write the result from estimator {}'.format(
                estimator_name))
            # Write the estimator header
            result_out[0].append('---------------------')
            result_out[1].append('{} ({}) '.format(
                estimator_name.upper(), self.units).rjust(21))
            result_out[2].append('---------------------')
            for index in range(1, num_states):
                result_out[2+index].append('{:.3f}  +-  {:.3f}'.format(
                    estimator.delta_f_.iloc[index-1, index]*self.scaling_factor,
                    estimator.d_delta_f_.iloc[index-1, index]*self.scaling_factor
                ).rjust(21))

            result_out[2+num_states].append('---------------------')

            self.logger.info('write the staged result from estimator {}'.format(
                estimator_name))
            for index, stage in enumerate(stages):
                start = list(reversed(
                    [state[index] for state in estimator.states_])).index(0)
                start = num_states - start - 1
                end = [state[index] for state in estimator.states_].index(1)
                self.logger.info(
                    'Stage {} is from state {} to state {}.'.format(
                        stage, start, end))
                result = estimator.delta_f_.iloc[start, end]*self.scaling_factor
                if estimator_name != 'bar':
                    error = estimator.d_delta_f_.iloc[start, end]*self.scaling_factor
                else:
                    error = np.sqrt(sum(
                        [estimator.d_delta_f_.iloc[start, start+1]**2
                         for i in range(start, end + 1)])) * self.scaling_factor
                result_out[3 + num_states + index].append(
                    '{:.3f}  +-  {:.3f}'.format(result, error,).rjust(21))

            # Total result
            result = estimator.delta_f_.iloc[0, -1] * self.scaling_factor
            if estimator_name != 'bar':
                error = estimator.d_delta_f_.iloc[0, -1] * self.scaling_factor
            else:
                error = np.sqrt(sum(
                    [estimator.d_delta_f_.iloc[i, i + 1] ** 2
                     for i in range(num_states - 1)])) * self.scaling_factor
            result_out[3 + num_states + len(stages)].append(
                '{:.3f}  +-  {:.3f}'.format(result, error, ).rjust(21))
        self.logger.info('Write results:\n'+
                         '\n'.join([' '.join(line) for line in result_out]))
        with open(join(self.out, resultfilename), 'w') as f:
            f.write('\n'.join([' '.join(line) for line in result_out]))

    def plot_overlap_matrix(self, overlap='O_MBAR.pdf', ax=None):
        self.logger.info('Plot overlap matrix.')
        if 'mbar' in self.estimator:
            ax = plot_mbar_overlap_matrix(self.estimator['mbar'].overlap_matrix,
                                          ax=ax)
            ax.figure.savefig(join(self.out, overlap))
            self.logger.info('Plot overlap matrix to {} under {}.'
                             ''.format(self.out, overlap))
            return ax
        else:
            self.logger.warning('MBAR estimator not found. '
                                'Overlap matrix not plotted.')

    def plot_ti_dhdl(self, dhdl_TI='dhdl_TI.pdf', units=None, labels=None,
                     colors=None, ax=None):
        self._update_units(units)
        self.logger.info('Plot TI dHdl.')
        if 'ti' in self.estimator:
            ax = plot_ti_dhdl(self.estimator['ti'], units=self.units,
                              labels=labels, colors=colors, ax=ax,
                              scaling_factor=self.scaling_factor)
            ax.figure.savefig(join(self.out, dhdl_TI))
            self.logger.info('Plot TI dHdl to {} under {}.'
                             ''.format(dhdl_TI, self.out))

    def plot_dF_state(self, dF_state='dF_state.pdf', labels=None, colors=None,
                      units=None, orientation='portrait', nb=10):
        self._update_units(units)
        self.logger.info('Plot dF states.')
        fig = plot_dF_state(self.estimator.values(), labels=labels, colors=colors,
                            units=self.units,
                            scaling_factor=self.scaling_factor,
                            orientation=orientation, nb=nb)
        fig.savefig(join(self.out, dF_state))
        self.logger.info('Plot dF state to {} under {}.'
                         ''.format(dF_state, self.out))

    def check_convergence(self, forwrev, estimator='mbar', dF_t='dF_t.pdf',
                          units=None):
        self._update_units(units)
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
            except AttributeError:
                self.logger.warning('dHdl is not available.')

        try:
            u_nk_list = self.u_nk_sample_list
            self.logger.info('Subsampled u_nk is available.')
        except AttributeError:
            try:
                u_nk_list = self.u_nk_list
                self.logger.info('Subsampled u_nk not available, '
                                 'use original data instead.')
            except AttributeError:
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
        else:
            self.logger.warning(
                '{} is not a valid estimator.'.format(estimator))

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
            sample = pd.concat(sample)
            result = estimator_fit(sample)
            forward_list.append(result.delta_f_.iloc[0, -1])
            if estimator.lower() == 'bar':
                error = np.sqrt(sum(
                    [result.d_delta_f_.iloc[i, i + 1] ** 2
                     for i in range(len(result.d_delta_f_) - 1)]))
                forward_error_list.append(error)
            else:
                forward_error_list.append(result.d_delta_f_.iloc[0, -1])
            self.logger.info('{:.2f} +/- {:.2f} kBT'.format(forward_list[-1],
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
            sample = pd.concat(sample)
            result = estimator_fit(sample)
            backward_list.append(result.delta_f_.iloc[0, -1])
            if estimator.lower() == 'bar':
                error = np.sqrt(sum(
                    [result.d_delta_f_.iloc[i, i + 1] ** 2
                     for i in range(len(result.d_delta_f_) - 1)]))
                backward_error_list.append(error)
            else:
                backward_error_list.append(result.d_delta_f_.iloc[0, -1])
            self.logger.info('{:.2f} +/- {:.2f} kBT'.format(backward_list[-1],
                                                        backward_error_list[-1]))

        convergence = pd.DataFrame({'Forward (kBT)': forward_list,
                                    'F. Error (kBT)': forward_error_list,
                                    'Backward (kBT)': backward_list,
                                    'B. Error (kBT)': backward_error_list})

        self.convergence = convergence
        self.logger.info('Plot convergence analysis to {} under {}.'
                         ''.format(dF_t, self.out))
        ax = plot_convergence(np.array(forward_list) * self.scaling_factor,
                              np.array(forward_error_list) * self.scaling_factor,
                              np.array(backward_list) * self.scaling_factor,
                              np.array(backward_error_list) * self.scaling_factor,
                              units=self.units)
        ax.figure.savefig(join(self.out, dF_t))
