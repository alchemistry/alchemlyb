import os
import pandas as pd
import logging

from ..parsing import gmx, amber, namd, gomc
from ..preprocessing.subsampling import statistical_inefficiency
from ..estimators import MBAR, BAR, TI

class ABFE():
    def __init__(self, software='Gromacs', dir='./', prefix='dhdl',
                 suffix='xvg', T=298, skiptime=None, uncorr=None,
                 threshold=50, estimator=None, out='./', forwrev=0):
        self.logger = logging.getLogger('alchemlyb.workflows.ABFE')
        self.logger.info('Initialise Alchemlyb ABFE Workflow')
        self.logger.info('Finding files with prefix: {}, suffix: {} under '
                         'directory {} produced by {}'.format(prefix, suffix,
                                                              dir, software))
        self.file_list = []
        file_list = os.listdir(dir)
        for file in file_list:
            if file[:len(prefix)] == prefix and file[-len(suffix):] == suffix:
                self.file_list.append(os.path.join(dir, file))

        self.logger.info('Found {} xvg files.'.format(len(self.file_list)))
        self.logger.debug('File list: \n {}'.format('\n'.join(self.file_list)))

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

        self.u_nk_list = []
        self.dHdl_list = []
        for xvg in self.file_list:
            try:
                u_nk = extract_u_nk(xvg, T=T)
                self.logger.debug(
                    'Reading {} lines of u_nk from {}'.format(len(u_nk), xvg))
                self.u_nk_list.append(u_nk)
            except:
                self.logger.warning(
                    'Error reading read u_nk from {}.'.format(xvg))

            try:
                dhdl = extract_dHdl(xvg, T=T)
                self.logger.debug(
                    'Reading {} lines of dhdl from {}'.format(len(dhdl), xvg))
                self.dHdl_list.append(dhdl)
            except:
                self.logger.warning(
                    'Error reading read dhdl from {}.'.format(xvg))

        # Sort the files according to the state
        self.u_nk_list.sort(key=lambda x: x.attrs['state'])
        self.dHdl_list.sort(key=lambda x: x.attrs['state'])

        if skiptime is not None and uncorr is not None:
            self.preprocess(skiptime=skiptime, uncorr=uncorr,
                               threshold=threshold)
        if estimator is not None:
            self.estimate(estimator, out=out)



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
                # Get rid of the skiptime
                u_nk = u_nk[u_nk.index.get_level_values('time')>skiptime]
                if uncorr == 'dhdl':
                    # Find the current column index
                    # Select the first row and remove the first column (Time)
                    key = u_nk.index.values[0][1:]
                    col = u_nk[key]
                    subsample = statistical_inefficiency(u_nk, u_nk[key])
                elif uncorr == 'dhdl_all':
                    subsample = statistical_inefficiency(u_nk, u_nk.sum())
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
                subsample = statistical_inefficiency(dHdl, dHdl)
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

    def estimate(self, estimators=('mbar', 'bar', 'ti'), out='./'):
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


















