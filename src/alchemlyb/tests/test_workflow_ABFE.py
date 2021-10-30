import numpy as np
import pytest
import os

from alchemlyb.workflows import ABFE
from alchemtest.gmx import load_ABFE, load_benzene

class Test_automatic_ABFE():
    '''Test the full automatic workflow for load_ABFE from alchemtest.gmx for
    three stage transformation.'''

    @staticmethod
    @pytest.fixture(scope='class')
    def workflow():
        dir = os.path.dirname(load_ABFE()['data']['complex'][0])
        workflow = ABFE(units='kcal/mol', software='Gromacs', dir=dir,
                        prefix='dhdl', suffix='xvg', T=310, skiptime=10,
                        uncorr='dhdl', threshold=50,
                        methods=('mbar', 'bar', 'ti'), out='./',
                        overlap='O_MBAR.pdf',
                        breakdown=True, forwrev=10, log='result.log')
        return workflow

    def test_read(self, workflow):
        '''test if the files has been loaded correctly.'''
        assert len(workflow.u_nk_list) == 30
        assert len(workflow.dHdl_list) == 30
        assert all([len(u_nk) == 1001 for u_nk in workflow.u_nk_list])
        assert all([len(dHdl) == 1001 for dHdl in workflow.dHdl_list])

    def test_subsample(self, workflow):
        '''Test if the data has been shrinked by subsampling.'''
        assert len(workflow.u_nk_sample_list) == 30
        assert len(workflow.dHdl_sample_list) == 30
        assert all([len(u_nk) < 1001 for u_nk in workflow.u_nk_sample_list])
        assert all([len(dHdl) < 1001 for dHdl in workflow.dHdl_sample_list])

    def test_estomator(self, workflow):
        '''Test if all three estimator has been used.'''
        assert len(workflow.estimator) == 3
        assert 'mbar' in workflow.estimator
        assert 'ti' in workflow.estimator
        assert 'bar' in workflow.estimator

    def test_summary(self, workflow):
        '''Test if if the summary is right.'''
        summary = workflow.generate_result()
        assert np.isclose(summary['MBAR']['Stages']['TOTAL'], 21.788, 0.1)

    def test_O_MBAR(self, workflow):
        '''test if the O_MBAR.pdf has been plotted.'''
        assert os.path.isfile('O_MBAR.pdf')
        os.remove('O_MBAR.pdf')

    def test_dhdl_TI(self, workflow):
        '''test if the dhdl_TI.pdf has been plotted.'''
        assert os.path.isfile('dhdl_TI.pdf')
        os.remove('dhdl_TI.pdf')

    def test_dF_state(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        assert os.path.isfile('dF_state.pdf')
        os.remove('dF_state.pdf')
        assert os.path.isfile('dF_state_long.pdf')
        os.remove('dF_state_long.pdf')

    def test_convergence(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        assert os.path.isfile('dF_t.pdf')
        os.remove('dF_t.pdf')
        assert len(workflow.convergence) == 10

class Test_manual_ABFE():
    '''Test the manual workflow for load_ABFE from alchemtest.gmx for three
    stage transformation.'''

    @staticmethod
    @pytest.fixture(scope='class')
    def workflow():
        dir = os.path.dirname(load_ABFE()['data']['complex'][0])
        workflow = ABFE(software='Gromacs', dir=dir, prefix='dhdl',
                        suffix='xvg', T=310)
        workflow.update_units('kcal/mol')
        workflow.preprocess(skiptime=10, uncorr='dhdl', threshold=50)
        workflow.estimate(methods=('mbar', 'bar', 'ti'))
        workflow.plot_overlap_matrix(overlap='O_MBAR.pdf')
        workflow.plot_ti_dhdl(dhdl_TI='dhdl_TI.pdf')
        workflow.plot_dF_state(dF_state='dF_state.pdf')
        workflow.check_convergence(10, dF_t='dF_t.pdf')
        return workflow

    def test_read(self, workflow):
        '''test if the files has been loaded correctly.'''
        assert len(workflow.u_nk_list) == 30
        assert len(workflow.dHdl_list) == 30
        assert all([len(u_nk) == 1001 for u_nk in workflow.u_nk_list])
        assert all([len(dHdl) == 1001 for dHdl in workflow.dHdl_list])

    def test_subsample(self, workflow):
        '''Test if the data has been shrinked by subsampling.'''
        assert len(workflow.u_nk_sample_list) == 30
        assert len(workflow.dHdl_sample_list) == 30
        assert all([len(u_nk) < 1001 for u_nk in workflow.u_nk_sample_list])
        assert all([len(dHdl) < 1001 for dHdl in workflow.dHdl_sample_list])

    def test_estomator(self, workflow):
        '''Test if all three estimator has been used.'''
        assert len(workflow.estimator) == 3
        assert 'mbar' in workflow.estimator
        assert 'ti' in workflow.estimator
        assert 'bar' in workflow.estimator

    def test_O_MBAR(self, workflow):
        '''test if the O_MBAR.pdf has been plotted.'''
        assert os.path.isfile('O_MBAR.pdf')
        os.remove('O_MBAR.pdf')

    def test_dhdl_TI(self, workflow):
        '''test if the dhdl_TI.pdf has been plotted.'''
        assert os.path.isfile('dhdl_TI.pdf')
        os.remove('dhdl_TI.pdf')

    def test_dF_state(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        assert os.path.isfile('dF_state.pdf')
        os.remove('dF_state.pdf')

    def test_convergence(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        assert os.path.isfile('dF_t.pdf')
        os.remove('dF_t.pdf')
        assert len(workflow.convergence) == 10

    def test_convergence_nosample_u_nk(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        u_nk_sample_list = workflow.u_nk_sample_list
        delattr(workflow, 'u_nk_sample_list')
        workflow.check_convergence(10)
        os.remove('dF_t.pdf')
        assert len(workflow.convergence) == 10
        workflow.u_nk_sample_list = u_nk_sample_list

    def test_convergence_nosample_dhdl(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        dHdl_sample_list = workflow.dHdl_sample_list
        delattr(workflow, 'dHdl_sample_list')
        workflow.check_convergence(10, estimator='ti')
        os.remove('dF_t.pdf')
        assert len(workflow.convergence) == 10
        workflow.dHdl_sample_list = dHdl_sample_list

    def test_convergence_dhdl(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        workflow.check_convergence(10, estimator='ti')
        os.remove('dF_t.pdf')
        assert len(workflow.convergence) == 10

    def test_convergence_TI(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        workflow.check_convergence(10, estimator='ti', dF_t='dF_t.pdf')
        assert os.path.isfile('dF_t.pdf')
        os.remove('dF_t.pdf')
        assert len(workflow.convergence) == 10

    def test_dhdl_TI_noTI(self, workflow):
        '''Test to plot the dhdl_TI when ti estimator is not there'''
        full_estimator = workflow.estimator
        workflow.estimator.pop('ti')
        workflow.plot_ti_dhdl(dhdl_TI='dhdl_TI.pdf')
        assert os.path.isfile('dhdl_TI.pdf') == False
        workflow.estimator = full_estimator

class Test_automatic_benzene():
    '''Test the full automatic workflow for load_benzene from alchemtest.gmx for
    single stage transformation.'''

    @staticmethod
    @pytest.fixture(scope='class')
    def workflow():
        dir = os.path.dirname(os.path.dirname(
            load_benzene()['data']['Coulomb'][0]))
        dir = os.path.join(dir, '*')
        workflow = ABFE(units='kcal/mol', software='Gromacs', dir=dir,
                        prefix='dhdl', suffix='bz2', T=310, skiptime=0,
                        uncorr='dhdl', threshold=50,
                        methods=('mbar', 'bar', 'ti'), out='./',
                        overlap='O_MBAR.pdf',
                        breakdown=True, forwrev=10, log='result.log')
        return workflow

    def test_read(self, workflow):
        '''test if the files has been loaded correctly.'''
        assert len(workflow.u_nk_list) == 5
        assert len(workflow.dHdl_list) == 5
        assert all([len(u_nk) == 4001 for u_nk in workflow.u_nk_list])
        assert all([len(dHdl) == 4001 for dHdl in workflow.dHdl_list])

    def test_estomator(self, workflow):
        '''Test if all three estimator has been used.'''
        assert len(workflow.estimator) == 3
        assert 'mbar' in workflow.estimator
        assert 'ti' in workflow.estimator
        assert 'bar' in workflow.estimator

    def test_O_MBAR(self, workflow):
        '''test if the O_MBAR.pdf has been plotted.'''
        assert os.path.isfile('O_MBAR.pdf')
        os.remove('O_MBAR.pdf')

    def test_dhdl_TI(self, workflow):
        '''test if the dhdl_TI.pdf has been plotted.'''
        assert os.path.isfile('dhdl_TI.pdf')
        os.remove('dhdl_TI.pdf')

    def test_dF_state(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        assert os.path.isfile('dF_state.pdf')
        os.remove('dF_state.pdf')
        assert os.path.isfile('dF_state_long.pdf')
        os.remove('dF_state_long.pdf')

    def test_convergence(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        assert os.path.isfile('dF_t.pdf')
        os.remove('dF_t.pdf')
        assert len(workflow.convergence) == 10

class Test_unpertubed_lambda():
    '''Test the if two lamdas present and one of them is not pertubed.

                                       fep  bound
time    fep-lambda bound-lambda
0.0     0.5        0             12.958159      0
10.0    0.5        0             -1.062968      0
20.0    0.5        0              1.019020      0
30.0    0.5        0              5.029051      0
40.0    0.5        0              7.768072      0

Where only fep-lambda changes but the bonded-lambda is always 0.
    '''

    @staticmethod
    @pytest.fixture(scope='class')
    def workflow():
        dir = os.path.dirname(os.path.dirname(
            load_benzene()['data']['Coulomb'][0]))
        dir = os.path.join(dir, '*')
        workflow = ABFE(software='Gromacs', dir=dir, prefix='dhdl',
                        suffix='bz2', T=310)
        # Block the n_uk
        workflow.u_nk_list = []
        # Add another lambda column
        for dHdl in workflow.dHdl_list:
            dHdl.insert(1, 'bound-lambda', [1.0, ] * len(dHdl))
            dHdl.insert(1, 'bound', [1.0, ] * len(dHdl))
            dHdl.set_index('bound-lambda', append=True, inplace=True)

        workflow.estimate(methods=('ti', ))
        workflow.plot_ti_dhdl(dhdl_TI='dhdl_TI.pdf')
        workflow.plot_dF_state(dF_state='dF_state.pdf')
        workflow.check_convergence(10, dF_t='dF_t.pdf', estimator='ti')
        return workflow

    def test_dhdl_TI(self, workflow):
        '''test if the dhdl_TI.pdf has been plotted.'''
        assert os.path.isfile('dhdl_TI.pdf')
        os.remove('dhdl_TI.pdf')

    def test_dF_state(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        assert os.path.isfile('dF_state.pdf')
        os.remove('dF_state.pdf')

    def test_convergence(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        assert os.path.isfile('dF_t.pdf')
        os.remove('dF_t.pdf')
        assert len(workflow.convergence) == 10

class Test_methods():
    '''Test various methods.'''

    @staticmethod
    @pytest.fixture(scope='class')
    def workflow():
        dir = os.path.dirname(os.path.dirname(
            load_benzene()['data']['Coulomb'][0]))
        dir = os.path.join(dir, '*')
        workflow = ABFE(software='Gromacs', dir=dir, prefix='dhdl',
                        suffix='bz2', T=310)
        return workflow

    def test_uncorr_threshold(self, workflow):
        original_u_nk = workflow.u_nk_list
        original_dHdl = workflow.dHdl_list
        workflow.u_nk_list = [u_nk[:40] for u_nk in original_u_nk]
        workflow.dHdl_list = [dHdl[:40] for dHdl in original_dHdl]
        workflow.preprocess(threshold=50)
        assert all([len(u_nk) == 40 for u_nk in workflow.u_nk_sample_list])
        assert all([len(dHdl) == 40 for dHdl in workflow.dHdl_sample_list])
        workflow.u_nk_list = original_u_nk
        workflow.dHdl_list = original_dHdl

    def test_single_estimator(self, workflow):
        workflow.estimate(methods='mbar')
        assert len(workflow.estimator) == 1
        assert 'mbar' in workflow.estimator

    def test_bar_convergence(self, workflow):
        workflow.check_convergence(10, estimator='bar')
        assert os.path.isfile('dF_t.pdf')
        os.remove('dF_t.pdf')

    def test_unprocessed_n_uk(self, workflow):
        workflow.u_nk_sample_list = []
        workflow.estimate()
        assert len(workflow.estimator) == 3
        assert 'mbar' in workflow.estimator
