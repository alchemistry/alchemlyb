import numpy as np
import pytest
import os

from alchemlyb.workflows.abfe import ABFE
from alchemtest.gmx import load_ABFE, load_benzene
from alchemtest.amber import load_bace_example

class Test_automatic_ABFE():
    '''Test the full automatic workflow for load_ABFE from alchemtest.gmx for
    three stage transformation.'''

    @staticmethod
    @pytest.fixture(scope='session')
    def workflow(tmp_path_factory):
        outdir = tmp_path_factory.mktemp("out")
        dir = os.path.dirname(load_ABFE()['data']['complex'][0])
        workflow = ABFE(units='kcal/mol', software='Gromacs', dir=dir,
                        prefix='dhdl', suffix='xvg', T=310, outdirectory=str(outdir))
        workflow.run(skiptime=10, uncorr='dhdl', threshold=50,
                     methods=('mbar', 'bar', 'ti'), overlap='O_MBAR.pdf',
                     breakdown=True, forwrev=10)
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

    def test_estimator(self, workflow):
        '''Test if all three estimators have been used.'''
        assert len(workflow.estimator) == 3
        assert 'mbar' in workflow.estimator
        assert 'ti' in workflow.estimator
        assert 'bar' in workflow.estimator

    def test_summary(self, workflow):
        '''Test if if the summary is right.'''
        summary = workflow.generate_result()
        assert np.isclose(summary['MBAR']['Stages']['TOTAL'], 21.8, 0.1)
        assert np.isclose(summary['TI']['Stages']['TOTAL'], 21.8, 0.1)
        assert np.isclose(summary['BAR']['Stages']['TOTAL'], 21.8, 0.1)

    def test_plot_O_MBAR(self, workflow):
        '''test if the O_MBAR.pdf has been plotted.'''
        assert os.path.isfile(os.path.join(workflow.out, 'O_MBAR.pdf'))

    def test_plot_dhdl_TI(self, workflow):
        '''test if the dhdl_TI.pdf has been plotted.'''
        assert os.path.isfile(os.path.join(workflow.out, 'dhdl_TI.pdf'))

    def test_plot_dF_state(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        assert os.path.isfile(os.path.join(workflow.out, 'dF_state.pdf'))
        assert os.path.isfile(os.path.join(workflow.out, 'dF_state_long.pdf'))

    def test_check_convergence(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        assert os.path.isfile(os.path.join(workflow.out, 'dF_t.pdf'))
        assert len(workflow.convergence) == 10

class Test_manual_ABFE(Test_automatic_ABFE):
    '''Test the manual workflow for load_ABFE from alchemtest.gmx for three
    stage transformation.'''

    @staticmethod
    @pytest.fixture(scope='session')
    def workflow(tmp_path_factory):
        outdir = tmp_path_factory.mktemp("out")
        dir = os.path.dirname(load_ABFE()['data']['complex'][0])
        workflow = ABFE(software='Gromacs', dir=dir, prefix='dhdl',
                        suffix='xvg', T=310, outdirectory=str(outdir))
        workflow.update_units('kcal/mol')
        workflow.read()
        workflow.preprocess(skiptime=10, uncorr='dhdl', threshold=50)
        workflow.estimate(methods=('mbar', 'bar', 'ti'))
        workflow.plot_overlap_matrix(overlap='O_MBAR.pdf')
        workflow.plot_ti_dhdl(dhdl_TI='dhdl_TI.pdf')
        workflow.plot_dF_state(dF_state='dF_state.pdf')
        workflow.check_convergence(10, dF_t='dF_t.pdf')
        return workflow

    def test_plot_dF_state(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        assert os.path.isfile(os.path.join(workflow.out, 'dF_state.pdf'))

    def test_convergence_nosample_u_nk(self, workflow, monkeypatch):
        '''test if the convergence routine would use the unsampled data
        when the data has not been subsampled.'''
        monkeypatch.setattr(workflow, 'u_nk_sample_list',
                            None)
        workflow.check_convergence(10)
        assert len(workflow.convergence) == 10

    def test_dhdl_TI_noTI(self, workflow, monkeypatch):
        '''Test to plot the dhdl_TI when ti estimator is not there'''
        no_TI = workflow.estimator
        no_TI.pop('ti')
        monkeypatch.setattr(workflow, 'estimator',
                            no_TI)
        with pytest.raises(ValueError):
            workflow.plot_ti_dhdl(dhdl_TI='dhdl_TI.pdf')

class Test_automatic_benzene():
    '''Test the full automatic workflow for load_benzene from alchemtest.gmx for
    single stage transformation.'''

    @staticmethod
    @pytest.fixture(scope='session')
    def workflow(tmp_path_factory):
        outdir = tmp_path_factory.mktemp("out")
        dir = os.path.dirname(os.path.dirname(
            load_benzene()['data']['Coulomb'][0]))
        dir = os.path.join(dir, '*')
        workflow = ABFE(units='kcal/mol', software='Gromacs', dir=dir,
                        prefix='dhdl', suffix='bz2', T=310,
                        outdirectory=outdir)
        workflow.run(skiptime=0, uncorr='dhdl', threshold=50,
                        methods=('mbar', 'bar', 'ti'), overlap='O_MBAR.pdf',
                        breakdown=True, forwrev=10)
        return workflow

    def test_read(self, workflow):
        '''test if the files has been loaded correctly.'''
        assert len(workflow.u_nk_list) == 5
        assert len(workflow.dHdl_list) == 5
        assert all([len(u_nk) == 4001 for u_nk in workflow.u_nk_list])
        assert all([len(dHdl) == 4001 for dHdl in workflow.dHdl_list])

    def test_estimator(self, workflow):
        '''Test if all three estimators have been used.'''
        assert len(workflow.estimator) == 3
        assert 'mbar' in workflow.estimator
        assert 'ti' in workflow.estimator
        assert 'bar' in workflow.estimator

    def test_O_MBAR(self, workflow):
        '''test if the O_MBAR.pdf has been plotted.'''
        assert os.path.isfile(os.path.join(workflow.out, 'O_MBAR.pdf'))

    def test_dhdl_TI(self, workflow):
        '''test if the dhdl_TI.pdf has been plotted.'''
        assert os.path.isfile(os.path.join(workflow.out, 'dhdl_TI.pdf'))

    def test_dF_state(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        assert os.path.isfile(os.path.join(workflow.out, 'dF_state.pdf'))

    def test_convergence(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        assert os.path.isfile(os.path.join(workflow.out, 'dF_t.pdf'))
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
    @pytest.fixture(scope='session')
    def workflow(tmp_path_factory):
        outdir = tmp_path_factory.mktemp("out")
        dir = os.path.dirname(os.path.dirname(
            load_benzene()['data']['Coulomb'][0]))
        dir = os.path.join(dir, '*')
        workflow = ABFE(software='Gromacs', dir=dir, prefix='dhdl',
                        suffix='bz2', T=310, outdirectory=outdir)
        workflow.read()
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
        assert os.path.isfile(os.path.join(workflow.out, 'dhdl_TI.pdf'))

    def test_dF_state(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        assert os.path.isfile(os.path.join(workflow.out, 'dF_state.pdf'))

    def test_convergence(self, workflow):
        '''test if the dF_state.pdf has been plotted.'''
        assert os.path.isfile(os.path.join(workflow.out, 'dF_t.pdf'))
        assert len(workflow.convergence) == 10

    def test_single_estimator_ti(self, workflow):
        workflow.estimate(methods='ti')
        summary = workflow.generate_result()
        assert np.isclose(summary['TI']['Stages']['TOTAL'], 2.946, 0.1)

class Test_methods():
    '''Test various methods.'''

    @staticmethod
    @pytest.fixture(scope='class')
    def workflow(tmp_path_factory):
        outdir = tmp_path_factory.mktemp("out")
        dir = os.path.dirname(os.path.dirname(
            load_benzene()['data']['Coulomb'][0]))
        dir = os.path.join(dir, '*')
        workflow = ABFE(software='Gromacs', dir=dir, prefix='dhdl',
                        suffix='bz2', T=310, outdirectory=outdir)
        workflow.read()
        return workflow

    def test_run_none(self, workflow):
        '''Don't run anything'''
        workflow.run(uncorr=None, methods=None, overlap=None, breakdown=None,
                     forwrev=None)

    def test_uncorr_threshold(self, workflow, monkeypatch):
        '''Test if the full data will be used when the number of data points
        are less than the threshold.'''
        monkeypatch.setattr(workflow, 'u_nk_list',
                            [u_nk[:40] for u_nk in workflow.u_nk_list])
        monkeypatch.setattr(workflow, 'dHdl_list',
                            [dHdl[:40] for dHdl in workflow.dHdl_list])
        workflow.preprocess(threshold=50)
        assert all([len(u_nk) == 40 for u_nk in workflow.u_nk_sample_list])
        assert all([len(dHdl) == 40 for dHdl in workflow.dHdl_sample_list])

    def test_single_estimator_mbar(self, workflow):
        workflow.estimate(methods='mbar')
        assert len(workflow.estimator) == 1
        assert 'mbar' in workflow.estimator
        summary = workflow.generate_result()
        assert np.isclose(summary['MBAR']['Stages']['TOTAL'], 2.946, 0.1)

    def test_single_estimator_ti(self, workflow):
        workflow.estimate(methods='ti')
        summary = workflow.generate_result()
        assert np.isclose(summary['TI']['Stages']['TOTAL'], 2.946, 0.1)

    def test_bar_convergence(self, workflow):
        workflow.check_convergence(10, estimator='bar')
        assert len(workflow.convergence) == 10

    def test_ti_convergence(self, workflow):
        workflow.check_convergence(10, estimator='ti')
        assert len(workflow.convergence) == 10

    def test_unprocessed_n_uk(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, 'u_nk_sample_list',
                            None)
        workflow.estimate()
        assert len(workflow.estimator) == 3
        assert 'mbar' in workflow.estimator

    def test_unprocessed_dhdl(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, 'dHdl_sample_list',
                            None)
        workflow.check_convergence(10, estimator='ti')
        assert len(workflow.convergence) == 10

class Test_automatic_amber():
    '''Test the full automatic workflow for load_ABFE from alchemtest.gmx for
    three stage transformation.'''

    @staticmethod
    @pytest.fixture(scope='session')
    def workflow(tmp_path_factory):
        outdir = tmp_path_factory.mktemp("out")
        dir, _ = os.path.split(
            os.path.dirname(load_bace_example()['data']['complex']['vdw'][0]))

        workflow = ABFE(units='kcal/mol', software='Amber', dir=dir,
                        prefix='ti', suffix='bz2', T=310, outdirectory=str(outdir))
        workflow.read()
        workflow.estimate(methods='ti')
        return workflow

    def test_summary(self, workflow):
        '''Test if if the summary is right.'''
        summary = workflow.generate_result()
        assert np.isclose(summary['TI']['Stages']['TOTAL'], 1.40405980473, 0.1)
