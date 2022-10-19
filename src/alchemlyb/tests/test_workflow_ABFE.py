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
        workflow = ABFE(units='kcal/mol', software='GROMACS', dir=dir,
                        prefix='dhdl', suffix='xvg', T=310, outdirectory=str(outdir))
        workflow.run(skiptime=10, uncorr='dE', threshold=50,
                     estimators=('MBAR', 'BAR', 'TI'), overlap='O_MBAR.pdf',
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
        assert 'MBAR' in workflow.estimator
        assert 'TI' in workflow.estimator
        assert 'BAR' in workflow.estimator

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

    def test_estimator_method(self, workflow, monkeypatch):
        '''Test if the method keyword could be passed to the AutoMBAR estimator.'''
        monkeypatch.setattr(workflow, 'estimator',
                            dict())
        workflow.estimate(estimators='MBAR', method='adaptive')
        assert 'MBAR' in workflow.estimator

    def test_convergence_method(self, workflow, monkeypatch):
        '''Test if the method keyword could be passed to the AutoMBAR estimator from convergence.'''
        monkeypatch.setattr(workflow, 'convergence', None)
        workflow.check_convergence(2, estimator='MBAR', method='adaptive')
        assert len(workflow.convergence) == 2

class Test_manual_ABFE(Test_automatic_ABFE):
    '''Test the manual workflow for load_ABFE from alchemtest.gmx for three
    stage transformation.'''

    @staticmethod
    @pytest.fixture(scope='session')
    def workflow(tmp_path_factory):
        outdir = tmp_path_factory.mktemp("out")
        dir = os.path.dirname(load_ABFE()['data']['complex'][0])
        workflow = ABFE(software='GROMACS', dir=dir, prefix='dhdl',
                        suffix='xvg', T=310, outdirectory=str(outdir))
        workflow.update_units('kcal/mol')
        workflow.read()
        workflow.preprocess(skiptime=10, uncorr='dE', threshold=50)
        workflow.estimate(estimators=('MBAR', 'BAR', 'TI'))
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
        no_TI.pop('TI')
        monkeypatch.setattr(workflow, 'estimator',
                            no_TI)
        with pytest.raises(ValueError):
            workflow.plot_ti_dhdl(dhdl_TI='dhdl_TI.pdf')

    def test_noMBAR_for_plot_overlap_matrix(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, 'estimator', {})
        assert workflow.plot_overlap_matrix() is None

    def test_no_u_nk_for_check_convergence(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, 'u_nk_list', None)
        monkeypatch.setattr(workflow, 'u_nk_sample_list', None)
        with pytest.raises(ValueError):
            workflow.check_convergence(10, estimator='MBAR')

    def test_no_dHdl_for_check_convergence(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, 'dHdl_list', None)
        monkeypatch.setattr(workflow, 'dHdl_sample_list', None)
        with pytest.raises(ValueError):
            workflow.check_convergence(10, estimator='TI')

    def test_no_update_units(self, workflow):
        assert workflow.update_units() is None

    def test_no_name_estimate(self, workflow):
        with pytest.raises(ValueError):
            workflow.estimate('aaa')


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
        workflow = ABFE(units='kcal/mol', software='GROMACS', dir=dir,
                        prefix='dhdl', suffix='bz2', T=310,
                        outdirectory=outdir)
        workflow.run(skiptime=0, uncorr='dE', threshold=50,
                        estimators=('MBAR', 'BAR', 'TI'), overlap='O_MBAR.pdf',
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
        assert 'MBAR' in workflow.estimator
        assert 'TI' in workflow.estimator
        assert 'BAR' in workflow.estimator

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
        workflow = ABFE(software='GROMACS', dir=dir, prefix='dhdl',
                        suffix='bz2', T=310, outdirectory=outdir)
        workflow.read()
        # Block the n_uk
        workflow.u_nk_list = []
        # Add another lambda column
        for dHdl in workflow.dHdl_list:
            dHdl.insert(1, 'bound-lambda', [1.0, ] * len(dHdl))
            dHdl.insert(1, 'bound', [1.0, ] * len(dHdl))
            dHdl.set_index('bound-lambda', append=True, inplace=True)

        workflow.estimate(estimators=('TI', ))
        workflow.plot_ti_dhdl(dhdl_TI='dhdl_TI.pdf')
        workflow.plot_dF_state(dF_state='dF_state.pdf')
        workflow.check_convergence(10, dF_t='dF_t.pdf', estimator='TI')
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
        workflow.estimate(estimators='TI')
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
        workflow = ABFE(software='GROMACS', dir=dir, prefix='dhdl',
                        suffix='bz2', T=310, outdirectory=outdir)
        workflow.read()
        return workflow

    def test_run_none(self, workflow):
        '''Don't run anything'''
        workflow.run(uncorr=None, estimators=None, overlap=None, breakdown=None,
                     forwrev=None)

    def test_run_single_estimator(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, 'u_nk_list', [])
        monkeypatch.setattr(workflow, 'dHdl_list', [])
        workflow.run(uncorr=None, estimators='MBAR', overlap=None, breakdown=True,
                     forwrev=None)

    def test_run_invalid_estimator(self, workflow):
        with pytest.raises(ValueError,
                           match=r'Estimator aaa is not supported.'):
            workflow.run(uncorr=None, estimators='aaa', overlap=None, breakdown=None,
                         forwrev=None)

    @pytest.mark.parametrize('read_u_nk', [True, False])
    @pytest.mark.parametrize('read_dHdl', [True, False])
    def test_read_TI_FEP(self, workflow, monkeypatch, read_u_nk, read_dHdl):
        monkeypatch.setattr(workflow, 'u_nk_list', [])
        monkeypatch.setattr(workflow, 'dHdl_list', [])
        workflow.read(read_u_nk, read_dHdl)
        if read_u_nk:
            assert len(workflow.u_nk_list) == 5
        else:
            assert len(workflow.u_nk_list) == 0

        if read_dHdl:
            assert len(workflow.dHdl_list) == 5
        else:
            assert len(workflow.dHdl_list) == 0

    def test_read_invalid_u_nk(self, workflow, monkeypatch):
        def extract_u_nk(self, T):
            raise IOError('Error read u_nk.')
        monkeypatch.setattr(workflow, '_extract_u_nk',
                            extract_u_nk)
        with pytest.raises(OSError,
                           match=r'Error reading u_nk .*dhdl\.xvg\.bz2'):
            workflow.read()

    def test_read_invalid_dHdl(self, workflow, monkeypatch):
        def extract_dHdl(self, T):
            raise IOError('Error read dHdl.')
        monkeypatch.setattr(workflow, '_extract_dHdl',
                            extract_dHdl)
        with pytest.raises(OSError,
                           match=r'Error reading dHdl .*dhdl\.xvg\.bz2'):
            workflow.read()

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

    def test_no_u_nk_preprocess(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, 'u_nk_list', [])
        workflow.preprocess(threshold=50)
        assert len(workflow.u_nk_list) == 0

    def test_no_dHdl_preprocess(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, 'dHdl_list', [])
        workflow.preprocess(threshold=50)
        assert len(workflow.dHdl_list) == 0

    def test_single_estimator_mbar(self, workflow):
        workflow.estimate(estimators='MBAR')
        assert len(workflow.estimator) == 1
        assert 'MBAR' in workflow.estimator
        summary = workflow.generate_result()
        assert np.isclose(summary['MBAR']['Stages']['TOTAL'], 2.946, 0.1)

    def test_single_estimator_ti(self, workflow):
        workflow.estimate(estimators='TI')
        summary = workflow.generate_result()
        assert np.isclose(summary['TI']['Stages']['TOTAL'], 2.946, 0.1)

    def test_bar_convergence(self, workflow):
        workflow.check_convergence(10, estimator='BAR')
        assert len(workflow.convergence) == 10

    def test_convergence_invalid_estimator(self, workflow):
        with pytest.raises(ValueError):
            workflow.check_convergence(10, estimator='aaa')

    def test_ti_convergence(self, workflow):
        workflow.check_convergence(10, estimator='TI')
        assert len(workflow.convergence) == 10

    def test_unprocessed_n_uk(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, 'u_nk_sample_list',
                            None)
        workflow.estimate()
        assert len(workflow.estimator) == 3
        assert 'MBAR' in workflow.estimator

    def test_unprocessed_dhdl(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, 'dHdl_sample_list',
                            None)
        workflow.check_convergence(10, estimator='TI')
        assert len(workflow.convergence) == 10

class Test_automatic_amber():
    '''Test the full automatic workflow for load_ABFE from alchemtest.amber for
    three stage transformation.'''

    @staticmethod
    @pytest.fixture(scope='session')
    def workflow(tmp_path_factory):
        outdir = tmp_path_factory.mktemp("out")
        dir, _ = os.path.split(
            os.path.dirname(load_bace_example()['data']['complex']['vdw'][0]))

        workflow = ABFE(units='kcal/mol', software='AMBER', dir=dir,
                        prefix='ti', suffix='bz2', T=298.0, outdirectory=str(
                outdir))
        workflow.read()
        workflow.estimate(estimators='TI')
        return workflow

    def test_summary(self, workflow):
        '''Test if if the summary is right.'''
        summary = workflow.generate_result()
        assert np.isclose(summary['TI']['Stages']['TOTAL'], 1.40405980473, 0.1)

def test_no_parser():
    with pytest.raises(NotImplementedError):
        workflow = ABFE(units='kcal/mol', software='aaa',
                        prefix='ti', suffix='bz2', T=298.0)
