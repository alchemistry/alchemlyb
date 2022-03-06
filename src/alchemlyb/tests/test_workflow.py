import pytest
from alchemlyb.workflows import base
import pandas as pd
from unittest.mock import patch
import os

class Test_automatic_base():
    @staticmethod
    @pytest.fixture(scope='class')
    def workflow():
        workflow = base.WorkflowBase()
        workflow.run()
        yield workflow

    # def test_write(self, workflow, tmpdir):
    #     '''Patch the output directory to tmpdir'''
    #     with patch('workflow.out', tmpdir.strpath):
    #         workflow.result.to_pickle(os.path.join(workflow.out, 'result.pkl'))

    def test_read(self, workflow):
        assert len(workflow.u_nk_list) == 0
        assert len(workflow.dHdl_list) == 0

    def test_subsample(self, workflow):
        assert len(workflow.u_nk_sample_list) == 0
        assert len(workflow.dHdl_sample_list) == 0

    def test_estimator(self, workflow):
        assert isinstance(workflow.result, pd.DataFrame)

    def test_convergence(self, workflow):
        assert isinstance(workflow.convergence, pd.DataFrame)
