import os
from pathlib import Path

import pandas as pd
import pytest

from alchemlyb.workflows import base


class Test_automatic_base:
    @staticmethod
    @pytest.fixture(scope="session")
    def workflow(tmp_path_factory):
        outdir = tmp_path_factory.mktemp("out") / "abc" / "def"
        workflow = base.WorkflowBase(out=str(outdir))
        workflow.run()
        return workflow

    def test_outdir(self, workflow):
        assert Path(workflow.out).is_dir()

    def test_write(self, workflow):
        """Patch the output directory to tmpdir"""
        workflow.result.to_pickle(os.path.join(workflow.out, "result.pkl"))
        assert os.path.exists(os.path.join(workflow.out, "result.pkl"))

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
