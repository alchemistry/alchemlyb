import os

import numpy as np
import pytest
from alchemtest.amber import load_bace_example
from alchemtest.gmx import load_ABFE

import alchemlyb.parsing.amber
from alchemlyb.workflows.abfe import ABFE


@pytest.fixture(scope="module")
def workflow(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("out")
    dir = os.path.dirname(load_ABFE()["data"]["complex"][0])
    workflow = ABFE(
        units="kcal/mol",
        software="GROMACS",
        dir=dir,
        prefix="dhdl",
        suffix="xvg",
        T=310,
        outdirectory=str(outdir),
    )
    workflow.run(
        skiptime=10,
        uncorr="dE",
        threshold=50,
        estimators=("MBAR", "BAR", "TI"),
        overlap="O_MBAR.pdf",
        breakdown=True,
        forwrev=10,
    )
    return workflow


class TestInit:
    def test_nofilematch(self, tmp_path):
        with pytest.raises(ValueError, match="No file has been matched to"):
            ABFE(
                dir=str(tmp_path),
                prefix="dhdl",
                suffix="xvg",
                T=310,
            )

    def test_notdir(self):
        with pytest.raises(ValueError, match="The input directory `dir`="):
            ABFE(
                dir="abfasfsd",
                prefix="dhdl",
                suffix="xvg",
                T=310,
            )

    def test_wildcard_in_dir(self):
        with pytest.raises(ValueError):
            with pytest.warns(match="A real directory is expected in `dir`="):
                ABFE(
                    dir="/*/",
                    prefix="dhdl",
                    suffix="xvg",
                    T=310,
                )


class TestRun:
    def test_none(self, workflow):
        """Don't run anything"""
        workflow.run(
            uncorr=None, estimators=None, overlap=None, breakdown=None, forwrev=None
        )

    def test_invalid_estimator(self, workflow):
        with pytest.raises(ValueError, match=r"Estimator aaa is not supported."):
            workflow.run(
                uncorr=None,
                estimators="aaa",
                overlap=None,
                breakdown=None,
                forwrev=None,
            )

    def test_single_estimator(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "u_nk_list", [])
        monkeypatch.setattr(workflow, "dHdl_list", [])
        monkeypatch.setattr(workflow, "u_nk_sample_list", [])
        monkeypatch.setattr(workflow, "dHdl_sample_list", [])
        monkeypatch.setattr(workflow, "estimator", dict())
        workflow.run(
            uncorr=None, estimators="MBAR", overlap=None, breakdown=True, forwrev=None
        )
        assert "MBAR" in workflow.estimator

    @pytest.mark.parametrize("forwrev", [None, False, 0])
    def test_no_forwrev(self, workflow, monkeypatch, forwrev):
        monkeypatch.setattr(workflow, "convergence", None)
        workflow.run(
            uncorr=None, estimators=None, overlap=None, breakdown=None, forwrev=forwrev
        )
        assert workflow.convergence is None


class TestRead:
    def test_default(self, workflow):
        """test if the files has been loaded correctly."""
        assert len(workflow.u_nk_list) == 30
        assert len(workflow.dHdl_list) == 30
        assert all([len(u_nk) == 1001 for u_nk in workflow.u_nk_list])
        assert all([len(dHdl) == 1001 for dHdl in workflow.dHdl_list])

    def test_no_parser(self):
        dir = os.path.dirname(load_ABFE()["data"]["complex"][0])
        with pytest.raises(NotImplementedError):
            workflow = ABFE(
                units="kcal/mol",
                software="aaa",
                dir=dir,
                prefix="dhdl",
                suffix="xvg",
                T=310,
            )

    @pytest.mark.parametrize("read_u_nk", [True, False])
    @pytest.mark.parametrize("read_dHdl", [True, False])
    def test_read_TI_FEP(self, workflow, monkeypatch, read_u_nk, read_dHdl):
        monkeypatch.setattr(workflow, "u_nk_list", [])
        monkeypatch.setattr(workflow, "dHdl_list", [])
        monkeypatch.setattr(workflow, "u_nk_sample_list", [])
        monkeypatch.setattr(workflow, "dHdl_sample_list", [])
        workflow.read(read_u_nk, read_dHdl)
        if read_u_nk:
            assert len(workflow.u_nk_list) == 30
        else:
            assert len(workflow.u_nk_list) == 0

        if read_dHdl:
            assert len(workflow.dHdl_list) == 30
        else:
            assert len(workflow.dHdl_list) == 0

    def test_read_invalid_u_nk(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "u_nk_sample_list", [])
        monkeypatch.setattr(workflow, "dHdl_sample_list", [])

        def extract_u_nk(self, T):
            raise IOError("Error read u_nk.")

        monkeypatch.setattr(workflow, "_extract_u_nk", extract_u_nk)
        with pytest.raises(OSError, match=r"Error reading u_nk"):
            workflow.read()

    def test_read_invalid_dHdl(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "u_nk_sample_list", [])
        monkeypatch.setattr(workflow, "dHdl_sample_list", [])

        def extract_dHdl(self, T):
            raise IOError("Error read dHdl.")

        monkeypatch.setattr(workflow, "_extract_dHdl", extract_dHdl)
        with pytest.raises(OSError, match=r"Error reading dHdl"):
            workflow.read()


class TestSubsample:
    def test_default(self, workflow):
        """Test if the data has been shrinked by subsampling."""
        assert len(workflow.u_nk_sample_list) == 30
        assert len(workflow.dHdl_sample_list) == 30
        assert all([len(u_nk) < 1001 for u_nk in workflow.u_nk_sample_list])
        assert all([len(dHdl) < 1001 for dHdl in workflow.dHdl_sample_list])

    def test_uncorr_threshold(self, workflow, monkeypatch):
        """Test if the full data will be used when the number of data points
        are less than the threshold."""
        monkeypatch.setattr(
            workflow, "u_nk_list", [u_nk[:40] for u_nk in workflow.u_nk_list]
        )
        monkeypatch.setattr(
            workflow, "dHdl_list", [dHdl[:40] for dHdl in workflow.dHdl_list]
        )
        monkeypatch.setattr(workflow, "u_nk_sample_list", [])
        monkeypatch.setattr(workflow, "dHdl_sample_list", [])
        workflow.preprocess(threshold=50)
        assert all([len(u_nk) == 40 for u_nk in workflow.u_nk_sample_list])
        assert all([len(dHdl) == 40 for dHdl in workflow.dHdl_sample_list])

    def test_no_u_nk_preprocess(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "u_nk_list", [])
        monkeypatch.setattr(workflow, "u_nk_sample_list", [])
        monkeypatch.setattr(workflow, "dHdl_sample_list", [])
        workflow.preprocess(threshold=50)
        assert len(workflow.u_nk_list) == 0

    def test_no_dHdl_preprocess(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "dHdl_list", [])
        monkeypatch.setattr(workflow, "u_nk_sample_list", [])
        monkeypatch.setattr(workflow, "dHdl_sample_list", [])
        workflow.preprocess(threshold=50)
        assert len(workflow.dHdl_list) == 0


class TestEstimator:
    def test_default(self, workflow):
        """Test if all three estimators have been used."""
        assert len(workflow.estimator) == 3
        assert "MBAR" in workflow.estimator
        assert "TI" in workflow.estimator
        assert "BAR" in workflow.estimator

    def test_method(self, workflow, monkeypatch):
        """Test if the method keyword could be passed to the AutoMBAR estimator."""
        monkeypatch.setattr(workflow, "estimator", dict())
        workflow.estimate(estimators="MBAR", method="adaptive")
        assert "MBAR" in workflow.estimator

    def test_single_estimator_ti(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "estimator", dict())
        monkeypatch.setattr(workflow, "summary", None)
        workflow.estimate(estimators="TI")
        summary = workflow.generate_result()
        assert np.isclose(summary["TI"]["Stages"]["TOTAL"], 2.946, 0.1)

    def test_summary(self, workflow):
        """Test if if the summary is right."""
        summary = workflow.generate_result()
        assert np.isclose(summary["MBAR"]["Stages"]["TOTAL"], 21.8, 0.1)
        assert np.isclose(summary["TI"]["Stages"]["TOTAL"], 21.8, 0.1)
        assert np.isclose(summary["BAR"]["Stages"]["TOTAL"], 21.8, 0.1)

    def test_no_name_estimate(self, workflow):
        with pytest.raises(ValueError):
            workflow.estimate("aaa")

    def test_single_estimator_mbar(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "estimator", dict())
        workflow.estimate(estimators="MBAR")
        assert len(workflow.estimator) == 1
        assert "MBAR" in workflow.estimator
        summary = workflow.generate_result()
        assert np.isclose(summary["MBAR"]["Stages"]["TOTAL"], 21.645742066696315, 0.1)

    def test_mbar_n_bootstraps(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "estimator", dict())
        workflow.estimate(estimators="MBAR", n_bootstraps=2)
        summary = workflow.generate_result()
        bootstrap_error = summary["MBAR_Error"]["Stages"]["TOTAL"]
        monkeypatch.setattr(workflow, "estimator", dict())
        workflow.estimate(estimators="MBAR", n_bootstraps=0)
        summary = workflow.generate_result()
        non_bootstrap_error = summary["MBAR_Error"]["Stages"]["TOTAL"]
        assert bootstrap_error != non_bootstrap_error

    def test_single_estimator_ti(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "estimator", dict())
        monkeypatch.setattr(workflow, "summary", None)
        workflow.estimate(estimators="TI")
        summary = workflow.generate_result()
        assert np.isclose(summary["TI"]["Stages"]["TOTAL"], 21.51472826028906, 0.1)

    def test_unprocessed_u_nk(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "u_nk_sample_list", None)
        monkeypatch.setattr(workflow, "estimator", dict())
        workflow.estimate()
        assert len(workflow.estimator) == 3
        assert "MBAR" in workflow.estimator

    def test_unpertubed_lambda(self, workflow, monkeypatch, gmx_benzene_Coulomb_dHdl):
        """Test the if two lamdas present and one of them is not pertubed.

                                               fep  bound
        time    fep-lambda bound-lambda
        0.0     0.5        0             12.958159      0
        10.0    0.5        0             -1.062968      0
        20.0    0.5        0              1.019020      0
        30.0    0.5        0              5.029051      0
        40.0    0.5        0              7.768072      0

        Where only fep-lambda changes but the bonded-lambda is always 0.
        """
        monkeypatch.setattr(workflow, "u_nk_list", [])
        monkeypatch.setattr(workflow, "u_nk_sample_list", None)
        monkeypatch.setattr(workflow, "dHdl_list", gmx_benzene_Coulomb_dHdl)
        monkeypatch.setattr(workflow, "dHdl_sample_list", gmx_benzene_Coulomb_dHdl)
        monkeypatch.setattr(workflow, "estimator", dict())
        monkeypatch.setattr(workflow, "summary", None)
        # Add another lambda column
        for dHdl in workflow.dHdl_sample_list:
            dHdl.insert(1, "bound-lambda", [1.0] * len(dHdl))
            dHdl.insert(1, "bound", [1.0] * len(dHdl))
            dHdl.set_index("bound-lambda", append=True, inplace=True)

        workflow.estimate(estimators="TI")
        summary = workflow.generate_result()
        assert np.isclose(summary["TI"]["Stages"]["bound"], 0)


class TestVisualisation:
    def test_plot_O_MBAR(self, workflow):
        """test if the O_MBAR.pdf has been plotted."""
        assert os.path.isfile(os.path.join(workflow.out, "O_MBAR.pdf"))

    def test_plot_dhdl_TI(self, workflow):
        """test if the dhdl_TI.pdf has been plotted."""
        assert os.path.isfile(os.path.join(workflow.out, "dhdl_TI.pdf"))

    def test_plot_dF_state(self, workflow):
        """test if the dF_state.pdf has been plotted."""
        assert os.path.isfile(os.path.join(workflow.out, "dF_state.pdf"))
        assert os.path.isfile(os.path.join(workflow.out, "dF_state_long.pdf"))

    def test_dhdl_TI_noTI(self, workflow, monkeypatch):
        """Test to plot the dhdl_TI when ti estimator is not there"""
        no_TI = workflow.estimator.copy()
        no_TI.pop("TI")
        monkeypatch.setattr(workflow, "estimator", no_TI)
        with pytest.raises(ValueError):
            workflow.plot_ti_dhdl(dhdl_TI="dhdl_TI.pdf")

    def test_noMBAR_for_plot_overlap_matrix(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "estimator", {})
        assert workflow.plot_overlap_matrix() is None


class TestConvergence:
    def test_default(self, workflow):
        """test if the dF_state.pdf has been plotted."""
        assert os.path.isfile(os.path.join(workflow.out, "dF_t.pdf"))
        assert len(workflow.convergence) == 10

    def test_method(self, workflow, monkeypatch):
        """Test if the method keyword could be passed to the AutoMBAR estimator from convergence."""
        monkeypatch.setattr(workflow, "convergence", None)
        workflow.check_convergence(2, estimator="MBAR", method="adaptive")
        assert len(workflow.convergence) == 2

    def test_nosample_u_nk(self, workflow, monkeypatch):
        """test if the convergence routine would use the unsampled data
        when the data has not been subsampled."""
        monkeypatch.setattr(workflow, "u_nk_sample_list", None)
        monkeypatch.setattr(workflow, "convergence", None)
        workflow.check_convergence(10)
        assert len(workflow.convergence) == 10

    def test_no_u_nk_for_check_convergence(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "u_nk_list", None)
        monkeypatch.setattr(workflow, "u_nk_sample_list", None)
        with pytest.raises(ValueError):
            workflow.check_convergence(10, estimator="MBAR")

    def test_no_dHdl_for_check_convergence(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "dHdl_list", None)
        monkeypatch.setattr(workflow, "dHdl_sample_list", None)
        with pytest.raises(ValueError):
            workflow.check_convergence(10, estimator="TI")

    def test_bar_convergence(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "convergence", None)
        workflow.check_convergence(10, estimator="BAR")
        assert len(workflow.convergence) == 10

    def test_convergence_invalid_estimator(self, workflow):
        with pytest.raises(ValueError):
            workflow.check_convergence(10, estimator="aaa")

    def test_ti_convergence(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "convergence", None)
        workflow.check_convergence(10, estimator="TI")
        assert len(workflow.convergence) == 10

    def test_preserve_unit(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "convergence", None)
        monkeypatch.setattr(workflow, "units", "kcal/mol")
        workflow.check_convergence(2, estimator="TI")
        assert np.allclose(workflow.convergence["data_fraction"], [0.5, 1.0])

    def test_unprocessed_dhdl(self, workflow, monkeypatch):
        monkeypatch.setattr(workflow, "dHdl_sample_list", None)
        monkeypatch.setattr(workflow, "convergence", None)
        workflow.check_convergence(10, estimator="TI")
        assert len(workflow.convergence) == 10


class TestUnit:
    def test_no_update_units(self, workflow):
        assert workflow.update_units() is None


class Test_automatic_amber:
    """Test the full automatic workflow for load_ABFE from alchemtest.amber for
    three stage transformation."""

    @staticmethod
    @pytest.fixture(scope="class")
    def workflow(tmp_path_factory):
        outdir = tmp_path_factory.mktemp("out")
        dir, _ = os.path.split(
            os.path.dirname(load_bace_example()["data"]["complex"]["vdw"][0])
        )

        workflow = ABFE(
            units="kcal/mol",
            software="AMBER",
            dir=dir,
            prefix="ti",
            suffix="bz2",
            T=298.0,
            outdirectory=str(outdir),
        )
        workflow.read()
        workflow.estimate(estimators="TI")
        return workflow

    def test_summary(self, workflow):
        """Test if if the summary is right."""
        summary = workflow.generate_result()
        assert np.isclose(summary["TI"]["Stages"]["TOTAL"], 1.40405980473, 0.1)


class Test_automatic_parquet:
    """Test the full automatic workflow for load_ABFE from parquet data."""

    @staticmethod
    @pytest.fixture(scope="class")
    def workflow(tmp_path_factory):
        outdir = tmp_path_factory.mktemp("out")
        for i, u_nk in enumerate(load_bace_example()["data"]["complex"]["vdw"]):
            df = alchemlyb.parsing.amber.extract_u_nk(u_nk, T=298)
            df.to_parquet(path=f"{outdir}/u_nk_{i}.parquet", index=True)

        workflow = ABFE(
            units="kcal/mol",
            software="PARQUET",
            dir=str(outdir),
            prefix="u_nk_",
            suffix="parquet",
            T=298.0,
            outdirectory=str(outdir),
        )
        workflow.read()
        workflow.estimate(estimators="BAR")
        return workflow

    def test_summary(self, workflow):
        """Test if if the summary is right."""
        summary = workflow.generate_result()
        assert np.isclose(summary["BAR"]["Stages"]["TOTAL"], 1.40405980473, 0.1)
