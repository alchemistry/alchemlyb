"""Amber parser tests."""

import bz2
import logging

import pandas as pd
import pytest
from alchemtest.amber import load_bace_example
from alchemtest.amber import load_bace_improper
from alchemtest.amber import load_simplesolvated
from alchemtest.amber import load_testfiles
from numpy.testing import assert_allclose

from alchemlyb.parsing.amber import extract
from alchemlyb.parsing.amber import extract_dHdl
from alchemlyb.parsing.amber import extract_u_nk


##################################################################################
################ Check the parser behaviour with problematic files
##################################################################################


@pytest.fixture(name="testfiles", scope="module")
def fixture_testfiles():
    """Returns the testfiles data dictionary"""
    bunch = load_testfiles()
    return bunch["data"]


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        _ = extract("not_a_valid_file_name", T=298.0)


def test_no_dHdl_data_points(caplog, testfiles):
    """Test if we deal with a file without dHdl data points"""
    filename = testfiles["no_dHdl_data_points"][0]
    with caplog.at_level(logging.INFO):
        _ = extract(str(filename), T=298.0)
    assert "does not contain any dV/dl data" in caplog.text


def test_None_in_mbar(caplog, testfiles):
    """Test if we deal with an incorrect MBAR section"""
    filename = testfiles["none_in_mbar"][0]
    with pytest.raises(ValueError, match="strange parsing the following MBAR section"):
        with caplog.at_level(logging.ERROR):
            _ = extract(str(filename), T=298.0)
    assert "strange parsing the following MBAR section" in caplog.text


def test_unfinished_run(caplog, testfiles):
    """Test if we give a warning if we are parsing an unfinished run"""
    filename = testfiles["not_finished_run"][0]
    with caplog.at_level(logging.WARNING):
        _ = extract(str(filename), T=298.0)
    assert "is a prematurely terminated run" in caplog.text


def test_no_atomic_section(caplog, testfiles):
    """Test if raise an exception if there is no ATOMIC section"""
    filename = testfiles["no_atomic_section"][0]
    with pytest.raises(ValueError, match='no "ATOMIC" section found'):
        with caplog.at_level(logging.ERROR):
            _ = extract(str(filename), T=298.0)
    assert 'No "ATOMIC" section found' in caplog.text


def test_no_control_data(caplog, testfiles):
    """Test if we raise an exception if there is no CONTROL section"""
    filename = testfiles["no_control_data"][0]
    with pytest.raises(ValueError, match='no "CONTROL DATA" section found'):
        with caplog.at_level(logging.ERROR):
            _ = extract(str(filename), T=298.0)
    assert 'No "CONTROL DATA" section found' in caplog.text


def test_no_free_energy_info(caplog, testfiles):
    """Test if we raise an exception if there is no free energy section"""
    filename = testfiles["no_free_energy_info"][0]
    with pytest.raises(ValueError, match="no free energy section found"):
        with caplog.at_level(logging.ERROR):
            _ = extract(str(filename), T=298.0)
    assert "No free energy section found" in caplog.text


def test_no_useful_data(caplog, testfiles):
    """Test if we raise an exception if there is no data"""
    filename = testfiles["no_useful_data"][0]
    with pytest.raises(ValueError, match="does not contain any data"):
        with caplog.at_level(logging.ERROR):
            _ = extract(str(filename), T=298.0)
    assert "does not contain any data" in caplog.text


def test_no_temp0_set(caplog, testfiles):
    """Test if we raise an exception if there is no temp0 set"""
    filename = testfiles["no_temp0_set"][0]
    with pytest.raises(ValueError, match='no valid "temp0" record found'):
        with caplog.at_level(logging.ERROR):
            _ = extract(str(filename), T=298.0)
    assert 'No valid "temp0" record found' in caplog.text


def test_no_results_section(caplog, testfiles):
    """Test if we raise an exception if there is no RESULTS section"""
    filename = testfiles["no_results_section"][0]
    with pytest.raises(ValueError, match='no "RESULTS" section found'):
        with caplog.at_level(logging.ERROR):
            _ = extract(str(filename), T=298.0)
    assert 'No "RESULTS" section found' in caplog.text


def test_long_and_wrong_number_MBAR(caplog, testfiles):
    """
    Test if we have a high number of MBAR states, and also a different
    number of MBAR states than expected
    """
    filename = testfiles["high_and_wrong_number_of_mbar_windows"][0]
    with pytest.raises(ValueError, match="the number of lambda windows read"):
        with caplog.at_level(logging.ERROR):
            _ = extract_u_nk(str(filename), T=300.0)
    assert "the number of lambda windows read" in caplog.text


def test_no_starting_time(caplog, testfiles):
    """Test if raise an exception if the starting time is not read"""
    filename = testfiles["no_starting_simulation_time"][0]
    with pytest.raises(ValueError, match="No starting simulation time in file"):
        with caplog.at_level(logging.ERROR):
            _ = extract(str(filename), T=298.0)
    assert "No starting simulation time in file" in caplog.text


def test_parse_without_spaces_around_equal(testfiles):
    """
    Test if the regex is able to extract values where the are no
    spaces around the equal sign
    """
    filename = testfiles["no_spaces_around_equal"][0]
    df_dict = extract(str(filename), T=298.0)
    assert isinstance(df_dict["dHdl"], pd.DataFrame)


##################################################################################
################ Check the parser behaviour with standard single files
##################################################################################


@pytest.fixture(name="single_u_nk", scope="module")
def fixture_single_u_nk():
    """return a single file to check u_unk parsing"""
    return load_bace_example().data["complex"]["vdw"][0]


@pytest.fixture(name="single_dHdl", scope="module")
def fixture_single_dHdl():
    """return a single file to check dHdl parsing"""
    return load_simplesolvated().data["charge"][0]


def test_dHdl_time_reading(single_dHdl):
    """Test if time information is read correctly when extracting dHdl"""
    dHdl = extract_dHdl(single_dHdl, T=298.0)
    assert_allclose(dHdl.index.values[0][0], 22.0)
    assert_allclose(dHdl.index.values[-1][0], 1020.0)


def test_u_nk_time_reading(single_u_nk):
    """Test if time information is read correctly when extracting u_nk"""
    u_nk = extract_u_nk(single_u_nk, T=298.0)
    assert_allclose(u_nk.index.values[0][0], 22.0)
    assert_allclose(u_nk.index.values[-1][0], 1020.0)


def test_extract_with_both_data(single_u_nk):
    """Test that dHdl and u_nk have the correct form when
    extracted from files with the single "extract" funcion."""
    df_dict = extract(single_u_nk, T=298.0)
    assert df_dict["dHdl"].index.names == ("time", "lambdas")
    assert df_dict["dHdl"].shape == (500, 1)
    assert df_dict["u_nk"].index.names == ("time", "lambdas")


def test_extract_with_only_dhdl_data(single_dHdl):
    """Test that parsing with the extract function a file
    with just dHdl gives the correct results"""
    df_dict = extract(single_dHdl, T=298.0)
    assert df_dict["dHdl"].index.names == ("time", "lambdas")
    assert df_dict["dHdl"].shape == (500, 1)
    assert df_dict["u_nk"] is None


def test_wrong_T_should_raise_warning(single_dHdl, T=300.0):
    """
    Test if calling extract with differnt T from what's
    read from the AMBER file gives a warning
    """
    with pytest.raises(
        ValueError, match="is different from the temperature passed as parameter"
    ):
        _ = extract(single_dHdl, T=T)


def test_u_nk_time_reading_bar_intervall(single_u_nk, tmp_path):
    """Test if time information is read correctly when extracting u_nk"""
    with bz2.open(single_u_nk, "rt") as bz2_file:
        content = bz2_file.read()

    with open(tmp_path / "amber.log", "w") as text_file:
        text_file.write(
            content.replace("bar_intervall =     1000", "bar_intervall =   10")
        )

    u_nk = extract_u_nk(tmp_path / "amber.log", T=298.0)
    # was 22 before
    assert_allclose(u_nk.index.values[0][0], 20.02)


###################################################################
################ Check the behaviour on proper datasets
###################################################################


@pytest.mark.parametrize(
    "filename",
    [filename for leg in load_simplesolvated()["data"].values() for filename in leg],
)
def test_dHdl(filename, names=("time", "lambdas"), shape=(500, 1)):
    """Test that dHdl has the correct form when extracted from files."""
    dHdl = extract_dHdl(filename, T=298.0)

    assert dHdl.index.names == names
    assert dHdl.shape == shape


@pytest.mark.parametrize(
    "mbar_filename",
    [
        mbar_filename
        for leg in load_bace_example()["data"]["complex"].values()
        for mbar_filename in leg
    ],
)
def test_u_nk(mbar_filename, names=("time", "lambdas")):
    """Test the u_nk has the correct form when extracted from files"""
    u_nk = extract_u_nk(mbar_filename, T=298.0)

    assert u_nk.index.names == names


@pytest.mark.parametrize(
    "improper_filename",
    [
        improper_filename
        for leg in load_bace_improper()["data"].values()
        for improper_filename in leg
    ],
)
def test_u_nk_improper(improper_filename, names=("time", "lambdas")):
    """Test the u_nk has the correct form when extracted from files"""
    try:
        u_nk = extract_u_nk(improper_filename, T=298.0)
        assert u_nk.index.names == names
    except Exception:
        assert "0.5626" in improper_filename


def test_concatenated_amber_u_nk(tmp_path):
    with bz2.open(load_bace_example()["data"]["complex"]["decharge"][0], "rt") as file:
        content = file.read()

    with open(tmp_path / "amber.out", "w") as f:
        f.write(content)
        f.write("\n")
        f.write(content)

    with pytest.raises(
        ValueError,
        match="MBAR Energy detected after the TIMINGS section.",
    ):
        extract(tmp_path / "amber.out", 298)


def test_concatenated_amber_dhdl(tmp_path):
    with bz2.open(load_bace_example()["data"]["complex"]["decharge"][0], "rt") as file:
        content = file.read().replace("MBAR Energy analysis", "")

    with open(tmp_path / "amber.out", "w") as f:
        f.write(content)
        f.write("\n")
        f.write(content)

    with pytest.raises(
        ValueError,
        match="TI Energy detected after the TIMINGS section.",
    ):
        extract(tmp_path / "amber.out", 298)
