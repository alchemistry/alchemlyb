"""Gromacs parser tests.

"""

import bz2

import pytest
from alchemtest.gmx import load_benzene
from alchemtest.gmx import (
    load_expanded_ensemble_case_1,
    load_expanded_ensemble_case_2,
    load_expanded_ensemble_case_3,
)
from alchemtest.gmx import load_water_particle_with_potential_energy
from alchemtest.gmx import load_water_particle_with_total_energy
from alchemtest.gmx import load_water_particle_without_energy
from numpy.testing import assert_almost_equal

from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk, extract


def test_dHdl():
    """Test that dHdl has the correct form when extracted from files."""
    dataset = load_benzene()

    for leg in dataset["data"]:
        for filename in dataset["data"][leg]:
            dHdl = extract_dHdl(filename, T=300)

            assert dHdl.index.names == ["time", "fep-lambda"]
            assert dHdl.shape == (4001, 1)


def test_u_nk():
    """Test that u_nk has the correct form when extracted from files."""
    dataset = load_benzene()

    for leg in dataset["data"]:
        for filename in dataset["data"][leg]:
            u_nk = extract_u_nk(filename, T=300)

            assert u_nk.index.names == ["time", "fep-lambda"]
            if leg == "Coulomb":
                assert u_nk.shape == (4001, 5)
            elif leg == "VDW":
                assert u_nk.shape == (4001, 16)


def test_u_nk_case1():
    """Test that u_nk has the correct form when extracted from expanded ensemble files (case 1)."""
    dataset = load_expanded_ensemble_case_1()

    for leg in dataset["data"]:
        for filename in dataset["data"][leg]:
            u_nk = extract_u_nk(filename, T=300, filter=False)

            assert u_nk.index.names == [
                "time",
                "fep-lambda",
                "coul-lambda",
                "vdw-lambda",
                "restraint-lambda",
            ]

            assert u_nk.shape == (50001, 28)


def test_dHdl_case1():
    """Test that dHdl has the correct form when extracted from expanded ensemble files (case 1)."""
    dataset = load_expanded_ensemble_case_1()

    for leg in dataset["data"]:
        for filename in dataset["data"][leg]:
            dHdl = extract_dHdl(filename, T=300, filter=False)

            assert dHdl.index.names == [
                "time",
                "fep-lambda",
                "coul-lambda",
                "vdw-lambda",
                "restraint-lambda",
            ]
            assert dHdl.shape == (50001, 4)


def test_u_nk_case2():
    """Test that u_nk has the correct form when extracted from expanded ensemble files (case 2)."""
    dataset = load_expanded_ensemble_case_2()

    for leg in dataset["data"]:
        for filename in dataset["data"][leg]:
            u_nk = extract_u_nk(filename, T=300, filter=False)

            assert u_nk.index.names == [
                "time",
                "fep-lambda",
                "coul-lambda",
                "vdw-lambda",
                "restraint-lambda",
            ]

            assert u_nk.shape == (25001, 28)


def test_u_nk_case3():
    """Test that u_nk has the correct form when extracted from REX files (case 3)."""
    dataset = load_expanded_ensemble_case_3()

    for leg in dataset["data"]:
        for filename in dataset["data"][leg]:
            u_nk = extract_u_nk(filename, T=300, filter=False)

            assert u_nk.index.names == [
                "time",
                "fep-lambda",
                "coul-lambda",
                "vdw-lambda",
                "restraint-lambda",
            ]

            assert u_nk.shape == (2500, 28)


def test_dHdl_case3():
    """Test that dHdl has the correct form when extracted from REX files (case 3)."""
    dataset = load_expanded_ensemble_case_3()

    for leg in dataset["data"]:
        for filename in dataset["data"][leg]:
            dHdl = extract_dHdl(filename, T=300, filter=False)

            assert dHdl.index.names == [
                "time",
                "fep-lambda",
                "coul-lambda",
                "vdw-lambda",
                "restraint-lambda",
            ]
            assert dHdl.shape == (2500, 4)


def test_u_nk_with_total_energy():
    """Test that the reduced potential is calculated correctly when the total energy is given."""

    # Load dataset
    dataset = load_water_particle_with_total_energy()

    # Check if the sum of values on the diagonal has the correct value
    assert_almost_equal(_diag_sum(dataset), 47611374980.34574, decimal=4)

    # Check one specific value in the dataframe
    assert_almost_equal(
        extract_u_nk(dataset["data"]["AllStates"][0], T=300)
        .loc[0][(0.0, 0.0)]
        .values[0],
        -11211.577658852531,
        decimal=6,
    )


def test_u_nk_with_potential_energy():
    """Test that the reduced potential is calculated correctly when the potential energy is given."""

    # Load dataset
    dataset = load_water_particle_with_potential_energy()

    # Check if the sum of values on the diagonal has the correct value
    assert_almost_equal(_diag_sum(dataset), 16674040406778.867, decimal=2)

    # Check one specific value in the dataframe
    assert_almost_equal(
        extract_u_nk(dataset["data"]["AllStates"][0], T=300)
        .loc[0][(0.0, 0.0)]
        .values[0],
        -15656.557252200757,
        decimal=6,
    )


def test_u_nk_without_energy():
    """Test that the reduced potential is calculated correctly when no energy is given."""

    # Load dataset
    dataset = load_water_particle_without_energy()

    # Check if the sum of values on the diagonal has the correct value
    assert_almost_equal(_diag_sum(dataset), 20572986867158.184, decimal=2)

    # Check one specific value in the dataframe
    assert_almost_equal(
        extract_u_nk(dataset["data"]["AllStates"][0], T=300)
        .loc[0][(0.0, 0.0)]
        .values[0],
        0.0,
        decimal=6,
    )


def _diag_sum(dataset):
    """Calculate the sum of diagonal elements (i, i)"""

    # Initialize the sum variable
    ds = 0.0

    for leg in dataset["data"]:
        for filename in dataset["data"][leg]:
            u_nk = extract_u_nk(filename, T=300)

            # Calculate the sum of diagonal elements:
            for i, lambda_ in enumerate(u_nk.columns):
                # 18.6 is the time step
                ds += u_nk.loc[i * 186 / 10][lambda_].values[0]

    return ds


def test_extract_u_nk_unit():
    """Test if extract_u_nk assign the attr correctly"""
    dataset = load_benzene()
    u_nk = extract_u_nk(dataset["data"]["Coulomb"][0], 310)
    assert u_nk.attrs["temperature"] == 310
    assert u_nk.attrs["energy_unit"] == "kT"


def test_extract_dHdl_unit():
    """Test if extract_u_nk assign the attr correctly"""
    dataset = load_benzene()
    dhdl = extract_dHdl(dataset["data"]["Coulomb"][0], 310)
    assert dhdl.attrs["temperature"] == 310
    assert dhdl.attrs["energy_unit"] == "kT"


def test_calling_extract():
    """Test if the extract function is working"""
    dataset = load_benzene()
    df_dict = extract(dataset["data"]["Coulomb"][0], 310)
    assert df_dict["dHdl"].attrs["temperature"] == 310
    assert df_dict["dHdl"].attrs["energy_unit"] == "kT"
    assert df_dict["u_nk"].attrs["temperature"] == 310
    assert df_dict["u_nk"].attrs["energy_unit"] == "kT"


class TestRobustGMX:
    """Test dropping the row that is wrong in different way"""

    @staticmethod
    @pytest.fixture(scope="class")
    def data():
        dhdl = extract_dHdl(load_benzene()["data"]["Coulomb"][0], 310)
        with bz2.open(load_benzene()["data"]["Coulomb"][0], "rt") as bz_file:
            text = bz_file.read()
        return text, len(dhdl)

    def test_sanity(self, data, tmp_path):
        """Test if the test routine is working."""
        text, length = data
        new_text = tmp_path / "text.xvg"
        new_text.write_text(text)
        dhdl = extract_dHdl(new_text, 310)
        assert len(dhdl) == length

    def test_truncated_row(self, data, tmp_path):
        """Test the case where the last row has been truncated."""
        text, length = data
        new_text = tmp_path / "text.xvg"
        new_text.write_text(text + "40010.0 27.0\n")
        dhdl = extract_dHdl(new_text, 310, filter=True)
        assert len(dhdl) == length

    def test_truncated_number(self, data, tmp_path):
        """Test the case where the last row has been truncated and a - has
        been left."""
        text, length = data
        new_text = tmp_path / "text.xvg"
        new_text.write_text(text + "40010.0 27.0 -\n")
        dhdl = extract_dHdl(new_text, 310, filter=True)
        assert len(dhdl) == length

    def test_weirdnumber(self, data, tmp_path):
        """Test the case where the last number has been appended a weird
        number."""
        text, length = data
        new_text = tmp_path / "text.xvg"
        # Note the 27.040010.0 which is the sum of 27.0 and 40010.0
        new_text.write_text(
            text + "40010.0 27.040010.0 27.0 0.0 6.7 13.5 20.2 27.0 0.7 27.0 0.0 6.7 "
            "13.5 20.2 27.0 0.7\n"
        )
        dhdl = extract_dHdl(new_text, 310, filter=True)
        assert len(dhdl) == length

    def test_too_many_cols(self, data, tmp_path):
        """Test the case where the row has too many columns."""
        text, length = data
        new_text = tmp_path / "text.xvg"
        new_text.write_text(
            text
            + "40010.0 27.0 0.0 6.7 13.5 20.2 27.0 0.7 27.0 0.0 6.7 13.5 20.2 27.0 0.7\n"
        )
        dhdl = extract_dHdl(new_text, 310, filter=True)
        assert len(dhdl) == length
