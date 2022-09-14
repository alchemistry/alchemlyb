"""Amber parser tests.

"""
import pytest
from numpy import isclose

from alchemlyb.parsing.amber import extract_dHdl
from alchemlyb.parsing.amber import extract_u_nk
from alchemlyb.parsing.amber import file_validation
from alchemlyb.parsing.amber import any_none
from alchemtest.amber import load_simplesolvated
from alchemtest.amber import load_invalidfiles
from alchemtest.amber import load_bace_example
from alchemtest.amber import load_bace_improper


@pytest.fixture(
    name="invalid_file", scope="module",
    params=list(load_invalidfiles()['data'][0]))
def fixture_invalid_file(request):
    """
    Set 'invalid_file' for the subsequent tests, 
    returning each file in load_invalidfiles
    """
    return request.param


@pytest.fixture(name="single_u_nk", scope="module")
def fixture_single_u_nk():
    """return a single file to check u_unk parsing"""
    return load_bace_example().data['solvated']['vdw'][0]


@pytest.fixture(name="single_dHdl", scope="module")
def fixture_single_dHdl():
    """return a single file to check u_unk parsing"""
    return load_simplesolvated().data['charge'][0]


def test_any_none():
    """Test the any None function to ensure if the None value will be caught"""
    None_value_result = [150000, None, None, None, None, None, None, None, None]
    assert any_none(None_value_result) is True


def test_invalidfiles(invalid_file):
    """
    Test the file validation function to ensure the 
    function returning False if the file is invalid
    """
    assert file_validation(invalid_file) is False


def test_dHdl_invalidfiles(invalid_file):
    """Test if we catch possible parsing errors in invalid files"""
    assert extract_dHdl(invalid_file, T=300) is None


def test_dHdl_time_reading(single_dHdl, first_time=22.0, last_time=1020.0):
    """Test if time information is read correctly when extracting dHdl"""
    dHdl = extract_dHdl(single_dHdl, T=300)
    assert isclose(dHdl.index.values[0][0], first_time)
    assert isclose(dHdl.index.values[-1][0], last_time)


def test_u_nk_time_reading(single_u_nk, first_time=22.0, last_time=1020.0):
    """Test if time information is read correctly when extracting u_nk"""
    u_nk = extract_u_nk(single_u_nk, T=300)
    assert isclose(u_nk.index.values[0][0], first_time)
    assert isclose(u_nk.index.values[-1][0], last_time)


@pytest.mark.parametrize("filename",
                         [filename
                          for leg in load_simplesolvated()['data'].values()
                          for filename in leg])
def test_dHdl(filename,
              names=('time', 'lambdas'),
              shape=(500, 1)):
    """Test that dHdl has the correct form when extracted from files."""
    dHdl = extract_dHdl(filename, T=300)

    assert dHdl.index.names == names
    assert dHdl.shape == shape


@pytest.mark.parametrize("mbar_filename",
                         [mbar_filename
                          for leg in load_bace_example()['data']['complex'].values()
                          for mbar_filename in leg])
def test_u_nk(mbar_filename,
              names=('time', 'lambdas')):
    """Test the u_nk has the correct form when extracted from files"""
    u_nk = extract_u_nk(mbar_filename, T=300)

    assert u_nk.index.names == names


@pytest.mark.parametrize("improper_filename",
                         [improper_filename
                          for leg in load_bace_improper()['data'].values()
                          for improper_filename in leg])
def test_u_nk_improper(improper_filename,
                       names=('time', 'lambdas')):
    """Test the u_nk has the correct form when extracted from files"""
    try:
        u_nk = extract_u_nk(improper_filename, T=300)
        assert u_nk.index.names == names
    except Exception:
        assert '0.5626' in improper_filename
