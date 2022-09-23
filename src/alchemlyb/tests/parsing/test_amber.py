"""Amber parser tests.

"""
import pytest
from numpy.testing import assert_allclose

from alchemlyb.parsing.amber import extract_dHdl
from alchemlyb.parsing.amber import extract_u_nk
from alchemlyb.parsing.amber import file_validation
from alchemlyb.parsing.amber import extract
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
    return load_bace_example().data['complex']['vdw'][0]


@pytest.fixture(name="single_dHdl", scope="module")
def fixture_single_dHdl():
    """return a single file to check u_unk parsing"""
    return load_simplesolvated().data['charge'][0]


def test_invalidfiles(invalid_file):
    """
    Test the file validation function to ensure the 
    function returning False if the file is invalid
    """
    assert file_validation(invalid_file) is False


def test_dHdl_invalidfiles(invalid_file):
    """Test if we catch possible parsing errors in invalid files"""
    assert extract_dHdl(invalid_file, T=298.0) is None


def test_dHdl_time_reading(single_dHdl, first_time=22.0, last_time=1020.0):
    """Test if time information is read correctly when extracting dHdl"""
    dHdl = extract_dHdl(single_dHdl, T=298.0)
    assert_allclose(dHdl.index.values[0][0], first_time)
    assert_allclose(dHdl.index.values[-1][0], last_time)


def test_u_nk_time_reading(single_u_nk, first_time=22.0, last_time=1020.0):
    """Test if time information is read correctly when extracting u_nk"""
    u_nk = extract_u_nk(single_u_nk, T=298.0)
    assert_allclose(u_nk.index.values[0][0], first_time)
    assert_allclose(u_nk.index.values[-1][0], last_time)


def test_extract_with_both_data(
    single_u_nk,
    mbar_names=('time', 'lambdas'),
    dhdl_names=('time', 'lambdas'),
    dhdl_shape=(500, 1)):
    """Test that dHdl and u_nk have the correct form when 
    extracted from files with the extract funcion."""
    df_dict = extract(single_u_nk, T=298.0)
    assert df_dict['dHdl'].index.names == dhdl_names
    assert df_dict['dHdl'].shape == dhdl_shape
    assert df_dict['u_nk'].index.names == mbar_names


def test_extract_with_only_dhdl_data(
    single_dHdl,
    dhdl_names=('time', 'lambdas'),
    dhdl_shape=(500, 1)):
    """Test that parsing with the extract function a file
     with just dHdl gives the correct results"""
    df_dict = extract(single_dHdl, T=298.0)
    assert df_dict['dHdl'].index.names == dhdl_names
    assert df_dict['dHdl'].shape == dhdl_shape
    assert df_dict['u_nk'] is None


def test_wrong_T_should_raise_warning_in_extract_dHdl(single_dHdl, T=300.0):
    """
    Test if calling extract_dHdl with differnt T from what's
    read from the AMBER file gives a warning
    """
    with pytest.raises(
        ValueError,
        match="is different from the temperature passed as parameter"):
        _ = extract_dHdl(single_dHdl, T=T)


def test_wrong_T_should_raise_warning_in_extract_u_nk(single_u_nk, T=300.0):
    """
    Test if calling extract_u_nk with differnt T from what's
    read from the AMBER file gives a warning
    """
    with pytest.raises(
        ValueError,
        match="is different from the temperature passed as parameter"):
        _ = extract_u_nk(single_u_nk, T=T)


@pytest.mark.parametrize("filename",
                         [filename
                          for leg in load_simplesolvated()['data'].values()
                          for filename in leg])
def test_dHdl(filename,
              names=('time', 'lambdas'),
              shape=(500, 1)):
    """Test that dHdl has the correct form when extracted from files."""
    dHdl = extract_dHdl(filename, T=298.0)

    assert dHdl.index.names == names
    assert dHdl.shape == shape


@pytest.mark.parametrize("mbar_filename",
                         [mbar_filename
                          for leg in load_bace_example()['data']['complex'].values()
                          for mbar_filename in leg])
def test_u_nk(mbar_filename,
              names=('time', 'lambdas')):
    """Test the u_nk has the correct form when extracted from files"""
    u_nk = extract_u_nk(mbar_filename, T=298.0)

    assert u_nk.index.names == names


@pytest.mark.parametrize("improper_filename",
                         [improper_filename
                          for leg in load_bace_improper()['data'].values()
                          for improper_filename in leg])
def test_u_nk_improper(improper_filename,
                       names=('time', 'lambdas')):
    """Test the u_nk has the correct form when extracted from files"""
    try:
        u_nk = extract_u_nk(improper_filename, T=298.0)
        assert u_nk.index.names == names
    except Exception:
        assert '0.5626' in improper_filename
