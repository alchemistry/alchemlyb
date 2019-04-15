"""Amber parser tests.

"""
import pytest

from six.moves import zip

from alchemlyb.parsing.amber import extract_dHdl
from alchemlyb.parsing.amber import extract_u_nk
from alchemlyb.parsing.amber import file_validation
from alchemlyb.parsing.amber import any_none
from alchemtest.amber import load_simplesolvated
from alchemtest.amber import load_invalidfiles
from alchemtest.amber import load_bace_example
from alchemtest.amber import load_bace_improper


@pytest.fixture(scope="module",
                params=[filename for filename in load_invalidfiles()['data'][0]])
def invalid_file(request):
    return request.param


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

    except:
        assert '0.5626' in improper_filename


def test_invalidfiles(invalid_file):
    """Test the file validation function to ensure the function returning False if the file is invalid
    """
    assert file_validation(invalid_file) == False


def test_dHdl_invalidfiles(invalid_file):
    assert extract_dHdl(invalid_file, T=300) is None


def test_any_none():
    """Test the any None function to ensure if the None value will be caught"""
    None_value_result = [150000, None, None, None, None, None, None, None, None]
    assert any_none(None_value_result) == True
