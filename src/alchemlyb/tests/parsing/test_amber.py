"""Amber parser tests.

"""
from alchemlyb.parsing.amber import extract_dHdl
from alchemlyb.parsing.amber import file_validation
from alchemlyb.parsing.amber import any_none
from alchemtest.amber import load_simplesolvated
from alchemtest.amber import load_invalidfiles


def test_dHdl():
    """Test that dHdl has the correct form when extracted from files.

    """
    dataset = load_simplesolvated()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            dHdl = extract_dHdl(filename,)

            assert dHdl.index.names == ['time', 'lambdas']
            assert dHdl.shape == (500, 1)

def test_invalidfiles():
    """Test the file validation function to ensure the function returning False if the file is invalid
    
    """
    invalid_files = load_invalidfiles()
    
    for invalid_file_list in invalid_files['data']:
        for invalid_file in invalid_file_list:
            assert file_validation(invalid_file) == False

def test_any_none():
    """Test the any None function to ensure if the None value will be caught
    """
    None_value_result = [150000, None, None, None, None, None, None, None, None]
    assert any_none(None_value_result) == True
