"""Amber parser tests.

"""
from alchemlyb.parsing.amber import extract_dHdl
from alchemlyb.parsing.amber import file_validation
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
