import pytest

from alchemtest.gmx import load_expanded_ensemble_case_1
from alchemlyb.parsing.util import anyopen


def test_gzip():
    """Test that gzip reads .gz files in the correct (text) mode.

    """
    dataset = load_expanded_ensemble_case_1()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            with anyopen(filename, 'r') as f:
                assert type(f.readline()) is str

def test_gzip_stream():
    """Test that `anyopen` reads streams with specified compression.

    """
    dataset = load_expanded_ensemble_case_1()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            with open(filename, 'rb') as f:
                with anyopen(f, mode='r', compression='gz') as f_uc:
                    assert type(f_uc.readline()) is str

def test_gzip_stream_wrong():
    """Test that `anyopen` gives failure for attempting to decompress gzip
    stream with bz2.

    """
    dataset = load_expanded_ensemble_case_1()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            with open(filename, 'rb') as f:
                with anyopen(f, mode='r', compression='bz2') as f_uc:
                    with pytest.raises(OSError, match='Invalid data stream'):
                        assert type(f_uc.readline()) is str

def test_gzip_stream_wrong_no_compression():
    """Test that `anyopen` gives passthrough when no compression specified on a
    stream.

    """
    dataset = load_expanded_ensemble_case_1()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            with open(filename, 'rb') as f:
                with anyopen(f, mode='r') as f_uc:
                    assert type(f_uc.readline()) is bytes
