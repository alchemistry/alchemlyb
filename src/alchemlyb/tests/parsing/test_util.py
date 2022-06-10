import io
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

@pytest.mark.parametrize('compression', ['bz2', 'gz'])
def test_file_roundtrip(compression, tmp_path):
    """Test that roundtripping write/read to a file works with `anyopen`

    """

    data = "my momma told me to pick the very best one and you are not it"

    filepath = tmp_path / f'testfile.txt.{compression}'
    with anyopen(filepath, mode='w') as f:
        f.write(data)

    with anyopen(filepath, 'r') as f:
        data_out = f.read()

    assert data_out == data

@pytest.mark.parametrize('compression', ['bz2', 'gz'])
def test_file_roundtrip(compression):
    """Test that roundtripping write/read to a file works with `anyopen`

    """

    data = "my momma told me to pick the very best one and you are not it"

    with io.BytesIO() as stream:

        # write to stream
        with anyopen(stream, mode='w', compression=compression) as f:
            f.write(data)

        # start at the beginning
        stream.seek(0)

        # read from stream
        with anyopen(stream, 'r', compression=compression) as f:
            data_out = f.read()

        assert data_out == data
