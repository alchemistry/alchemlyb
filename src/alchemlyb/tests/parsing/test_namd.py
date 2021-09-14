"""NAMD parser tests.

"""
from os import unlink, close
from tempfile import mkstemp
import bz2
import pytest

from alchemlyb.parsing.namd import extract_u_nk
from alchemtest.namd import load_tyr2ala
from alchemtest.namd import load_idws
from alchemtest.namd import load_restarted


@pytest.fixture(scope="module")
def dataset():
    return load_tyr2ala()

@pytest.mark.parametrize("direction,shape",
                         [('forward', (21021, 21)),
                          ('backward', (21021, 21)),
                          ])
def test_u_nk(dataset, direction, shape):
    """Test that u_nk has the correct form when extracted from files.
    """
    for filename in dataset['data'][direction]:
        u_nk = extract_u_nk(filename, T=300)

        assert u_nk.index.names == ['time', 'fep-lambda']
        assert u_nk.shape == shape

def test_u_nk_idws():
    """Test that u_nk has the correct form when extracted from files.
    """

    filenames = load_idws()['data']['forward']
    u_nk = extract_u_nk(filenames, T=300)

    assert u_nk.index.names == ['time', 'fep-lambda']
    assert u_nk.shape == (29252, 11)


@pytest.fixture
def restarted_dataset():
    return load_restarted()


@pytest.fixture
def restarted_dataset_free_line_broken():
    """Returns intentionally messed up dataset."""
    fh, fname = mkstemp(suffix='.fepout.bz2', prefix='restarted_free_line_broken_')
    # We explicitly call load_restarted() again because we are modifying the
    # singleton dataset object and don't want to break other tests
    restarted_dataset = load_restarted()
    filenames = restarted_dataset['data']['both']
    close(fh)

    # Corrupt a lambda value
    INDEX_TO_CORRUPT = -3
    with bz2.open(fname, 'wt') as f_out:
        with bz2.open(filenames[INDEX_TO_CORRUPT], 'rt') as f_in:
            for line in f_in:
                l = line.split()
                if l[0] == '#Free':
                    l[7] = '0.777'
                    line = ' '.join(l)
                f_out.write(line)
    restarted_dataset['data']['both'][INDEX_TO_CORRUPT] = fname
    yield restarted_dataset

    unlink(fname)


@pytest.fixture
def restarted_dataset_lambda2_broken():
    """Returns intentionally messed up dataset, where there are too many lambda2 values for a given lambda1."""
    fh, fname = mkstemp(suffix='.fepout.bz2', prefix='restarted_lambda2_broken_')
    # We explicitly call load_restarted() again because we are modifying the
    # singleton dataset object and don't want to break other tests
    restarted_dataset = load_restarted()
    filenames = restarted_dataset['data']['both']
    close(fh)

    # TODO: 
    # Put lambda2 = 1.0 for some .fepout that is not the last one

    # Corrupt a lambda value
    INDEX_TO_CORRUPT = -3
    with bz2.open(fname, 'wt') as f_out:
        with bz2.open(filenames[INDEX_TO_CORRUPT], 'rt') as f_in:
            for line in f_in:
                l = line.split()
                if l[0] == '#Free':
                    l[7] = '1.0'
                    line = ' '.join(l) + '\n'
                f_out.write(line)
    restarted_dataset['data']['both'][INDEX_TO_CORRUPT] = fname
    yield restarted_dataset

    unlink(fname)


@pytest.fixture
def restarted_dataset_idws_broken():
    """Returns intentionally messed up dataset, where there are too mamy lambda_idws values for a given lambda1."""
    fh, fname = mkstemp(suffix='.fepout.bz2', prefix='restarted_idws_broken_')
    # We explicitly call load_restarted() again because we are modifying the
    # singleton dataset object and don't want to break other tests
    restarted_dataset = load_restarted()
    filenames = restarted_dataset['data']['both']
    close(fh)

    # Corrupt a lambda value
    INDEX_TO_CORRUPT = -4
    with bz2.open(fname, 'wt') as f_out:
        with bz2.open(filenames[INDEX_TO_CORRUPT], 'rt') as f_in:
            for line in f_in:
                l = line.split()
                if l[0] == '#NEW':
                    l[10] = '0.0'
                    line = ' '.join(l) + '\n'
                f_out.write(line)
    restarted_dataset['data']['both'][INDEX_TO_CORRUPT] = fname
    yield restarted_dataset

    # unlink(fname)


def test_u_nk_restarted(restarted_dataset, restarted_dataset_free_line_broken,
    restarted_dataset_idws_broken):
    """Test that u_nk has the correct form when extracted from an IDWS
    FEP run that includes a termination and restart.
    """

    filenames = restarted_dataset['data']['both']
    u_nk = extract_u_nk(filenames, T=300)

    assert u_nk.index.names == ['time', 'fep-lambda']
    assert u_nk.shape == (30061, 11)

    with pytest.raises(ValueError):
        u_nk = extract_u_nk(restarted_dataset_free_line_broken['data']['both'],
            T=300)

    with pytest.raises(ValueError, match='Inconsistent lambda values'):
        u_nk = extract_u_nk(restarted_dataset_idws_broken['data']['both'],
            T=300)

