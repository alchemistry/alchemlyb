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

# Indices of lambda values in the following line in NAMD fepout files:
# #NEW FEP WINDOW: LAMBDA SET TO 0.6 LAMBDA2 0.7 LAMBDA_IDWS 0.5
LAMBDA1_IDX_NEW = 6
LAMBDA2_IDX_NEW = 8
LAMBDA_IDWS_IDX_NEW = 10
# Indices of lambda values in the following type of line in NAMD fepout files:
# #Free energy change for lambda window [ 0.6 0.7 ] is 0.12345 ; net change until now is 0.12345
LAMBDA1_IDX_FREE = 7
LAMBDA2_IDX_FREE = 8

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


def _corrupt_fepout(fepout_in, params):
    """Corrups specific lines in a fepout file according to each line's prefix,
    using caller-supplied functions.

    Parameters
    ----------
    fepout_in: str
        Path to fepout file to be modified. This file will not be overwritten.
    
    params: list of tuples
        For each tuple, the first element must be a str that will be passed to
        startswith() to identify the line(s) to modify (e.g. "#NEW"). The
        second element must be a function that accepts a list of strs which is
        the output of running split() on the identified line and returns
        a modified list of tokens that will be reassembled into the line to be
        output.

    Returns
    -------
    The name of a temporary file which the caller must unlink.
    """

    fh, fepout_out = mkstemp(suffix='.fepout.bz2', prefix='restarted_')
    close(fh)
    with bz2.open(fepout_out, 'wt') as f_out:
        with bz2.open(fepout_in, 'rt') as f_in:
            for line in f_in:
                for prefix, func in params:
                    if line.startswith(prefix):
                        line = ' '.join(func(line.split())) + '\n'
                f_out.write(line)
    return fepout_out


@pytest.fixture
def restarted_dataset_inconsistent():
    """Returns intentionally messed up dataset."""
    # We explicitly call load_restarted() again because we are modifying the
    # singleton dataset object and don't want to break other tests
    dataset = load_restarted()
    filenames = dataset['data']['both']

    changed = False
    def func_free_line(l):
        if float(l[7]) >= 0.7 and float(l[7]) < 0.9:
            l[7] = str(float(l[7]) + 0.0001)
            changed = True
        return l

    temp_fnames = []
    for i in range(len(filenames)):
        fname = _corrupt_fepout(filenames[i], [('#Free', func_free_line)])
        dataset['data']['both'][i] = fname
        temp_fnames.append(fname)
        # Only actually modify one window so we don't trigger the wrong exception
        if changed is True:
            break

    yield dataset
    for fname in temp_fnames:
        unlink(fname)


@pytest.fixture
def restarted_dataset_toomany_lambda2():
    """Returns intentionally messed up dataset, where there are too many lambda2 values for a
    given lambda1."""
    dataset = load_restarted()
    filenames = dataset['data']['both']

    # For the same l1 and lidws we retain old lambda2 values thus ensuring a collision
    def func_new_line(l):
        l[LAMBDA1_IDX_NEW] = '0.2'
        if len(l) > 9 and l[9] == 'LAMBDA_IDWS':
            l[LAMBDA_IDWS_IDX_NEW] = '0.1'
        return l

    def func_free_line(l):
        l[LAMBDA1_IDX_FREE] = '0.2'
        return l

    temp_fnames = []
    for i in range(len(filenames)):
        fname = _corrupt_fepout(filenames[i], [('#NEW', func_new_line), ('#Free', func_free_line)])
        dataset['data']['both'][i] = fname
        temp_fnames.append(fname)

    yield dataset
    for fname in temp_fnames:
        unlink(fname)


@pytest.fixture
def restarted_dataset_toomany_lambda_idws():
    """Returns intentionally messed up dataset, where there are too many lambda2 values for a
    given lambda1."""
    dataset = load_restarted()
    filenames = dataset['data']['both']

    # For the same lambda1 and lambda2 we retain old lambda_idws values thus ensuring a collision
    changed = False
    def func_new_line(l):
        # Ensure that changing these lamda values won't cause a reversal in direction and trigger
        # an exception we're not trying to test here
        if len(l) > 9 and float(l[LAMBDA_IDWS_IDX_NEW]) < 0.5:
            l[LAMBDA1_IDX_NEW], l[LAMBDA2_IDX_NEW] = '0.5', '0.6'
            changed = True
        return l

    def func_free_line(l):
        if changed:
            l[LAMBDA1_IDX_FREE], l[LAMBDA2_IDX_FREE] = '0.5', '0.6'
        return l

    temp_fnames = []
    for i in range(len(filenames)):
        fname = _corrupt_fepout(filenames[i], [('#NEW', func_new_line)])
        dataset['data']['both'][i] = fname
        temp_fnames.append(fname)

    yield dataset
    for fname in temp_fnames:
        unlink(fname)


@pytest.fixture
def restarted_dataset_direction_changed():
    """Returns intentionally messed up dataset, with one window where the lambda values are reversed."""
    dataset = load_restarted()
    filenames = dataset['data']['both']

    def func_new_line(l):
        l[6], l[8], l[10] = l[10], l[8], l[6]
        return l

    def func_free_line(l):
        l[7], l[8] = l[8], l[7]
        return l
    
    # Reverse the direction of lambdas for this window
    idx_to_corrupt = filenames.index(sorted(filenames)[-3])
    fname1 = _corrupt_fepout(filenames[idx_to_corrupt], [('#NEW', func_new_line), ('#Free', func_free_line)])
    dataset['data']['both'][idx_to_corrupt] = fname1
    yield dataset

    unlink(fname1)


def test_u_nk_restarted(restarted_dataset, restarted_dataset_direction_changed,
    restarted_dataset_toomany_lambda_idws, restarted_dataset_toomany_lambda2,
    restarted_dataset_inconsistent):
    """Test that u_nk has the correct form when extracted from an IDWS
    FEP run that includes a termination and restart, and that the parser throws
    exceptions when it encounters various forms of corrupted fepout files.
    """

    filenames = restarted_dataset['data']['both']
    u_nk = extract_u_nk(filenames, T=300)

    assert u_nk.index.names == ['time', 'fep-lambda']
    assert u_nk.shape == (30061, 11)

    with pytest.raises(ValueError, match='Inconsistent lambda values within the same window'):
        u_nk = extract_u_nk(restarted_dataset_inconsistent['data']['both'], T=300)

    with pytest.raises(ValueError, match='Lambda values change direction'):
        u_nk = extract_u_nk(restarted_dataset_direction_changed['data']['both'], T=300)

    with pytest.raises(ValueError, match='More than one lambda_idws value for a particular lambda1'):
        u_nk = extract_u_nk(restarted_dataset_toomany_lambda_idws['data']['both'], T=300)

    with pytest.raises(ValueError, match='More than one lambda2 value for a particular lambda1'):
        u_nk = extract_u_nk(restarted_dataset_toomany_lambda2['data']['both'], T=300)

