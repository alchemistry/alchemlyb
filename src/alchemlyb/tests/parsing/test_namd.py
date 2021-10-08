"""NAMD parser tests.

"""
from os.path import basename
from re import search
import bz2
import pytest

from alchemlyb.parsing.namd import extract_u_nk
from alchemtest.namd import load_tyr2ala
from alchemtest.namd import load_idws
from alchemtest.namd import load_restarted
from alchemtest.namd import load_restarted_reversed

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


@pytest.fixture(params=[load_restarted, load_restarted_reversed])
def restarted_dataset(request):
    return request.param()


def _corrupt_fepout(fepout_in, params, tmp_path):
    """Corrupts specific lines in a fepout file according to each line's prefix,
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
    The name of a temporary file which pytest will unlink.
    """

    fepout_out = tmp_path / basename(fepout_in)
    with bz2.open(fepout_out, 'wt') as f_out:
        with bz2.open(fepout_in, 'rt') as f_in:
            for line in f_in:
                for prefix, func in params:
                    if line.startswith(prefix):
                        line = ' '.join(func(line.split())) + '\n'
                f_out.write(line)
    return str(fepout_out)


@pytest.fixture
def restarted_dataset_inconsistent(restarted_dataset, tmp_path):
    """Returns intentionally messed up dataset where lambda1 and lambda2 at start and end of
    a window are different."""

    filenames = sorted(restarted_dataset['data']['both'])

    changed = False
    def func_free_line(l):
        nonlocal changed
        if float(l[7]) >= 0.7 and float(l[7]) < 0.9:
            l[7] = str(float(l[7]) + 0.0001)
            changed = True
        return l

    for i in range(len(filenames)):
        restarted_dataset['data']['both'][i] = \
            _corrupt_fepout(filenames[i], [('#Free', func_free_line)], tmp_path)
        # Only actually modify one window so we don't trigger the wrong exception
        if changed is True:
            break

    return restarted_dataset


@pytest.fixture
def restarted_dataset_idws_without_lambda_idws(restarted_dataset, tmp_path):
    """Returns intentionally messed up dataset where the first window has IDWS data
    but no lambda_idws.
    """

    # First window won't have any IDWS data so we just drop all its files and fudge the lambdas
    # in the next window to include 0.0 or 1.0 (as appropriate) so we still have a nominally complete calculation
    
    filenames = [x for x in sorted(restarted_dataset['data']['both']) if search('000[a-z]?.fepout', x) is None]

    def func_new_line(l):
        if float(l[LAMBDA1_IDX_NEW]) > 0.5: # 1->0 (reversed) calculation
            l[LAMBDA1_IDX_NEW] == '1.0'
        else: # regular 0->1 calculation
            l[LAMBDA1_IDX_NEW] = '0.0'
        # Drop the lambda_idws
        return l[:9]
    
    def func_free_line(l):
        if float(l[LAMBDA1_IDX_FREE]) > 0.5: # 1->0 (reversed) calculation
            l[LAMBDA1_IDX_FREE] == '1.0'
        else: # regular 0->1 calculation
            l[LAMBDA1_IDX_FREE] = '0.0'
        return l
        
    filenames[0] = _corrupt_fepout(filenames[0], [('#NEW', func_new_line), ('#Free', func_free_line)], tmp_path)
    restarted_dataset['data']['both'] = filenames
    return restarted_dataset


@pytest.fixture
def restarted_dataset_toomany_lambda2(restarted_dataset, tmp_path):
    """Returns intentionally messed up dataset, where there are too many lambda2 values for a
    given lambda1."""

    filenames = sorted(restarted_dataset['data']['both'])

    # For the same l1 and lidws we retain old lambda2 values thus ensuring a collision
    # Also, don't make a window where lambda1 >= lambda2 because this will trigger the
    # "direction changed" exception instead
    def func_new_line(l):
        if float(l[LAMBDA2_IDX_NEW]) <= 0.2:
            return l
        l[LAMBDA1_IDX_NEW] = '0.2'
        if len(l) > 9 and l[9] == 'LAMBDA_IDWS':
            l[LAMBDA_IDWS_IDX_NEW] = '0.1'
        return l

    def func_free_line(l):
        if float(l[LAMBDA2_IDX_FREE]) <= 0.2:
            return l
        l[LAMBDA1_IDX_FREE] = '0.2'
        return l

    for i in range(len(filenames)):
        restarted_dataset['data']['both'][i] = \
            _corrupt_fepout(filenames[i], [('#NEW', func_new_line), ('#Free', func_free_line)], tmp_path)

    return restarted_dataset


@pytest.fixture
def restarted_dataset_toomany_lambda_idws(restarted_dataset, tmp_path):
    """Returns intentionally messed up dataset, where there are too many lambda2 values for a
    given lambda1."""

    filenames = sorted(restarted_dataset['data']['both'])

    # For the same lambda1 and lambda2 we retain the first set of lambda1/lambda2 values
    # and replicate them across all windows thus ensuring that there will be more than
    # one lambda_idws value for a given lambda1 and lambda2
    this_lambda1, this_lambda2 = None, None

    def func_new_line(l):
        nonlocal this_lambda1, this_lambda2
        
        if this_lambda1 is None:
            this_lambda1, this_lambda2 = l[LAMBDA1_IDX_NEW], l[LAMBDA2_IDX_NEW]
        # Ensure that changing these lambda values won't cause a reversal in direction and trigger
        # an exception we're not trying to test here
        if len(l) > 9 and float(l[LAMBDA_IDWS_IDX_NEW]) < 0.5:
            l[LAMBDA1_IDX_NEW], l[LAMBDA2_IDX_NEW] = this_lambda1, this_lambda2
        return l

    def func_free_line(l):
        l[LAMBDA1_IDX_FREE], l[LAMBDA2_IDX_FREE] = this_lambda1, this_lambda2
        return l

    for i in range(len(filenames)):
        restarted_dataset['data']['both'][i] = _corrupt_fepout(filenames[i], [('#NEW', func_new_line)], tmp_path)

    return restarted_dataset


@pytest.fixture
def restarted_dataset_direction_changed(restarted_dataset, tmp_path):
    """Returns intentionally messed up dataset, with one window where the lambda values are reversed."""

    filenames = sorted(restarted_dataset['data']['both'])

    def func_new_line(l):
        l[6], l[8], l[10] = l[10], l[8], l[6]
        return l

    def func_free_line(l):
        l[7], l[8] = l[8], l[7]
        return l
    
    # Reverse the direction of lambdas for this window
    idx_to_corrupt = filenames.index(sorted(filenames)[-3])
    fname1 = _corrupt_fepout(filenames[idx_to_corrupt], [('#NEW', func_new_line), ('#Free', func_free_line)], tmp_path)
    restarted_dataset['data']['both'][idx_to_corrupt] = fname1
    return restarted_dataset


def test_u_nk_restarted():
    """Test that u_nk has the correct form when extracted from an IDWS
    FEP run that includes terminations and restarts.
    """
    filenames = load_restarted()['data']['both']
    u_nk = extract_u_nk(filenames, T=300)
    
    assert u_nk.index.names == ['time', 'fep-lambda']
    assert u_nk.shape == (30061, 11)


def test_u_nk_restarted_reversed():
    filenames = load_restarted_reversed()['data']['both']
    u_nk = extract_u_nk(filenames, T=300)
    
    assert u_nk.index.names == ['time', 'fep-lambda']
    assert u_nk.shape == (30170, 11)


def test_u_nk_restarted_direction_changed(restarted_dataset_direction_changed):
    """Test that when lambda values change direction within a dataset, parsing throws an error."""

    with pytest.raises(ValueError, match='Lambda values change direction'):
        u_nk = extract_u_nk(restarted_dataset_direction_changed['data']['both'], T=300)


def test_u_nk_restarted_idws_without_lambda_idws(restarted_dataset_idws_without_lambda_idws):
    """Test that when the first window has IDWS data but no lambda_idws, parsing throws an error.
    
    In this situation, the lambda_idws cannot be inferred, because there's no previous lambda
    value available.
    """

    with pytest.raises(ValueError, match='IDWS data present in first window but lambda_idws not included'):
        u_nk = extract_u_nk(restarted_dataset_idws_without_lambda_idws['data']['both'], T=300)


def test_u_nk_restarted_inconsistent(restarted_dataset_inconsistent):
    """Test that when lambda values are inconsistent between start and end of a single window,
    parsing throws an error.
    """

    with pytest.raises(ValueError, match='Inconsistent lambda values within the same window'):
        u_nk = extract_u_nk(restarted_dataset_inconsistent['data']['both'], T=300)


def test_u_nk_restarted_toomany_lambda_idws(restarted_dataset_toomany_lambda_idws):
    """Test that when there is more than one lambda_idws for a given lambda1, parsing throws an error."""

    with pytest.raises(ValueError, match='More than one lambda_idws value for a particular lambda1'):
        u_nk = extract_u_nk(restarted_dataset_toomany_lambda_idws['data']['both'], T=300)


def test_u_nk_restarted_toomany_lambda2(restarted_dataset_toomany_lambda2):
    """Test that when there is more than one lambda2 for a given lambda1, parsing throws an error."""

    with pytest.raises(ValueError, match='More than one lambda2 value for a particular lambda1'):
        u_nk = extract_u_nk(restarted_dataset_toomany_lambda2['data']['both'], T=300)
