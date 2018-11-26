"""Gromacs parser tests.

"""

from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk
from alchemtest.gmx import load_benzene
from alchemtest.gmx import load_expanded_ensemble_case_1, load_expanded_ensemble_case_2, load_expanded_ensemble_case_3
from alchemtest.gmx import load_water_particle_with_total_energy
from alchemtest.gmx import load_water_particle_with_potential_energy
from alchemtest.gmx import load_water_particle_without_energy
from numpy.testing import assert_almost_equal


def test_dHdl():
    """Test that dHdl has the correct form when extracted from files.

    """
    dataset = load_benzene()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            dHdl = extract_dHdl(filename, T=300)

            assert dHdl.index.names == ['time', 'fep-lambda']
            assert dHdl.shape == (4001, 1)

def test_u_nk():
    """Test that u_nk has the correct form when extracted from files.

    """
    dataset = load_benzene()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            u_nk = extract_u_nk(filename, T=300)

            assert u_nk.index.names == ['time', 'fep-lambda']
            if leg == 'Coulomb':
                assert u_nk.shape == (4001, 5)
            elif leg == 'VDW':
                assert u_nk.shape == (4001, 16)

def test_u_nk_case1():
    """Test that u_nk has the correct form when extracted from expanded ensemble files (case 1).

    """
    dataset = load_expanded_ensemble_case_1()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            u_nk = extract_u_nk(filename, T=300)

            assert u_nk.index.names == ['time', 'fep-lambda', 'coul-lambda', 'vdw-lambda', 'restraint-lambda']

            assert u_nk.shape == (50001, 28)

def test_dHdl_case1():
    """Test that dHdl has the correct form when extracted from expanded ensemble files (case 1).

    """
    dataset = load_expanded_ensemble_case_1()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            dHdl = extract_dHdl(filename, T=300)

            assert dHdl.index.names == ['time', 'fep-lambda', 'coul-lambda', 'vdw-lambda', 'restraint-lambda']
            assert dHdl.shape == (50001, 4)

def test_u_nk_case2():
    """Test that u_nk has the correct form when extracted from expanded ensemble files (case 2).

    """
    dataset = load_expanded_ensemble_case_2()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            u_nk = extract_u_nk(filename, T=300)

            assert u_nk.index.names == ['time', 'fep-lambda', 'coul-lambda', 'vdw-lambda', 'restraint-lambda']

            assert u_nk.shape == (25001, 28)

def test_u_nk_case3():
    """Test that u_nk has the correct form when extracted from REX files (case 3).

    """
    dataset = load_expanded_ensemble_case_3()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            u_nk = extract_u_nk(filename, T=300)

            assert u_nk.index.names == ['time', 'fep-lambda', 'coul-lambda', 'vdw-lambda', 'restraint-lambda']

            assert u_nk.shape == (2500, 28)

def test_dHdl_case3():
    """Test that dHdl has the correct form when extracted from REX files (case 3).

    """
    dataset = load_expanded_ensemble_case_3()

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            dHdl = extract_dHdl(filename, T=300)

            assert dHdl.index.names == ['time', 'fep-lambda', 'coul-lambda', 'vdw-lambda', 'restraint-lambda']
            assert dHdl.shape == (2500, 4)

def test_u_nk_with_total_energy():
    """Test that the reduced potential is calculated correctly when the total energy is given.

    """

    # Load dataset
    dataset = load_water_particle_with_total_energy()

    # Check if the sum of values on the diagonal has the correct value
    assert_almost_equal(_diag_sum(dataset), 47611377946.58586, decimal=4)

    # Check one specific value in the dataframe
    assert_almost_equal(
        extract_u_nk(dataset['data']['AllStates'][0], T=300).iloc[0][0],
        -11211.578357345974,
        decimal=6
    )

def test_u_nk_with_potential_energy():
    """Test that the reduced potential is calculated correctly when the potential energy is given.

    """

    # Load dataset
    dataset = load_water_particle_with_potential_energy()

    # Check if the sum of values on the diagonal has the correct value
    assert_almost_equal(_diag_sum(dataset), 16674041445589.646, decimal=2)

    # Check one specific value in the dataframe
    assert_almost_equal(
        extract_u_nk(dataset['data']['AllStates'][0], T=300).iloc[0][0],
        -15656.558227621246,
        decimal=6
    )


def test_u_nk_without_energy():
    """Test that the reduced potential is calculated correctly when no energy is given.

    """

    # Load dataset
    dataset = load_water_particle_without_energy()

    # Check if the sum of values on the diagonal has the correct value
    assert_almost_equal(_diag_sum(dataset), 20572988148877.555, decimal=2)

    # Check one specific value in the dataframe
    assert_almost_equal(
        extract_u_nk(dataset['data']['AllStates'][0], T=300).iloc[0][0],
        0.0,
        decimal=6
    )


def _diag_sum(dataset):
    """Calculate the sum of diagonal elements (i, i)

    """

    # Initialize the sum variable
    ds = 0.0

    for leg in dataset['data']:
        for filename in dataset['data'][leg]:
            u_nk = extract_u_nk(filename, T=300)

            # Calculate the sum of diagonal elements:
            for i in range(len(dataset['data'][leg])):
                ds += u_nk.iloc[i][i]

    return ds
