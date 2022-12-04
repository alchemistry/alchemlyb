import pytest
from alchemtest.amber import load_bace_example
from alchemtest.gmx import (
    load_benzene,
    load_expanded_ensemble_case_1,
    load_expanded_ensemble_case_2,
    load_expanded_ensemble_case_3,
    load_water_particle_with_total_energy,
    load_water_particle_with_potential_energy,
    load_water_particle_without_energy,
    load_ABFE,
)
from alchemtest.gomc import load_benzene as gomc_load_benzene
from alchemtest.namd import (
    load_tyr2ala,
    load_idws,
    load_restarted,
    load_restarted_reversed,
)

from alchemlyb.parsing import gmx, amber, gomc, namd


@pytest.fixture
def gmx_benzene():
    dataset = load_benzene()
    return dataset["data"]


@pytest.fixture
def gmx_benzene_Coulomb_dHdl(gmx_benzene):
    return [gmx.extract_dHdl(file, T=300) for file in gmx_benzene["Coulomb"]]


@pytest.fixture
def gmx_benzene_Coulomb_u_nk(gmx_benzene):
    return [gmx.extract_u_nk(file, T=300) for file in gmx_benzene["Coulomb"]]


@pytest.fixture
def gmx_benzene_VDW_u_nk(gmx_benzene):
    return [gmx.extract_u_nk(file, T=300) for file in gmx_benzene["VDW"]]


@pytest.fixture
def gmx_ABFE():
    dataset = load_ABFE()
    return dataset["data"]


@pytest.fixture
def gmx_ABFE_complex_n_uk(gmx_ABFE):
    return [gmx.extract_u_nk(file, T=300) for file in gmx_ABFE["complex"]]


@pytest.fixture
def gmx_ABFE_complex_dHdl(gmx_ABFE):
    return [gmx.extract_dHdl(file, T=300) for file in gmx_ABFE["complex"]]


@pytest.fixture
def gmx_expanded_ensemble_case_1():
    dataset = load_expanded_ensemble_case_1()

    return [
        gmx.extract_u_nk(filename, T=300, filter=False)
        for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def gmx_expanded_ensemble_case_2():
    dataset = load_expanded_ensemble_case_2()

    return [
        gmx.extract_u_nk(filename, T=300, filter=False)
        for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def gmx_expanded_ensemble_case_3():
    dataset = load_expanded_ensemble_case_3()

    return [
        gmx.extract_u_nk(filename, T=300, filter=False)
        for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def gmx_water_particle_with_total_energy():
    dataset = load_water_particle_with_total_energy()

    return [
        gmx.extract_u_nk(filename, T=300) for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def gmx_water_particle_with_potential_energy():
    dataset = load_water_particle_with_potential_energy()

    return [
        gmx.extract_u_nk(filename, T=300) for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def gmx_water_particle_without_energy():
    dataset = load_water_particle_without_energy()

    return [
        gmx.extract_u_nk(filename, T=300) for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def amber_bace_example_complex_vdw():
    dataset = load_bace_example()

    return [
        amber.extract_u_nk(filename, T=298.0)
        for filename in dataset["data"]["complex"]["vdw"]
    ]


@pytest.fixture
def gomc_benzene_u_nk():
    dataset = gomc_load_benzene()

    return [gomc.extract_u_nk(filename, T=298) for filename in dataset["data"]]


@pytest.fixture
def namd_tyr2ala():
    dataset = load_tyr2ala()
    u_nk1 = namd.extract_u_nk(dataset["data"]["forward"][0], T=300)
    u_nk2 = namd.extract_u_nk(dataset["data"]["backward"][0], T=300)

    # combine dataframes of fwd and rev directions
    u_nk1[u_nk1.isna()] = u_nk2
    u_nk = u_nk1.sort_index(level=u_nk1.index.names[1:])

    return u_nk


@pytest.fixture
def namd_idws():
    dataset = load_idws()
    u_nk = namd.extract_u_nk(dataset["data"]["forward"], T=300)

    return u_nk


@pytest.fixture
def namd_idws_restarted():
    dataset = load_restarted()
    u_nk = namd.extract_u_nk(dataset["data"]["both"], T=300)

    return u_nk


@pytest.fixture
def namd_idws_restarted_reversed():
    dataset = load_restarted_reversed()
    u_nk = namd.extract_u_nk(dataset["data"]["both"], T=300)

    return u_nk
