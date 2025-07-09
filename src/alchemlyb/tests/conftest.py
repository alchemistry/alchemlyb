"""Storing the fixture to be used for the tests. Note that this file will only contain
fixture that are made directly from parsing the files. Any additional operations like
concat should be done at local level."""

import pytest
from pytest import LogCaptureFixture
from alchemtest.amber import load_bace_example, load_simplesolvated, load_tyk2_example
from alchemtest.gmx import (
    load_benzene,
    load_ethanol,
    load_expanded_ensemble_case_1,
    load_expanded_ensemble_case_2,
    load_expanded_ensemble_case_3,
    load_water_particle_with_total_energy,
    load_water_particle_with_potential_energy,
    load_water_particle_without_energy,
    load_ABFE,
)
import pandas as pd
from alchemtest.gomc import load_benzene as gomc_load_benzene
from alchemtest.namd import (
    load_tyr2ala,
    load_idws,
    load_restarted,
    load_restarted_reversed,
)
from loguru import logger

from alchemlyb.parsing import gmx, amber, gomc, namd


@pytest.fixture
def gmx_benzene() -> dict[str, list[str]]:
    dataset = load_benzene()
    return dataset["data"]  # type: ignore[no-any-return]


@pytest.fixture
def gmx_benzene_Coulomb_dHdl(gmx_benzene: dict[str, list[str]]) -> list[pd.DataFrame]:
    return [gmx.extract_dHdl(file, T=300) for file in gmx_benzene["Coulomb"]]


@pytest.fixture
def gmx_benzene_VDW_dHdl(gmx_benzene: dict[str, list[str]]) -> list[pd.DataFrame]:
    return [gmx.extract_dHdl(file, T=300) for file in gmx_benzene["VDW"]]


@pytest.fixture
def gmx_ethanol() -> dict[str, list[str]]:
    dataset = load_ethanol()
    return dataset["data"]  # type: ignore[no-any-return]


@pytest.fixture
def gmx_ethanol_Coulomb_dHdl(gmx_ethanol: dict[str, list[str]]) -> list[pd.DataFrame]:
    return [gmx.extract_dHdl(file, T=300) for file in gmx_ethanol["Coulomb"]]


@pytest.fixture
def gmx_ethanol_VDW_dHdl(gmx_ethanol: dict[str, list[str]]) -> list[pd.DataFrame]:
    return [gmx.extract_dHdl(file, T=300) for file in gmx_ethanol["VDW"]]


@pytest.fixture
def gmx_benzene_Coulomb_u_nk(gmx_benzene: dict[str, list[str]]) -> list[pd.DataFrame]:
    return [gmx.extract_u_nk(file, T=300) for file in gmx_benzene["Coulomb"]]


@pytest.fixture
def gmx_benzene_VDW_u_nk(gmx_benzene: dict[str, list[str]]) -> list[pd.DataFrame]:
    return [gmx.extract_u_nk(file, T=300) for file in gmx_benzene["VDW"]]


@pytest.fixture
def gmx_ABFE() -> dict[str, list[str]]:
    dataset = load_ABFE()
    return dataset["data"]  # type: ignore[no-any-return]


@pytest.fixture
def gmx_ABFE_complex_u_nk(gmx_ABFE: dict[str, list[str]]) -> list[pd.DataFrame]:
    return [gmx.extract_u_nk(file, T=300) for file in gmx_ABFE["complex"]]


@pytest.fixture
def gmx_ABFE_complex_dHdl(gmx_ABFE: dict[str, list[str]]) -> list[pd.DataFrame]:
    return [gmx.extract_dHdl(file, T=300) for file in gmx_ABFE["complex"]]


@pytest.fixture
def gmx_expanded_ensemble_case_1() -> list[pd.DataFrame]:
    dataset = load_expanded_ensemble_case_1()

    return [
        gmx.extract_u_nk(filename, T=300, filter=False)
        for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def gmx_expanded_ensemble_case_1_dHdl() -> list[pd.DataFrame]:
    dataset = load_expanded_ensemble_case_1()

    return [
        gmx.extract_dHdl(filename, T=300, filter=False)
        for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def gmx_expanded_ensemble_case_2() -> list[pd.DataFrame]:
    dataset = load_expanded_ensemble_case_2()

    return [
        gmx.extract_u_nk(filename, T=300, filter=False)
        for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def gmx_expanded_ensemble_case_2_dHdl() -> list[pd.DataFrame]:
    dataset = load_expanded_ensemble_case_2()

    return [
        gmx.extract_dHdl(filename, T=300, filter=False)
        for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def gmx_expanded_ensemble_case_3() -> list[pd.DataFrame]:
    dataset = load_expanded_ensemble_case_3()

    return [
        gmx.extract_u_nk(filename, T=300, filter=False)
        for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def gmx_expanded_ensemble_case_3_dHdl() -> list[pd.DataFrame]:
    dataset = load_expanded_ensemble_case_3()

    return [
        gmx.extract_dHdl(filename, T=300, filter=False)
        for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def gmx_water_particle_with_total_energy() -> list[pd.DataFrame]:
    dataset = load_water_particle_with_total_energy()

    return [
        gmx.extract_u_nk(filename, T=300) for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def gmx_water_particle_with_total_energy_dHdl() -> list[pd.DataFrame]:
    dataset = load_water_particle_with_total_energy()

    return [
        gmx.extract_dHdl(filename, T=300) for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def gmx_water_particle_with_potential_energy() -> list[pd.DataFrame]:
    dataset = load_water_particle_with_potential_energy()

    return [
        gmx.extract_u_nk(filename, T=300) for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def gmx_water_particle_with_potential_energy_dHdl() -> list[pd.DataFrame]:
    dataset = load_water_particle_with_potential_energy()

    return [
        gmx.extract_dHdl(filename, T=300) for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def gmx_water_particle_without_energy() -> list[pd.DataFrame]:
    dataset = load_water_particle_without_energy()

    return [
        gmx.extract_u_nk(filename, T=300) for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def gmx_water_particle_without_energy_dHdl() -> list[pd.DataFrame]:
    dataset = load_water_particle_without_energy()

    return [
        gmx.extract_dHdl(filename, T=300) for filename in dataset["data"]["AllStates"]
    ]


@pytest.fixture
def amber_simplesolvated() -> dict[str, list[str]]:
    dataset = load_simplesolvated()
    return dataset["data"]  # type: ignore[no-any-return]


@pytest.fixture
def amber_simplesolvated_charge_dHdl(
    amber_simplesolvated: dict[str, list[str]],
) -> list[pd.DataFrame]:
    return [
        amber.extract_dHdl(filename, T=298.0)  # type: ignore[misc]
        for filename in amber_simplesolvated["charge"]
    ]


@pytest.fixture
def amber_simplesolvated_vdw_dHdl(
    amber_simplesolvated: dict[str, list[str]],
) -> list[pd.DataFrame]:
    return [
        amber.extract_dHdl(filename, T=298.0)  # type: ignore[misc]
        for filename in amber_simplesolvated["vdw"]
    ]


@pytest.fixture
def amber_bace_example_complex_vdw() -> list[pd.DataFrame]:
    dataset = load_bace_example()

    return [
        amber.extract_u_nk(filename, T=298.0)  # type: ignore[misc]
        for filename in dataset["data"]["complex"]["vdw"]
    ]


@pytest.fixture
def amber_tyk2_example_complex() -> list[pd.DataFrame]:
    dataset = load_tyk2_example()

    return [
        amber.extract_dHdl(filename, T=300.0)
        for filename in dataset["data"]["complex"]  # type: ignore[misc]
    ]


@pytest.fixture
def gomc_benzene() -> list[str]:
    dataset = gomc_load_benzene()
    return dataset["data"]  # type: ignore[no-any-return]


@pytest.fixture
def gomc_benzene_u_nk(gomc_benzene: list[str]) -> list[pd.DataFrame]:
    return [gomc.extract_u_nk(filename, T=298) for filename in gomc_benzene]


@pytest.fixture
def gomc_benzene_dHdl(gomc_benzene: list[str]) -> list[pd.DataFrame]:
    return [gomc.extract_dHdl(filename, T=298) for filename in gomc_benzene]


@pytest.fixture
def namd_tyr2ala() -> pd.DataFrame:
    dataset = load_tyr2ala()
    u_nk1 = namd.extract_u_nk(dataset["data"]["forward"][0], T=300)
    u_nk2 = namd.extract_u_nk(dataset["data"]["backward"][0], T=300)

    # combine dataframes of fwd and rev directions
    u_nk1[u_nk1.isna()] = u_nk2
    u_nk = u_nk1.sort_index(level=u_nk1.index.names[1:])

    return u_nk  # type: ignore[no-any-return]


@pytest.fixture
def namd_idws() -> pd.DataFrame:
    dataset = load_idws()
    u_nk = namd.extract_u_nk(dataset["data"]["forward"], T=300)

    return u_nk  # type: ignore[no-any-return]


@pytest.fixture
def namd_idws_restarted() -> pd.DataFrame:
    dataset = load_restarted()
    u_nk = namd.extract_u_nk(dataset["data"]["both"], T=300)

    return u_nk  # type: ignore[no-any-return]


@pytest.fixture
def namd_idws_restarted_reversed() -> pd.DataFrame:
    dataset = load_restarted_reversed()
    u_nk = namd.extract_u_nk(dataset["data"]["both"], T=300)

    return u_nk  # type: ignore[no-any-return]


@pytest.fixture
def caplog(caplog: LogCaptureFixture) -> LogCaptureFixture:  # type: ignore[misc]
    handler_id = logger.add(caplog.handler, format="{message}")
    yield caplog
    logger.remove(handler_id)
