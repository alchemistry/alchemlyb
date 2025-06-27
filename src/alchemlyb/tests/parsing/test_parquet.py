import pytest

from alchemlyb.parsing.parquet import extract_u_nk, extract_dHdl


@pytest.mark.parametrize(
    "dHdl_list", ["gmx_benzene_Coulomb_dHdl", "gmx_ABFE_complex_dHdl"]
)
def test_extract_dHdl(dHdl_list, request, tmp_path):
    dHdl = request.getfixturevalue(dHdl_list)[0]
    dHdl.to_parquet(path=str(tmp_path / "dhdl.parquet"), index=True)
    new_dHdl = extract_dHdl(str(tmp_path / "dhdl.parquet"), T=300)
    assert (new_dHdl.columns == dHdl.columns).all()
    assert (new_dHdl.index == dHdl.index).all()
    assert new_dHdl.attrs["temperature"] == 300
    assert new_dHdl.attrs["energy_unit"] == "kT"


@pytest.mark.parametrize("u_nk_list", ["gmx_benzene_VDW_u_nk", "gmx_ABFE_complex_u_nk"])
def test_extract_u_nk(u_nk_list, request, tmp_path):
    u_nk = request.getfixturevalue(u_nk_list)[0]
    u_nk.to_parquet(path=str(tmp_path / "u_nk.parquet"), index=True)
    new_u_nk = extract_u_nk(str(tmp_path / "u_nk.parquet"), T=300)
    assert (new_u_nk.columns == u_nk.columns).all()
    assert (new_u_nk.index == u_nk.index).all()
    assert new_u_nk.attrs["temperature"] == 300
    assert new_u_nk.attrs["energy_unit"] == "kT"


@pytest.fixture()
def u_nk(gmx_ABFE_complex_u_nk):
    return gmx_ABFE_complex_u_nk[0]


def test_no_T(u_nk, tmp_path, caplog):
    u_nk.attrs = {}
    u_nk.to_parquet(path=str(tmp_path / "temp.parquet"), index=True)
    extract_u_nk(str(tmp_path / "temp.parquet"), 300)
    assert (
        "Serialise the Dataframe with pandas>=2.1 to preserve the metadata."
        in caplog.text
    )


def test_wrong_T(u_nk, tmp_path, caplog):
    u_nk.to_parquet(path=str(tmp_path / "temp.parquet"), index=True)
    with pytest.raises(ValueError, match="doesn't match the temperature"):
        extract_u_nk(str(tmp_path / "temp.parquet"), 400)


def test_metadata_unchanged(u_nk, tmp_path):
    u_nk.attrs = {"temperature": 400, "energy_unit": "kcal/mol"}
    u_nk.to_parquet(path=str(tmp_path / "temp.parquet"), index=True)
    new_u_nk = extract_u_nk(str(tmp_path / "temp.parquet"), 400)
    assert new_u_nk.attrs["temperature"] == 400
    assert new_u_nk.attrs["energy_unit"] == "kcal/mol"
