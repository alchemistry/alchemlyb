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


@pytest.mark.parametrize("u_nk_list", ["gmx_benzene_VDW_u_nk", "gmx_ABFE_complex_n_uk"])
def test_extract_dHdl(u_nk_list, request, tmp_path):
    u_nk = request.getfixturevalue(u_nk_list)[0]
    u_nk.to_parquet(path=str(tmp_path / "u_nk.parquet"), index=True)
    new_u_nk = extract_u_nk(str(tmp_path / "u_nk.parquet"), T=300)
    assert (new_u_nk.columns == u_nk.columns).all()
    assert (new_u_nk.index == u_nk.index).all()
