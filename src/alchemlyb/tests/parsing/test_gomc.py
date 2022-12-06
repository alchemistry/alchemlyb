"""GOMC parser tests.

"""

from alchemtest.gomc import load_benzene

from alchemlyb.parsing.gomc import extract_dHdl, extract_u_nk, extract


def test_dHdl():
    """Test that dHdl has the correct form when extracted from files."""
    dataset = load_benzene()

    for filename in dataset["data"]:
        dHdl = extract_dHdl(filename, T=298)

        assert dHdl.index.names == ["time", "Coulomb-lambda", "VDW-lambda"]
        assert dHdl.shape == (1000, 2)


def test_u_nk():
    """Test that u_nk has the correct form when extracted from files."""
    dataset = load_benzene()

    for filename in dataset["data"]:
        u_nk = extract_u_nk(filename, T=298)

        assert u_nk.index.names == ["time", "Coulomb-lambda", "VDW-lambda"]
        assert u_nk.shape == (1000, 23)


def test_extract():
    """Test that u_nk and dHdl have the correct form when extracted from files."""
    dataset = load_benzene()

    df_dict = extract(dataset["data"][0], T=298)

    assert df_dict["u_nk"].index.names == ["time", "Coulomb-lambda", "VDW-lambda"]
    assert df_dict["u_nk"].shape == (1000, 23)
    assert df_dict["dHdl"].index.names == ["time", "Coulomb-lambda", "VDW-lambda"]
    assert df_dict["dHdl"].shape == (1000, 2)
