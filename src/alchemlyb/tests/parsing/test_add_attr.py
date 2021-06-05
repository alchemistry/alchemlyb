from alchemlyb.parsing.gmx import extract_u_nk, extract_dHdl
from alchemtest.gmx import load_benzene

def test_extract_u_nk():
    '''Test if extract_u_nk assign the attr correctly'''
    dataset = load_benzene()
    u_nk = extract_u_nk(dataset['data']['Coulomb'][0], 310)
    assert u_nk.attrs['temperature'] == 310
    assert u_nk.attrs['energy_unit'] == 'kT'

def test_extract_dHdl():
    '''Test if extract_u_nk assign the attr correctly'''
    dataset = load_benzene()
    dhdl = extract_dHdl(dataset['data']['Coulomb'][0], 310)
    assert dhdl.attrs['temperature'] == 310
    assert dhdl.attrs['energy_unit'] == 'kT'