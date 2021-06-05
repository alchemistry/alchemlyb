import pytest

from alchemtest.gmx import load_benzene
from alchemlyb.parsing.gmx import extract_dHdl
from alchemlyb.postprocessors.units import to_kT

def test_noT():
    '''Test no temperature error'''
    dataset = load_benzene()
    dhdl = extract_dHdl(dataset['data']['Coulomb'][0], 310)
    dhdl.attrs.pop('temperature', None)
    with pytest.raises(TypeError):
        to_kT(dhdl)

def test_nounit():
    '''Test no unit error'''
    dataset = load_benzene()
    dhdl = extract_dHdl(dataset['data']['Coulomb'][0], 310)
    dhdl.attrs.pop('energy_unit', None)
    with pytest.raises(TypeError):
        to_kT(dhdl)
        