import pytest
import pandas as pd

import alchemlyb
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

def test_concat():
    '''Test if different attrs could will give rise to error.'''
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df1 = pd.DataFrame(data=d)
    df1.attrs = {1: 1}
    df2 = pd.DataFrame(data=d)
    df2.attrs = {1: 2}
    with pytest.raises(ValueError):
        alchemlyb.concat([df1, df2])

def test_setT():
    '''Test setting temperature.'''
    df = pd.DataFrame(data={'col1': [1, 2]})
    df.attrs = {'temperature': 300, 'energy_unit': 'kT'}
    new = to_kT(df, 310)
    assert new.attrs['temperature'] == 310
