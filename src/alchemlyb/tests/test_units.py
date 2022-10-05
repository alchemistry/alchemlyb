import pytest
import pandas as pd

import alchemlyb
from alchemlyb import pass_attrs
from alchemtest.gmx import load_benzene
from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk
from alchemlyb.postprocessors.units import to_kT
from alchemlyb.preprocessing import (dhdl2series, u_nk2series,
                                     decorrelate_u_nk, decorrelate_dhdl,
                                     slicing, statistical_inefficiency,
                                     equilibrium_detection)

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

def test_concat_empty():
    '''Test if empty raise the right error.'''
    with pytest.raises(ValueError):
        alchemlyb.concat([])

def test_setT():
    '''Test setting temperature.'''
    df = pd.DataFrame(data={'col1': [1, 2]})
    df.attrs = {'temperature': 300, 'energy_unit': 'kT'}
    new = to_kT(df, 310)
    assert new.attrs['temperature'] == 310

class Test_Conversion():
    '''Test the preprocessing module.'''
    @staticmethod
    @pytest.fixture(scope='class')
    def dhdl():
        dataset = load_benzene()
        dhdl = extract_dHdl(dataset['data']['Coulomb'][0], 310)
        return dhdl

    def test_kt2kt_number(self, dhdl):
        new_dhdl = to_kT(dhdl)
        assert 12.9 == pytest.approx(new_dhdl.iloc[0, 0], 0.1)

    def test_kt2kt_unit(self, dhdl):
        new_dhdl = to_kT(dhdl)
        assert new_dhdl.attrs['energy_unit'] == 'kT'

    def test_kj2kt_unit(self, dhdl):
        dhdl.attrs['energy_unit'] = 'kJ/mol'
        new_dhdl = to_kT(dhdl)
        assert new_dhdl.attrs['energy_unit'] == 'kT'

    def test_kj2kt_number(self, dhdl):
        dhdl.attrs['energy_unit'] = 'kJ/mol'
        new_dhdl = to_kT(dhdl)
        assert 5.0 == pytest.approx(new_dhdl.iloc[0, 0], 0.1)

    def test_kcal2kt_unit(self, dhdl):
        dhdl.attrs['energy_unit'] = 'kcal/mol'
        new_dhdl = to_kT(dhdl)
        assert new_dhdl.attrs['energy_unit'] == 'kT'

    def test_kcal2kt_number(self, dhdl):
        dhdl.attrs['energy_unit'] = 'kcal/mol'
        new_dhdl = to_kT(dhdl)
        assert 21.0 == pytest.approx(new_dhdl.iloc[0, 0], 0.1)

    def test_unknown2kt(self, dhdl):
        dhdl.attrs['energy_unit'] = 'ddd'
        with pytest.raises(ValueError):
            to_kT(dhdl)

def test_pd_concat():
    '''Test if concat will preserve the metadata.
    When this test is being made, the pd.concat will discard the attrs of
    the input dataframe. However, this should get fixed in the future.
    pandas-dev/pandas#28283
    <https://github.com/pandas-dev/pandas/issues/28283>
    '''
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df1 = pd.DataFrame(data=d)
    df1.attrs = {1: 1}
    df2 = pd.DataFrame(data=d)
    df2.attrs = {1: 1}
    df = pd.concat([df1, df2])
    assert df.attrs == {1: 1}

def test_pass_attrs():
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df1 = pd.DataFrame(data=d)
    df1.attrs = {1: 1}
    df2 = pd.DataFrame(data=d)
    df2.attrs = {1: 1}

    @pass_attrs
    def concat(df1, df2):
        return pd.concat([df1, df2])
    assert concat(df1, df2).attrs == {1: 1}

def test_pd_slice():
    '''Test if slicing will preserve the metadata.'''
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    df.attrs = {1: 1}
    assert df[::2].attrs == {1: 1}

class TestRetainUnit():
    '''This test tests if the functions that should retain the unit would actually
     retain the units.'''
    @staticmethod
    @pytest.fixture(scope='class')
    def dhdl():
        dataset = load_benzene()
        dhdl = extract_dHdl(dataset['data']['Coulomb'][0], 310)
        return dhdl

    @staticmethod
    @pytest.fixture(scope='class')
    def u_nk():
        dataset = load_benzene()
        u_nk = extract_u_nk(dataset['data']['Coulomb'][0], 310)
        return u_nk

    @pytest.mark.parametrize('func,fixture_in',
                             [(dhdl2series, 'dhdl'),
                              (u_nk2series, 'u_nk'),
                              (decorrelate_u_nk, 'u_nk'),
                              (decorrelate_dhdl, 'dhdl'),
                              (slicing, 'dhdl'),
                              (statistical_inefficiency, 'dhdl'),
                              (equilibrium_detection, 'dhdl')])
    def test_function(self, func, fixture_in, request):
        result = func(request.getfixturevalue(fixture_in))
        assert result.attrs['energy_unit'] is not None
