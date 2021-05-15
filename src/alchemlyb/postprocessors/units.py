'''Physical and mathematical constants and units.'''
# Taken from scipy.constants since py2 doesn't support it
#: Boltzmann's constant :math:`k_B` in kJ/(mol K); value from `NIST CODATA: k<https://physics.nist.gov/cgi-bin/cuu/Value?k>`_.
Boltzmann_constant = 1.380649e-23
#: Avogadro constant :math:`N_A` in 1/mol; value from `NIST CODATA: k<https://physics.nist.gov/cgi-bin/cuu/Value?na|search_for=avogadro>`_.
Avogadro_constant = 6.02214076e+23
kJ2kcal = 0.239006


def to_kT(df):
    """Convert the unit of a DataFrame to `kT`. The DataFrame need to have
    attribute `temperature` and `energy_unit`.

    Parameters
    ----------
    df : DataFrame
        DataFrame to convert unit.

    Returns
    -------
    DataFrame
        `df` converted.
    """
    assert 'temperature' in df.attrs, 'Attribute temperature not found in the' \
                                      ' input Dataframe.'
    assert 'energy_unit' in df.attrs, 'Attribute energy_unit not found in the' \
                                      ' input Dataframe.'
    if df.attrs['energy_unit'] == 'kT':
        return df
    elif df.attrs['energy_unit'] == 'kJ/mol':
        new_df = df.copy()
        new_df = new_df / (Boltzmann_constant * df.attrs['temperature'] *
                           Avogadro_constant / 1000)
        new_df.attrs['energy_unit'] = 'kJ/mol'
        return new_df
    elif df.attrs['energy_unit'] == 'kcal/mol':
        new_df = df.copy()
        new_df = new_df / (Boltzmann_constant * df.attrs['temperature'] *
                           Avogadro_constant / 1000 * kJ2kcal)
        new_df.attrs['energy_unit'] = 'kcal/mol'
        return new_df
    else:
        raise NameError('energy_unit {} can only be kT, kJ/mol or ' \
                        'kcal/mol.'.format(df.attrs['energy_unit']))


def to_kcalmol(df):
    """Convert the unit of a DataFrame to kcal/mol. The DataFrame need to have
    attribute `temperature`.

    Parameters
    ----------
    df : DataFrame
        DataFrame to convert unit.

    Returns
    -------
    DataFrame
        `df` converted.
    """
    kt_df = to_kT(df)
    kt_df = kt_df * (Boltzmann_constant * df.attrs['temperature'] *
                           Avogadro_constant / 1000 * kJ2kcal)
    return kt_df

def to_kJmol(df):
    """Convert the unit of a DataFrame to kJ/mol. The DataFrame need to have
    attribute `temperature`.

    Parameters
    ----------
    df : DataFrame
        DataFrame to convert unit.

    Returns
    -------
    DataFrame
        `df` converted.
    """
    kt_df = to_kT(df)
    kt_df = kt_df * (Boltzmann_constant * df.attrs['temperature'] *
                           Avogadro_constant / 1000)
    return kt_df

def pass_attrs(func):
    '''Pass the attributes from the input dataframe to the output dataframe'''
    def wrapper(input_dataframe, *args,**kwargs):
        dataframe = func(input_dataframe, *args,**kwargs)
        dataframe.attrs = input_dataframe.attrs
        return dataframe
    return wrapper