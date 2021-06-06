'''Unit conversion and constants.'''
# Taken from scipy.constants since py2 doesn't support it
#: Boltzmann's constant :math:`k_B` in kJ/(mol K); value from `NIST CODATA: k<https://physics.nist.gov/cgi-bin/cuu/Value?k>`_.
Boltzmann_constant = 1.380649e-23
#: Avogadro constant :math:`N_A` in 1/mol; value from `NIST CODATA: k<https://physics.nist.gov/cgi-bin/cuu/Value?na|search_for=avogadro>`_.
Avogadro_constant = 6.02214076e+23
kJ2kcal = 0.239006


def to_kT(df, T=None):
    """ Convert the unit of a DataFrame to `kT`.

    If temperature `T` is not provided, the DataFrame need to have attribute
    `temperature` and `energy_unit`.

    Parameters
    ----------
    df : DataFrame
        DataFrame to convert unit.
    T : float
        Temperature (default: None).

    Returns
    -------
    DataFrame
        `df` converted.
    """
    new_df = df.copy()
    if T is not None:
        new_df.attrs['temperature'] = T
    elif 'temperature' not in df.attrs:
        raise TypeError('Attribute temperature not found in the input '
                        'Dataframe.')

    if 'energy_unit' not in df.attrs:
        raise TypeError('Attribute energy_unit not found in the input '
                        'Dataframe.')

    attrs = new_df.attrs

    if df.attrs['energy_unit'] == 'kT':
        return new_df
    elif df.attrs['energy_unit'] == 'kJ/mol':
        new_df = new_df / (Boltzmann_constant * df.attrs['temperature'] *
                           Avogadro_constant / 1000)
        new_df.attrs = attrs
        new_df.attrs['energy_unit'] = 'kT'
        return new_df
    elif df.attrs['energy_unit'] == 'kcal/mol':
        new_df = new_df / (Boltzmann_constant * df.attrs['temperature'] *
                           Avogadro_constant / 1000 * kJ2kcal)
        new_df.attrs = attrs
        new_df.attrs['energy_unit'] = 'kT'
        return new_df
    else:
        raise ValueError('energy_unit {} can only be kT, kJ/mol or ' \
                        'kcal/mol.'.format(df.attrs['energy_unit']))


def to_kcalmol(df, T=None):
    """Convert the unit of a DataFrame to kcal/mol.

    If temperature `T` is not provided, the DataFrame need to have attribute
    `temperature` and `energy_unit`.

    Parameters
    ----------
    df : DataFrame
        DataFrame to convert unit.
    T : float
        Temperature (default: None).

    Returns
    -------
    DataFrame
        `df` converted.
    """
    kt_df = to_kT(df, T)
    kt_df = kt_df * (Boltzmann_constant * df.attrs['temperature'] *
                           Avogadro_constant / 1000 * kJ2kcal)
    kt_df.attrs = df.attrs
    kt_df.attrs['energy_unit'] = 'kcal/mol'
    return kt_df

def to_kJmol(df, T=None):
    """Convert the unit of a DataFrame to kJ/mol.

    If temperature `T` is not provided, the DataFrame need to have attribute
    `temperature` and `energy_unit`.

    Parameters
    ----------
    df : DataFrame
        DataFrame to convert unit.
    T : float
        Temperature (default: None).

    Returns
    -------
    DataFrame
        `df` converted.
    """
    kt_df = to_kT(df, T)
    kt_df = kt_df * (Boltzmann_constant * df.attrs['temperature'] *
                           Avogadro_constant / 1000)
    kt_df.attrs = df.attrs
    kt_df.attrs['energy_unit'] = 'kJ/mol'
    return kt_df
