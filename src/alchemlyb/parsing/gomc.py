"""Parsers for extracting alchemical data from `GOMC <http://gomc.eng.wayne.edu/>`_ output files.

"""
import pandas as pd

from .util import anyopen


# TODO: perhaps move constants elsewhere?
# these are the units we need for dealing with GOMC, so not
# a bad place for it, honestly
# (kB in kJ/molK)
k_b = 8.3144621E-3


def extract_u_nk(filename, T):
    """Return reduced potentials `u_nk` from a Hamiltonian differences dat file.

    Parameters
    ----------
    filename : str
        Path to free energy file to extract data from.
    T : float
        Temperature in Kelvin at which the simulation was sampled.

    Returns
    -------
    u_nk : DataFrame
        Potential energy for each alchemical state (k) for each frame (n).

    """

    dh_col_match = "dU/dL"
    h_col_match = "DelE"
    pv_col_match = 'PV'
    u_col_match = ['Total_En']
    beta = 1/(k_b * T)

    state, lambdas, statevec = _extract_state(filename)

    # extract a DataFrame from free energy file data
    df = _extract_dataframe(filename)

    times = df[df.columns[0]]

    # want to grab only dH columns
    DHcols = [col for col in df.columns if (h_col_match in col)]
    dH = df[DHcols]

    # GOMC also gives us pV directly; need this for reduced potential
    pv_cols = [col for col in df.columns if (pv_col_match in col)]
    pv = None
    if pv_cols:
        pv = df[pv_cols[0]]

    # GOMC also gives us total energy U directly; need this for reduced potential
    u_cols = [col for col in df.columns if any(single_u_col_match in col for single_u_col_match in u_col_match)]
    u = None
    if u_cols:
        u = df[u_cols[0]]

    u_k = dict()
    cols = list()
    for col in dH:
        u_col = eval(col.split('->')[1][:-1])
        # calculate reduced potential u_k = dH + pV + U
        u_k[u_col] = beta * dH[col].values
        if pv_cols:
            u_k[u_col] += beta * pv.values
        if u_cols:
            u_k[u_col] += beta * u.values
        cols.append(u_col)

    u_k = pd.DataFrame(u_k, columns=cols,
                       index=pd.Float64Index(times.values, name='time'))

    # Need to modify the lambda name
    cols = [l + "-lambda" for l in lambdas]
    # create columns for each lambda, indicating state each row sampled from
    for i, l in enumerate(cols):
        u_k[l] = statevec[i]

    # set up new multi-index
    newind = ['time'] + cols
    u_k = u_k.reset_index().set_index(newind)

    u_k.name = 'u_nk'

    return u_k


def extract_dHdl(filename, T):
    """Return gradients `dH/dl` from a Hamiltonian differences free energy file.

    Parameters
    ----------
    filename : str
        Path to free energy file to extract data from.
    T : float
        Temperature in Kelvin at which the simulation was sampled.

    Returns
    -------
    dH/dl : Series
        dH/dl as a function of step for this lambda window.

    """
    beta = 1/(k_b * T)

    state, lambdas, statevec = _extract_state(filename)

    # extract a DataFrame from free energy data
    df = _extract_dataframe(filename)

    times = df[df.columns[0]]

    # want to grab only dH/dl columns
    dHcols = []
    for l in lambdas:
        dHcols.extend([col for col in df.columns if (l in col)])

    dHdl = df[dHcols]

    # make dimensionless
    dHdl *= beta

    dHdl = pd.DataFrame(dHdl.values, columns=lambdas,
                        index=pd.Float64Index(times.values, name='time'))

    # Need to modify the lambda name
    cols = [l + "-lambda" for l in lambdas]
    # create columns for each lambda, indicating state each row sampled from
    for i, l in enumerate(cols):
        dHdl[l] = statevec[i]

    # set up new multi-index
    newind = ['time'] + cols
    dHdl= dHdl.reset_index().set_index(newind)

    dHdl.name='dH/dl'

    return dHdl


def _extract_state(filename):
    """Extract information on state sampled, names of lambdas.

    """
    state = None
    with anyopen(filename, 'r') as f:
        for line in f:
            if ('#' in line) and ('State' in line):
                state = int(line.split('State')[1].split(':')[0])
                # GOMC always print these two fields
                lambdas = ['Coulomb', 'VDW']
                statevec = eval(line.strip().split(' = ')[-1])
                break

    return state, lambdas, statevec


def _extract_dataframe(filename):
    """Extract a DataFrame from free energy data.

    """
    dh_col_match = "dU/dL"
    h_col_match = "DelE"
    pv_col_match = 'PV'
    u_col_match = 'Total_En'

    xaxis = "time"
    with anyopen(filename, 'r') as f:
        names = []
        rows = []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                # avoid parsing empty line
                continue
            elif line.startswith('#T'):
                # this line has state information. No need to be parsed
                continue
            elif line.startswith("#Steps"):
                # read the headers
                elements = line.split()
                for i, element in enumerate(elements):
                    if element.startswith(u_col_match):
                        names.append(element)
                    elif element.startswith(dh_col_match):
                        names.append(element)
                    elif element.startswith(h_col_match):
                        names.append(element)
                    elif element.startswith(pv_col_match):
                        names.append(element)
            else:
                # parse line as floats
                row = map(float, line.split())
                rows.append(row)

    cols = [xaxis]
    cols.extend(names)

    return pd.DataFrame(rows, columns=cols)
