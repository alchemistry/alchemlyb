"""Parsers for extracting alchemical data from `Gromacs <http://www.gromacs.org/>`_ output files.

"""
import pandas as pd
import numpy as np

from .util import anyopen
from . import _init_attrs
from ..postprocessors.units import R_kJmol

k_b = R_kJmol

@_init_attrs
def extract_u_nk(xvg, T):
    r"""Return reduced potentials `u_nk` from a Hamiltonian differences XVG file.

    Parameters
    ----------
    xvg : str
        Path to XVG file to extract data from.
    T : float
        Temperature in Kelvin the simulations sampled.

    Returns
    -------
    u_nk : DataFrame
        Potential energy for each alchemical state (k) for each frame (n).


    Note
    -----
    Previous versions of alchemlyb (<0.5.0) used the `GROMACS value of the
    molar gas constant
    <https://manual.gromacs.org/documentation/2019/reference-manual/definitions.html>`_
    of :math:`R = 8.3144621 \times 10^{−3}\,
    \text{kJ}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}` instead of the scipy value
    :data:`scipy.constants.R` in :mod:`scipy.constants` (see
    :mod:`alchemlyb.postprocessors.units`).  The relative difference between
    the two values is :math:`6 \times 10^{-8}`.

    Therefore, results in :math:`kT` for GROMACS data will differ between
    alchemlyb ≥0.5.0 and previous versions; the relative difference is on the
    order of :math:`10^{-7}` for typical cases.


    .. versionchanged:: 0.5.0
        The :mod:`scipy.constants` is used for parsers instead of
        the constants used by the corresponding MD engine.
        This leads to slightly different results for GROMACS input compared to
        previous versions of alchemlyb.

    """

    h_col_match = r"\xD\f{}H \xl\f{}"
    pv_col_match = 'pV'
    u_col_match = ['Total Energy', 'Potential Energy']
    beta = 1/(k_b * T)

    state, lambdas, statevec = _extract_state(xvg)

    # extract a DataFrame from XVG data
    df = _extract_dataframe(xvg)

    times = df[df.columns[0]]

    # want to grab only dH columns
    DHcols = [col for col in df.columns if (h_col_match in col)]
    dH = df[DHcols]

    # gromacs also gives us pV directly; need this for reduced potential
    pv_cols = [col for col in df.columns if (pv_col_match in col)]
    pv = None
    if pv_cols:
        pv = df[pv_cols[0]]

    # gromacs also gives us total/potential energy U directly; need this for reduced potential
    u_cols = [col for col in df.columns if any(single_u_col_match in col for single_u_col_match in u_col_match)]
    u = None
    if u_cols:
        u = df[u_cols[0]]

    u_k = dict()
    cols = list()
    for col in dH:
        u_col = eval(col.split('to')[1])
        # calculate reduced potential u_k = dH + pV + U
        u_k[u_col] = beta * dH[col].values
        if pv_cols:
            u_k[u_col] += beta * pv.values
        if u_cols:
            u_k[u_col] += beta * u.values
        cols.append(u_col)

    u_k = pd.DataFrame(u_k, columns=cols,
                       index=pd.Float64Index(times.values, name='time'))

    # create columns for each lambda, indicating state each row sampled from
    # if state is None run as expanded ensemble data or REX
    if state is None:
        # if thermodynamic state is specified map thermodynamic
        # state data to lambda values, else (for REX)
        # define state based on the legend
        if 'Thermodynamic state' in df:
            ts_index = df.columns.get_loc('Thermodynamic state')
            thermo_state = df[df.columns[ts_index]]
            for i, l in enumerate(lambdas):
                v = []
                for t in thermo_state:
                    v.append(statevec[int(t)][i])
                u_k[l] = v
        else:
            state_legend = _extract_legend(xvg)
            for i, l in enumerate(state_legend):
                u_k[l] = state_legend[l]
    else:
        for i, l in enumerate(lambdas):
            try:
                u_k[l] = statevec[i]
            except TypeError:
                u_k[l] = statevec

    # set up new multi-index
    newind = ['time'] + lambdas
    u_k = u_k.reset_index().set_index(newind)

    u_k.name = 'u_nk'

    return u_k

@_init_attrs
def extract_dHdl(xvg, T):
    r"""Return gradients `dH/dl` from a Hamiltonian differences XVG file.

    Parameters
    ----------
    xvg : str
        Path to XVG file to extract data from.
    T : float
        Temperature in Kelvin the simulations sampled.

    Returns
    -------
    dH/dl : Series
        dH/dl as a function of time for this lambda window.

    Note
    -----
    Previous versions of alchemlyb (<0.5.0) used the `GROMACS value of the
    molar gas constant
    <https://manual.gromacs.org/documentation/2019/reference-manual/definitions.html>`_
    of :math:`R = 8.3144621 \times 10^{−3}\,
    \text{kJ}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}` instead of the scipy value
    :data:`scipy.constants.R` in :mod:`scipy.constants` (see
    :mod:`alchemlyb.postprocessors.units`).  The relative difference between
    the two values is :math:`6 \times 10^{-8}`.

    Therefore, results in :math:`kT` for GROMACS data will differ between
    alchemlyb ≥0.5.0 and previous versions; the relative difference is on the
    order of :math:`10^{-7}` for typical cases.


    .. versionchanged:: 0.5.0
        The :mod:`scipy.constants` is used for parsers instead of
        the constants used by the corresponding MD engine.
        This leads to slightly different results for GROMACS input compared to
        previous versions of alchemlyb.

    """
    beta = 1/(k_b * T)

    headers = _get_headers(xvg)
    state, lambdas, statevec = _extract_state(xvg, headers)

    # extract a DataFrame from XVG data
    df = _extract_dataframe(xvg, headers)

    times = df[df.columns[0]]

    # want to grab only dH/dl columns
    dHcols = []
    for l in lambdas:
        dHcols.extend([col for col in df.columns if (l in col)])

    dHdl = df[dHcols]

    # make dimensionless
    dHdl = beta * dHdl

    # rename columns to not include the word 'lambda', since we use this for
    # index below
    cols = [l.split('-')[0] for l in lambdas]

    dHdl = pd.DataFrame(dHdl.values, columns=cols,
                        index=pd.Float64Index(times.values, name='time'))

    # create columns for each lambda, indicating state each row sampled from
    # if state is None run as expanded ensemble data or REX
    if state is None:
        # if thermodynamic state is specified map thermodynamic
        # state data to lambda values, else (for REX)
        # define state based on the legend
        if 'Thermodynamic state' in df:
            ts_index = df.columns.get_loc('Thermodynamic state')
            thermo_state = df[df.columns[ts_index]]
            for i, l in enumerate(lambdas):
                v = []
                for t in thermo_state:
                    v.append(statevec[int(t)][i])
                dHdl[l] = v
        else:
            state_legend = _extract_legend(xvg)
            for i, l in enumerate(state_legend):
                dHdl[l] = state_legend[l]
    else:
        for i, l in enumerate(lambdas):
            try:
                dHdl[l] = statevec[i]
            except TypeError:
                dHdl[l] = statevec

    # set up new multi-index
    newind = ['time'] + lambdas
    dHdl= dHdl.reset_index().set_index(newind)

    dHdl.name='dH/dl'

    return dHdl


def _extract_state(xvg, headers=None):
    """Extract information on state sampled, names of lambdas.

    Parameters
    ----------
    xvg : str
        Path to XVG file to extract data from.
    headers: dict
       headers dictionary to search header information, reduced I/O by
       reusing if it is already parsed, e.g. _extract_state and
       _extract_dataframe in order need one-time header parsing

    """
    state = None
    if headers is None:
        headers = _get_headers(xvg)
    subtitle = _get_value_by_key(headers, 'subtitle')
    if subtitle and 'state' in subtitle:
        state = int(subtitle.split('state')[1].split(':')[0])
        lambdas = [word.strip(')(,') for word in subtitle.split() if 'lambda' in word]
        statevec = eval(subtitle.strip().split(' = ')[-1].strip('"'))

    # if expanded ensemble data is used the state variable will never be assigned
    # parsing expanded ensemble data
    if state is None:
        lambdas = []
        statevec = []
        for line in headers['_raw_lines']:
            if ('legend' in line) and ('lambda' in line):
                lambdas.append([word.strip(')(,') for word in line.split() if 'lambda' in word][0])
            if ('legend' in line) and (' to ' in line):
                statevec.append(([float(i) for i in line.strip().split(' to ')[-1].strip('"()').split(',')]))

    return state, lambdas, statevec


def _extract_legend(xvg):
    """Extract information on state sampled for REX simulations.

    """
    state_legend = {}
    with anyopen(xvg, 'r') as f:
        for line in f:
            if ('legend' in line) and ('lambda' in line):
                state_legend[line.split()[4]] = float(line.split()[6].strip('"'))

    return state_legend


def _extract_dataframe(xvg, headers=None):
    """Extract a DataFrame from XVG data using Pandas `read_csv()`.

    pd.read_csv() shows the same behavior building pandas Dataframe with better
    performance (approx. 2 to 4 times speed up). See Issue #81.

    Parameters
    ----------
    xvg: str
       Path to XVG file to extract data from.
    headers: dict
       headers dictionary to search header information, reduced I/O by
       reusing if it is already parsed. Direct access by key name

    """
    if headers is None:
        headers = _get_headers(xvg)

    xaxis = _get_value_by_key(headers, 'xaxis', 'label')
    names = [_get_value_by_key(headers, 's{}'.format(x), 'legend') for x in
            range(len(headers)) if 's{}'.format(x) in headers]
    cols = [xaxis] + names

    # march through column names, mark duplicates when found
    cols = [col + "{}[duplicated]".format(i) if col in cols[:i] else col
            for i, col, in enumerate(cols)]

    header_cnt = len(headers['_raw_lines'])
    df = pd.read_csv(xvg, sep=r"\s+", header=None, skiprows=header_cnt,
            na_filter=True, memory_map=True, names=cols, dtype=np.float64,
            float_precision='high')

    # drop duplicated columns (see PR #86 https://github.com/alchemistry/alchemlyb/pull/86/)
    df = df[df.columns[~df.columns.str.endswith("[duplicated]")]]

    return df


def _parse_header(line, headers={}, depth=2):
    """Build python dictionary for single line header

    Update python dictionary to ``headers`` by reading ``line`` separated by
    whitespace. If ``depth`` is given, at most ``depth`` nested key value store
    is added. `_val` key is reserved which contain remaining words from
    ``line``.

    Note
    ----
    No return value but 'headers' dictionary will be updated.

    Parameters
    ----------

    line: str
        header line to parse
    headers: dict
        headers dictionary to update, pass by reference
    depth: int
        depth of nested key and value store

    Examples
    --------
    "x y z" line turns into { 'x': { 'y': {'_val': 'z' }}}

    >>> headers={}
    >>> _parse_header('@ s0 legend "Potential Energy (kJ/mol)"', headers)
    >>> headers
    {'s0': {'legend': {'_val': 'Potential Energy (kJ/mol)'}}}

    """
    # Remove a first character, i.e. @
    s = line[1:].split(None, 1)
    next_t = headers[s[0]] = {}
    for i in range(1, depth):
        # ord('"') == 34
        # no further parsing for quoted value
        if len(s) > 1 and s[1][0] != '"':
            s = s[1].split(None, 1)
            next_t[s[0]] = {}
            next_t = next_t[s[0]]
        else:
            break

    next_t["_val"] = ''.join(s[1:]).rstrip().strip('"')


def _get_headers(xvg):
    """Build python dictionary from header lines

    Build nested key and value store by reading header ('@') lines from a file.
    Direct access to value provides reduced time complexity O(1).
    `_raw_lines` key is reserved to keep the original text.

    Example
    -------

    Given a xvg file containinig header lines like:

        ...
       @    title "dH/d\\xl\\f{} and \\xD\\f{}H"
       @    xaxis  label "Time (ps)"
       @    yaxis  label "dH/d\\xl\\f{} and \\xD\\f{}H (kJ/mol [\\xl\\f{}]\\S-1\\N)"
       @TYPE xy
       @ subtitle "T = 310 (K) \\xl\\f{} state 38: (coul-lambda, vdw-lambda) = (0.9500, 0.0000)"
       @ view 0.15, 0.15, 0.75, 0.85
       @ legend on
       @ legend box on
       @ legend loctype view
       @ legend 0.78, 0.8
       @ legend length 2
       @ s0 legend "Potential Energy (kJ/mol)"
       @ s1 legend "dH/d\\xl\\f{} coul-lambda = 0.9500"
       @ s2 legend "dH/d\\xl\\f{} vdw-lambda = 0.0000"
       ...

    >>> _get_headers(xvg)
    {'TYPE': {'xy': {'_val': ''}},
      'subtitle': {'_val': 'T = 310 (K) \\xl\\f{} state 38: (coul-lambda, vdw-lambda) = (0.9500, 0.0000)'},
      'title': {'_val': 'dH/d\\xl\\f{} and \\xD\\f{}H'},
      'view': {'0.15,': {'_val': '0.15, 0.75, 0.85'}},
      'xaxis': {'label': {'_val': 'Time (ps)'}},
      'yaxis': {'label': {'_val': 'dH/d\\xl\\f{} and \\xD\\f{}H (kJ/mol [\\xl\\f{}]\\S-1\\N)'}},
      ...(omitted)...
      '_raw_lines': ['@    title "dH/d\\xl\\f{} and \\xD\\f{}H"',
                    '@    xaxis  label "Time (ps)"',
                    '@    yaxis  label "dH/d\\xl\\f{} and \\xD\\f{}H (kJ/mol [\\xl\\f{}]\\S-1\\N)"',
                    '@TYPE xy',
                    '@ subtitle "T = 310 (K) \\xl\\f{} state 38: (coul-lambda, vdw-lambda) = (0.9500, 0.0000)"',
                    '@ view 0.15, 0.15, 0.75, 0.85',
                    '@ legend on',
                    '@ legend box on',
                    '@ legend loctype view',
                    '@ legend 0.78, 0.8',
                    '@ legend length 2',
                    '@ s0 legend "Potential Energy (kJ/mol)"',
                    '@ s1 legend "dH/d\\xl\\f{} coul-lambda = 0.9500"',
                    '@ s2 legend "dH/d\\xl\\f{} vdw-lambda = 0.0000"'],
      }

    Returns
    -------
    headers: dict

    """
    with anyopen(xvg, 'r') as f:
        headers = { '_raw_lines': [] }
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith('@'):
                _parse_header(line, headers)
                headers['_raw_lines'].append(line)
            elif line.startswith('#'):
                headers['_raw_lines'].append(line)
                continue
            # assuming to start a body section
            else:
                break

    return headers


def _get_value_by_key(headers, key1, key2=None):
    """Return value by two-level keys where the second key is optional

    Example
    -------

    >>> headers
    {'s0': {'legend': {'_val': 'Potential Energy (kJ/mol)'}},
            'subtitle': {'_val': 'T = 310 (K) \\xl\\f{} state 38: (coul-lambda,
                vdw-lambda) = (0.9500, 0.0000)'}}
    >>> _get_value_by_key(header, 's0','legend')
    'Potential Energy (kJ/mol)'
    >>> _get_value_by_key(header, 'subtitle')
    'T = 310 (K) \\xl\\f{} state 38: (coul-lambda, vdw-lambda) = (0.9500, 0.0000)'

    """
    val = None
    if key1 in headers:
        if key2 is not None and key2 in headers[key1]:
            val = headers[key1][key2]['_val']
        else:
            val = headers[key1]['_val']

    return val
