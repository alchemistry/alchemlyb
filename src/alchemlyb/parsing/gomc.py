"""Parsers for extracting alchemical data from `GOMC <http://gomc.eng.wayne.edu/>`_ output files."""

import pandas as pd

from . import _init_attrs
from .util import anyopen
from ..postprocessors.units import R_kJmol

k_b = R_kJmol


@_init_attrs
def extract_u_nk(filename: str, T: float) -> pd.DataFrame:
    """Return reduced potentials `u_nk` from a Hamiltonian differences dat file.

    Parameters
    ----------
    filename : str
        Path to free energy file to extract data from.
    T : float
        Temperature in Kelvin at which the simulation was sampled.

    Returns
    -------
    u_nk : :class:`pandas.DataFrame`
        Potential energy for each alchemical state (k) for each frame (n).


    .. versionchanged:: 0.5.0
        The :mod:`scipy.constants` is used for parsers instead of
        the constants used by the corresponding MD engine.

    """

    h_col_match = "DelE"
    pv_col_match = "PV"
    u_col_match = ["Total_En"]
    beta = 1 / (k_b * T)

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
    u_cols = [
        col
        for col in df.columns
        if any(single_u_col_match in col for single_u_col_match in u_col_match)
    ]
    u = None
    if u_cols:
        u = df[u_cols[0]]

    u_k = dict()
    cols = list()
    for col in dH:
        u_col = eval(col.split("->")[1][:-1])  # type: ignore[attr-defined]
        # calculate reduced potential u_k = dH + pV + U
        u_k[u_col] = beta * dH[col].values
        if pv_cols:
            u_k[u_col] += beta * pv.values  # type: ignore[union-attr]
        if u_cols:
            u_k[u_col] += beta * u.values  # type: ignore[union-attr]
        cols.append(u_col)

    u_k = pd.DataFrame(
        u_k, columns=cols, index=pd.Index(times.values, name="time", dtype="Float64")
    )  # type: ignore[assignment]

    # Need to modify the lambda name
    cols = [lambda_value + "-lambda" for lambda_value in lambdas]
    # create columns for each lambda, indicating state each row sampled from
    for i, lambda_value in enumerate(cols):
        u_k[lambda_value] = statevec[i]

    # set up new multi-index
    newind = ["time"] + cols
    u_k = u_k.reset_index().set_index(newind)  # type: ignore[attr-defined]

    u_k.name = "u_nk"

    return u_k  # type: ignore[no-any-return]


@_init_attrs
def extract_dHdl(filename: str, T: float) -> pd.DataFrame:
    """Return gradients `dH/dl` from a Hamiltonian differences free energy file.

    Parameters
    ----------
    filename : str
        Path to free energy file to extract data from.
    T : float
        Temperature in Kelvin at which the simulation was sampled.

    Returns
    -------
    dH/dl : :class:`pandas.Series`
        dH/dl as a function of step for this lambda window.


    .. versionchanged:: 0.5.0
        The :mod:`scipy.constants` is used for parsers instead of
        the constants used by the corresponding MD engine.

    """
    beta = 1 / (k_b * T)

    state, lambdas, statevec = _extract_state(filename)

    # extract a DataFrame from free energy data
    df = _extract_dataframe(filename)

    times = df[df.columns[0]]

    # want to grab only dH/dl columns
    dHcols = []
    for lambda_value in lambdas:
        dHcols.extend([col for col in df.columns if (lambda_value in col)])

    dHdl = df[dHcols]

    # make dimensionless
    dHdl *= beta

    dHdl = pd.DataFrame(
        dHdl.values,
        columns=lambdas,
        index=pd.Index(times.values, name="time", dtype="Float64"),
    )

    # Need to modify the lambda name
    cols = [lambda_value + "-lambda" for lambda_value in lambdas]
    # create columns for each lambda, indicating state each row sampled from
    for i, lambda_value in enumerate(cols):
        dHdl[lambda_value] = statevec[i]

    # set up new multi-index
    newind = ["time"] + cols
    dHdl = dHdl.reset_index().set_index(newind)

    dHdl.name = "dH/dl"  # type: ignore[attr-defined]

    return dHdl


def extract(filename: str, T: float) -> dict[str, pd.DataFrame | None]:
    r"""Return reduced potentials `u_nk` and gradients `dH/dl`
    from a Hamiltonian differences free energy file.

    Parameters
    ----------
    xvg : str
        Path to free energy file to extract data from.
    T : float
        Temperature in Kelvin the simulations sampled.
    filter : bool
        Filter out the lines that cannot be parsed.
        Such as rows with incorrect number of Columns and incorrectly
        formatted numbers (e.g. 123.45.67, nan or -).

    Returns
    -------
    :class:`dict`
        A dictionary with keys of 'u_nk', which is a :class:`~pandas.DataFrame` of
        potential energy for each alchemical state (k) for each frame (n),
        and 'dHdl', which is a :class:`~pandas.Series` of dH/dl
        as a function of time for this lambda window.


    .. versionadded:: 1.0.0
    """

    return {"u_nk": extract_u_nk(filename, T), "dHdl": extract_dHdl(filename, T)}


def _extract_state(filename: str) -> tuple[int, list[str], list[float]]:
    """Extract information on state sampled, names of lambdas."""
    state = None
    with anyopen(filename, "r") as f:  # type: ignore[arg-type]
        for line in f:
            if ("#" in line) and ("State" in line):
                state = int(line.split("State")[1].split(":")[0])
                # GOMC always print these two fields
                lambdas = ["Coulomb", "VDW"]
                statevec = eval(line.strip().split(" = ")[-1])
                break

    return state, lambdas, statevec  # type: ignore[return-value]


def _extract_dataframe(filename: str) -> pd.DataFrame:
    """Extract a DataFrame from free energy data."""
    dh_col_match = "dU/dL"
    h_col_match = "DelE"
    pv_col_match = "PV"
    u_col_match = "Total_En"

    xaxis = "time"
    with anyopen(filename, "r") as f:  # type: ignore[arg-type]
        names = []
        rows = []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                # avoid parsing empty line
                continue
            elif line.startswith("#T"):
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
