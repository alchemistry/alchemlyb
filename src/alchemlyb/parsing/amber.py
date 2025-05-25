"""Parsers for extracting alchemical data from `AMBER <http://ambermd.org>`_ output files.

Some of the file parsing parts are adapted from
`alchemical-analysis`_.

.. _alchemical-analysis: https://github.com/MobleyLab/alchemical-analysis

.. versionchanged:: 1.0.0
    Now raises :exc:`ValueError` when an invalid file is given to the parser.
    Now raises :exc:`ValueError` when inconsistency in MBAR states/data is found.

"""

import re

import numpy as np
import pandas as pd
from loguru import logger

from . import _init_attrs_dict
from .util import anyopen
from ..postprocessors.units import R_kJmol, kJ2kcal

k_b = R_kJmol * kJ2kcal

_FP_RE = r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?"


def convert_to_pandas(file_datum):
    """Convert the data structure from numpy to pandas format"""
    data_dic = {}
    data_dic["dHdl"] = []
    data_dic["lambdas"] = []
    data_dic["time"] = []
    for frame_index, frame_dhdl in enumerate(file_datum.gradients):
        data_dic["dHdl"].append(frame_dhdl)
        data_dic["lambdas"].append(file_datum.clambda)
        frame_time = file_datum.t0 + (frame_index + 1) * file_datum.dt * file_datum.ntpr
        data_dic["time"].append(frame_time)
    df = pd.DataFrame(
        data_dic["dHdl"],
        columns=["dHdl"],
        index=pd.Index(data_dic["time"], name="time", dtype="Float64"),
    )
    df["lambdas"] = data_dic["lambdas"][0]
    df = df.reset_index().set_index(["time"] + ["lambdas"])
    return df


def _pre_gen(it, first):
    """A generator that returns first first if it exists."""

    if first:
        yield first

    while it:
        try:
            yield next(it)
        except StopIteration:
            return


class SectionParser:
    """
    A simple parser to extract data values from sections.
    """

    def __init__(self, filename):
        """Opens a file according to its file type."""
        self.filename = filename
        try:
            self.fileh = anyopen(self.filename, "r")
        except:
            logger.exception("Cannot open file {}", filename)
            raise
        self.lineno = 0

    def skip_lines(self, nlines):
        """Skip a given number of lines."""
        lineno = 0
        for line in self:
            lineno += 1
            if lineno > nlines:
                return line
        return None

    def skip_after(self, pattern):
        """Skip until after a line that matches a regex pattern."""
        Found_pattern = False
        for line in self:
            match = re.search(pattern, line)
            if match:
                Found_pattern = True
                break
        return Found_pattern

    def extract_section(self, start, end, fields, limit=None, extra=""):
        """
        Extract data values (int, float) in fields from a section
        marked with start and end regexes.  Do not read further than
        limit regex.
        """
        inside = False
        lines = []
        for line in _pre_gen(self, extra):
            if limit and re.search(limit, line):
                break
            if re.search(start, line):
                inside = True
            if inside:
                if re.search(end, line):
                    break
                lines.append(line.rstrip("\n"))
        line = "".join(lines)
        result = []
        for field in fields:
            match = re.search(rf" {field}\s*=\s*(\*+|{_FP_RE}|\d+)", line)
            if match:
                value = match.group(1)
                if "*" in value:  # catch fortran format overflow
                    result.append(float("Inf"))
                else:
                    try:
                        result.append(int(value))
                    except ValueError:
                        result.append(float(value))
            else:  # section may be incomplete
                result.append(None)
        return result

    def __iter__(self):
        return self

    def __next__(self):
        """Read next line of the filehandle and check for EOF."""
        self.lineno += 1
        return next(self.fileh)

    def close(self):
        """Close the filehandle."""
        self.fileh.close()

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        self.close()


class FEData:
    """A simple struct container to collect data from individual files."""

    __slots__ = [
        "clambda",
        "t0",
        "dt",
        "T",
        "ntpr",
        "bar_intervall",
        "gradients",
        "mbar_energies",
        "have_mbar",
        "mbar_lambdas",
        "mbar_lambda_idx",
    ]

    def __init__(self):
        self.clambda = -1.0
        self.t0 = -1.0
        self.dt = -1.0
        self.T = -1.0
        self.ntpr = -1
        self.bar_intervall = -1
        self.gradients = []
        self.mbar_energies = []
        self.have_mbar = False
        self.mbar_lambdas = []
        self.mbar_lambda_idx = -1


def file_validation(outfile):
    """
    Function that validate and parse an AMBER output file.
    :exc:`ValueError` are risen if inconsinstencies in the input file are found.

    Parameters
    ----------
    outfile : str
        Path to AMBER .out file to validate and extract data from.

    Returns
    -------
    `:class:~FEData`
        FEData object populated with data from the parsed AMBER output file.

    """

    file_datum = FEData()
    with SectionParser(outfile) as secp:
        line = secp.skip_lines(5)

        if not line:
            logger.error("The file {} does not contain any data, it's empty.", outfile)
            raise ValueError(f"file {outfile} does not contain any data.")

        if not secp.skip_after("^   2.  CONTROL  DATA  FOR  THE  RUN"):
            logger.error('No "CONTROL DATA" section found in file {}.', outfile)
            raise ValueError(f'no "CONTROL DATA" section found in file {outfile}')

        (ntpr,) = secp.extract_section("^Nature and format of output:", "^$", ["ntpr"])
        nstlim, dt = secp.extract_section("Molecular dynamics:", "^$", ["nstlim", "dt"])
        (T,) = secp.extract_section("temperature regulation:", "^$", ["temp0"])
        if not T:
            logger.error('No valid "temp0" record found in file {}.', outfile)
            raise ValueError(f'no valid "temp0" record found in file {outfile}')

        (clambda,) = secp.extract_section(
            "^Free energy options:", "^$", ["clambda"], "^---"
        )
        if clambda is None:
            logger.error(
                'No free energy section found in file {}, "clambda" was None.', outfile
            )
            raise ValueError(f"no free energy section found in file {outfile}")

        mbar_ndata = 0
        have_mbar, bar_intervall, mbar_states = secp.extract_section(
            "^FEP MBAR options:",
            "^$",
            ["ifmbar", "bar_intervall", "mbar_states"],
            "^---",
        )
        if have_mbar:
            mbar_ndata = int(nstlim / bar_intervall)
            mbar_lambdas = _process_mbar_lambdas(secp)
            file_datum.mbar_lambdas = mbar_lambdas
            clambda_str = f"{clambda:6.4f}"

            if clambda_str not in mbar_lambdas:
                logger.warning(
                    "WARNING: lamba {} not contained in set of "
                    "MBAR lambas: {}\nNot using MBAR.",
                    clambda_str,
                    ", ".join(mbar_lambdas),
                )
                have_mbar = False
            else:
                mbar_nlambda = len(mbar_lambdas)
                if mbar_nlambda != mbar_states:
                    logger.error(
                        "the number of lambda windows read ({})"
                        "is different from what expected ({})",
                        ",".join(mbar_lambdas),
                        mbar_states,
                    )
                    raise ValueError(
                        f"the number of lambda windows read ({mbar_nlambda})"
                        f" is different from what expected ({mbar_states})"
                    )
                mbar_lambda_idx = mbar_lambdas.index(clambda_str)
                file_datum.mbar_lambda_idx = mbar_lambda_idx

                for _ in range(mbar_nlambda):
                    file_datum.mbar_energies.append([])

        if not secp.skip_after("^   3.  ATOMIC "):
            logger.error('No "ATOMIC" section found in the file {}.', outfile)
            raise ValueError(f'no "ATOMIC" section found in file {outfile}')

        (t0,) = secp.extract_section("^ begin time", "^$", ["coords"])
        if t0 is None:
            logger.error("No starting simulation time in file {}.", outfile)
            raise ValueError(f"No starting simulation time in file {outfile}")

        if not secp.skip_after("^   4.  RESULTS"):
            logger.error('No "RESULTS" section found in the file {}.', outfile)
            raise ValueError(f'no "RESULTS" section found in file {outfile}')

    file_datum.clambda = clambda
    file_datum.t0 = t0
    file_datum.dt = dt
    file_datum.ntpr = ntpr
    file_datum.bar_intervall = bar_intervall
    file_datum.T = T
    file_datum.have_mbar = have_mbar

    return file_datum


@_init_attrs_dict
def extract(outfile, T):
    """Return reduced potentials `u_nk` and gradients `dH/dl` from AMBER outputfile.

    Parameters
    ----------
    outfile : str
        Path to AMBER .out file to extract data from.
    T : float
        Temperature in Kelvin at which the simulations were performed;
        needed to generated the reduced potential (in units of kT)

    Returns
    -------
    Dict
        A dictionary with keys of 'u_nk', which is a pandas DataFrame of reduced potentials for each
        alchemical state (k) for each frame (n), and 'dHdl', which is a Series of dH/dl
        as a function of time for this lambda window.


    .. versionadded:: 1.0.0

    """

    beta = 1 / (k_b * T)

    file_datum = file_validation(outfile)

    if not np.isclose(T, file_datum.T, atol=0.01):
        msg = f"The temperature read from the input file ({file_datum.T:.2f} K)"
        msg += f" is different from the temperature passed as parameter ({T:.2f} K)"
        logger.error(msg)
        raise ValueError(msg)

    finished = False
    with SectionParser(outfile) as secp:
        line = secp.skip_lines(5)
        high_E_cnt = 0
        nensec = 0
        old_nstep = -1
        for line in secp:
            if "      A V E R A G E S   O V E R" in line:
                _ = secp.skip_after("^|=========================================")
            elif line.startswith(" NSTEP"):
                nstep, dvdl = secp.extract_section(
                    "^ NSTEP", "^ ---", ["NSTEP", "DV/DL"], extra=line
                )
                if nstep != old_nstep and dvdl is not None and nstep is not None:
                    if finished:
                        raise ValueError(
                            "TI Energy detected after the TIMINGS section."
                            + " Did you concatenate the output file?"
                        )
                    file_datum.gradients.append(dvdl)
                    nensec += 1
                    old_nstep = nstep
            elif line.startswith("MBAR Energy analysis") and file_datum.have_mbar:
                if finished:
                    raise ValueError(
                        "MBAR Energy detected after the TIMINGS section."
                        + " Did you concatenate the output file?"
                    )
                mbar = secp.extract_section(
                    "^MBAR", "^ ---", file_datum.mbar_lambdas, extra=line
                )

                if None in mbar:
                    msg = "Something strange parsing the following MBAR section."
                    msg += "\nMaybe the mbar_lambda values are incorrect?"
                    logger.error("{}\n{}", msg, mbar)
                    raise ValueError(msg)

                reference_energy = mbar[file_datum.mbar_lambda_idx]
                for lmbda, energy in enumerate(mbar):
                    if energy > 0.0:
                        high_E_cnt += 1

                    file_datum.mbar_energies[lmbda].append(
                        beta * (energy - reference_energy)
                    )
            elif line == "   5.  TIMINGS\n":
                finished = True

        if high_E_cnt:
            logger.warning(
                "{} MBAR energ{} > 0.0 kcal/mol",
                high_E_cnt,
                "ies are" if high_E_cnt > 1 else "y is",
            )

    if not finished:
        logger.warning("WARNING: file {} is a prematurely terminated run", outfile)

    if file_datum.have_mbar:
        mbar_time = [
            file_datum.t0 + (frame_index + 1) * file_datum.dt * file_datum.bar_intervall
            for frame_index in range(len(file_datum.mbar_energies[0]))
        ]

        mbar_df = pd.DataFrame(
            file_datum.mbar_energies,
            index=np.array(file_datum.mbar_lambdas, dtype=np.float64),
            columns=pd.MultiIndex.from_arrays(
                [mbar_time, np.repeat(file_datum.clambda, len(mbar_time))],
                names=["time", "lambdas"],
            ),
        ).T
    else:
        logger.info('WARNING: No MBAR energies found! "u_nk" entry will be None')
        mbar_df = None

    if not nensec:
        logger.warning("WARNING: File {} does not contain any dV/dl data", outfile)
        dHdl_df = None
    else:
        logger.info("Read {} dV/dl data points in file {}", nensec, outfile)
        dHdl_df = convert_to_pandas(file_datum)
        dHdl_df["dHdl"] *= beta

    return {"u_nk": mbar_df, "dHdl": dHdl_df}


def extract_dHdl(outfile, T):
    """Return gradients `dH/dl` from AMBER TI outputfile.

    Parameters
    ----------
    outfile : str
        Path to AMBER .out file to extract data from.
    T : float
        Temperature in Kelvin at which the simulations were performed

    Returns
    -------
    dH/dl : Series
        dH/dl as a function of time for this lambda window.


    .. versionchanged:: 0.5.0
        The :mod:`scipy.constants` is used for parsers instead of
        the constants used by the corresponding MD engine.

    """
    extracted = extract(outfile, T)
    return extracted["dHdl"]


def extract_u_nk(outfile, T):
    """Return reduced potentials `u_nk` from AMBER outputfile.

    Parameters
    ----------
    outfile : str
        Path to AMBER .out file to extract data from.
    T : float
        Temperature in Kelvin at which the simulations were performed;
        needed to generated the reduced potential (in units of kT)

    Returns
    -------
    u_nk : DataFrame
        Reduced potential for each alchemical state (k) for each frame (n).


    .. versionchanged:: 0.5.0
        The :mod:`scipy.constants` is used for parsers instead of
        the constants used by the corresponding MD engine.

    """
    extracted = extract(outfile, T)
    return extracted["u_nk"]


def _process_mbar_lambdas(secp):
    """
    Extract the lambda points used to compute MBAR energies from an AMBER MDOUT file.
    Parameters
    ----------
    secp: .out file from amber simulation.

    Returns
    -------
    mbar_lambdas: lambda values used for MBAR energy collection in simulation.

    """

    in_mbar = False
    mbar_lambdas = []

    for line in secp:
        if line.startswith("    MBAR - lambda values considered:"):
            in_mbar = True
            continue

        if in_mbar:
            if line.startswith("    Extra"):
                break

            if "total" in line:
                data = line.split()
                mbar_lambdas.extend(data[2:])
            else:
                # AMBER splits the MBAR states in more than one line
                # if there are many of them
                mbar_lambdas.extend(line.split())

    return mbar_lambdas
