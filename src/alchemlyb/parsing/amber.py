"""Parsers for extracting alchemical data from `Amber <http://ambermd.org>`_ output files.

Most of the file parsing parts are based on `alchemical-analysis`_.
.. _alchemical-analysis: https://github.com/MobleyLab/alchemical-analysis

"""

# pylint: disable=broad-except, logging-fstring-interpolation, disable=unbalanced-tuple-unpacking, invalid-name

import re
import logging
import typing

import pandas as pd
import numpy as np

from .util import anyopen
from . import _init_attrs
from ..postprocessors.units import R_kJmol, kJ2kcal

logger = logging.getLogger("alchemlyb.parsers.Amber")

k_b = R_kJmol * kJ2kcal


def convert_to_pandas(file_datum):
    """Convert the data structure from numpy to pandas format"""
    data_dic = {}
    data_dic["dHdl"] = []
    data_dic["lambdas"] = []
    data_dic["time"] = []
    for frame_index, frame_dhdl in enumerate(file_datum.gradients, start=1):
        data_dic["dHdl"].append(frame_dhdl)
        data_dic["lambdas"].append(file_datum.clambda)
        data_dic["time"].append(file_datum.t_0 + frame_index * file_datum.d_t * file_datum.ntpr)
    dh_dl_df = pd.DataFrame(data_dic["dHdl"], columns=["dHdl"],
                      index=pd.Index(data_dic["time"], name='time', dtype='Float64'))
    dh_dl_df["lambdas"] = data_dic["lambdas"][0]
    # print(data_dic["lambdas"][0], file_datum.clambda)
    dh_dl_df = dh_dl_df.reset_index().set_index(['time'] + ['lambdas'])
    return dh_dl_df


DVDL_COMPS = ['BOND', 'ANGLE', 'DIHED', '1-4 NB', '1-4 EEL', 'VDWAALS',
              'EELEC', 'RESTRAINT']
_FP_RE = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'


def any_none(sequence):
    """Check if any element of a sequence is None."""

    for element in sequence:
        if element is None:
            return True

    return False


def _pre_gen(it, first):
    """A generator that returns first first if it exists."""

    if first:
        yield first

    while it:
        try:
            yield next(it)
        except StopIteration:
            return


class SectionParser():
    """
    A simple parser to extract data values from sections.
    """

    def __init__(self, filename):
        """Opens an AMBER output file"""
        self.filename = filename
        try:
            self.fileh = anyopen(self.filename, 'r')
        except Exception:  # pragma: no cover
            logger.exception(f"Cannot open file {filename}")
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
        found_pattern = False
        for line in self:
            match = re.search(pattern, line)
            if match:
                found_pattern = True
                break
        return found_pattern

    def extract_section(self, start, end, fields, limit=None, extra=''):
        """
        Extract data values (int, float) in fields from a section marked
        with start and end regexes. Do not read further than limit regex.
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
                lines.append(line.rstrip('\n'))
        line = ''.join(lines)
        result = []
        for field in fields:
            # old with %, re.search(r' %s\s+=\s+(\*+|%s|\d+)' % (field, _FP_RE), line)
            match = re.search(fr' {field}\s+=\s+(\*+|{_FP_RE}|\d+)', line)
            if match:
                value = match.group(1)
                # NOTE: assumes fields are only integers or floats, it breaks otherwise
                # NOTE: I changed the method to check if the value is int or float
                if '*' in value:  # Fortran format overflow
                    result.append(float('Inf'))
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

    def next(self):
        """Read next line of the filehandle and check for EOF."""
        self.lineno += 1
        return next(self.fileh)

    # make compatible with python 3.6
    __next__ = next

    def close(self):
        """Close the filehandle."""
        self.fileh.close()

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        self.close()


class FEData():
    """A simple struct container to collect data from individual files."""

    __slots__ = ['clambda', 't_0', 'd_t', 'T', 'ntpr', 'gradients',
                'mbar_energies', 'nstlim',
                 'have_mbar', 'mbar_lambdas', 'mbar_lambda_idx']

    def __init__(self):
        self.clambda = None
        self.t_0 = None
        self.d_t = None
        self.T = None
        self.nstlim = None
        self.ntpr = None
        self.gradients = []
        self.mbar_energies = []
        self.have_mbar = False
        self.mbar_lambdas = []
        self.mbar_lambda_idx = -1


def file_validation(outfile:str, extract_mbar:bool):
    """validate the energy output file """
    file_datum = FEData()
    invalid = False
    with SectionParser(outfile) as secp:
        line = secp.skip_lines(5)
        if not line:
            logger.warning('File does not contain any useful data, ignoring file.')
            invalid = True
        if not secp.skip_after('^   2.  CONTROL  DATA  FOR  THE  RUN'):
            logger.warning('No CONTROL DATA found, ignoring file.')
            invalid = True
        file_datum.ntpr, = secp.extract_section('^Nature and format of output:', '^$', ['ntpr'])
        file_datum.nstlim, file_datum.d_t = secp.extract_section(
            'Molecular dynamics:', '^$', ['nstlim', 'dt'])
        file_datum.T, = secp.extract_section('temperature regulation:', '^$', ['temp0'])
        if not file_datum.T:
            logger.error('Non-constant temperature MD not currently supported.')
            invalid = True
        file_datum.clambda, = secp.extract_section(
            '^Free energy options:', '^$', ['clambda'], '^---')
        if file_datum.clambda is None:
            logger.warning('No free energy section found, ignoring file.')
            invalid = True

        mbar_ndata = 0

        # NOTE: MBAR data read only if needed
        # NOTE: errors in parsing MBAR should not break dHdl parsing
        if extract_mbar:
            file_datum.have_mbar, mbar_ndata = secp.extract_section(
                '^FEP MBAR options:', '^$', ['ifmbar', 'bar_intervall'], '^---')
        else:
            file_datum.have_mbar = False

        if file_datum.have_mbar:
            mbar_ndata = int(file_datum.nstlim / mbar_ndata)
            file_datum.mbar_lambdas = _process_mbar_lambdas(secp)
            clambda_str = f'{file_datum.clambda:6.4f}'

            if clambda_str not in file_datum.mbar_lambdas:
                logger.warning(f'WARNING: lamba {clambda_str} not contained in set of '
                               f'MBAR lambas: {", ".join(file_datum.mbar_lambdas)}\n'
                               'Not using MBAR.')

                file_datum.have_mbar = False
            else:
                mbar_nlambda = len(file_datum.mbar_lambdas)
                file_datum.mbar_lambda_idx = file_datum.mbar_lambdas.index(clambda_str)
                for _ in range(mbar_nlambda):
                    file_datum.mbar_energies.append([])

        if not secp.skip_after('^   3.  ATOMIC '):
            logger.warning('No ATOMIC section found, ignoring file.')
            invalid = True

        file_datum.t_0, = secp.extract_section('^ begin time', '^$', ['coords'])
        if not secp.skip_after('^   4.  RESULTS'):
            logger.warning('No RESULTS section found, ignoring file.')
            invalid = True
    if invalid:
        return None
    if extract_mbar and not file_datum.have_mbar:
        raise Exception(f'ERROR: No MBAR energies found in file {outfile}.')
    return file_datum


@_init_attrs
def extract_dHdl_and_u_nk(
    outfile:str, T:typing.Optional[float] = None,
    return_dh_dl:bool = True,
    return_u_nk:bool = True):
    """Return reduced potentials `u_nk` and gradients ``dH/dl`` from Amber outputfile.

    Parameters
    ----------
    outfile : str
        Path to Amber .out file to extract data from.
    T : float
        Temperature in Kelvin at which the simulations were performed;
        needed to generated the reduced potential (in units of kT)

    Returns
    -------
    u_nk : DataFrame
        Reduced potential for each alchemical state (k) for each frame (n).

    dH/dl : Series
        dH/dl as a function of time for this lambda window.


    .. versionchanged:: 0.5.0
        The :mod:`scipy.constants` is used for parsers instead of
        the constants used by the corresponding MD engine.

    """
    if return_u_nk:
        file_datum = file_validation(outfile, extract_mbar=True)
    else:
        file_datum = file_validation(outfile, extract_mbar=False)
    if file_datum is None:  # pragma: no cover
        return None

    if T is None:
        T = file_datum.T
    elif not np.isclose(T, file_datum.T):
        logger.warning(f'  WARNING: the temperature read from the input file ({file_datum.T} K), '
        'is different from the temperature passed as parameter ({T} K)')
    beta = 1/(k_b * T)

    finished = False
    with SectionParser(outfile) as secp:
        line = secp.skip_lines(5)
        n_en_sections = 0
        old_nstep = -1
        in_average_section = False
        n_high_en = 0
        for line in secp:
            if 'DV/DL, AVERAGES OVER' in line:
                in_average_section = True
            if line.startswith(' NSTEP'):
                if in_average_section:
                    _ = secp.extract_section('^ NSTEP', '^ ---', ['NSTEP'] + DVDL_COMPS, extra=line)
                    in_average_section = False
                else:
                    nstep, dvdl = secp.extract_section(
                        '^ NSTEP', '^ ---', ['NSTEP', 'DV/DL'], extra=line)
                    if nstep != old_nstep and dvdl is not None and nstep is not None:
                        file_datum.gradients.append(dvdl)
                        n_en_sections += 1
                        old_nstep = nstep
            elif line.startswith('MBAR Energy analysis') and return_u_nk:
                mbar = secp.extract_section('^MBAR', '^ ---', file_datum.mbar_lambdas, extra=line)
                if any_none(mbar):
                    continue
                ref_energy = mbar[file_datum.mbar_lambda_idx]
                for lmbda, energy in enumerate(mbar):
                    if energy > 0.0:
                        n_high_en += 1
                    file_datum.mbar_energies[lmbda].append(beta * (energy - ref_energy))
            elif line == '   5.  TIMINGS\n':
                finished = True
                break

        if n_high_en:
            logger.warning(
                f'WARNING: {n_high_en} MBAR energ{"ies are" if n_high_en > 1 else "y is"}'
                ' > 0.0 kcal/mol')

    if not finished:  # pragma: no cover
        logger.warning('  WARNING: prematurely terminated run')

    if return_dh_dl:  # pragma: no cover
        if not n_en_sections:
            logger.warning(f'WARNING: File {outfile} does not contain any DV/DL data')
        logger.info(f'{n_en_sections} DV/DL data points')


    if return_dh_dl:
        dh_dl = convert_to_pandas(file_datum)
        dh_dl['dHdl'] *= beta
    if return_u_nk:
        time = [file_datum.t_0 + (frame_index + 1) * file_datum.d_t * file_datum.ntpr
                for frame_index in range(len(file_datum.mbar_energies[0]))]
        u_nk = pd.DataFrame(file_datum.mbar_energies,
                            columns=pd.MultiIndex.from_arrays([time, np.repeat(file_datum.clambda,
                                                            len(time))], names=['time', 'lambdas']),
                            index=np.array(file_datum.mbar_lambdas, dtype=np.float64)).T

    if return_dh_dl and return_u_nk:
        return dh_dl, u_nk
    if return_dh_dl:
        return dh_dl
    if return_u_nk:
        return u_nk


@_init_attrs
def extract_dHdl(outfile:str, T:typing.Optional[float] = None):
    """Return gradients ```dH/dl`` from Amber TI outputfile.

    Parameters
    ----------
    outfile : str
        Path to Amber .out file to extract data from.
    T : float (optional, will be read from the output file)
        Temperature in Kelvin at which the simulations were performed

    Returns
    -------
    dH/dl : Series
        dH/dl as a function of time for this lambda window.


    .. versionchanged:: 0.5.0
        The :mod:`scipy.constants` is used for parsers instead of
        the constants used by the corresponding MD engine.

    """

    return extract_dHdl_and_u_nk(outfile=outfile, T=T, return_dh_dl=True, return_u_nk=False)


@_init_attrs
def extract_u_nk(outfile:str, T:typing.Optional[float] = None):
    """Return reduced potentials ``u_nk`` from Amber outputfile.

    Parameters
    ----------
    outfile : str
        Path to Amber .out file to extract data from.
    T : float (optional, will be read from the output file)
        Temperature in Kelvin at which the simulations were performed

    Returns
    -------
    u_nk : DataFrame
        Reduced potential for each alchemical state (k) for each frame (n).

    .. versionchanged:: 0.5.0
        The :mod:`scipy.constants` is used for parsers instead of
        the constants used by the corresponding MD engine.

    """
    return extract_dHdl_and_u_nk(outfile=outfile, T=T, return_dh_dl=False, return_u_nk=True)


def _process_mbar_lambdas(secp):
    """
    Extract the lambda points used to compute MBAR energies from an AMBER MDOUT file.
    Parameters
    ----------
    secp: SectionParser object from AMBER MDOUT file.

    Returns
    -------
    mbar_lambdas: lambda values used for MBAR energy collection in simulation.

    """

    in_mbar = False
    mbar_lambdas = []

    for line in secp:
        if line.startswith('    MBAR - lambda values considered:'):
            in_mbar = True
            continue

        if in_mbar:
            if line.startswith('    Extra'):
                break

            if 'total' in line:
                data = line.split()
                mbar_lambdas.extend(data[2:])
            else:
                mbar_lambdas.extend(line.split())

    return mbar_lambdas
