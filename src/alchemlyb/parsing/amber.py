"""Parsers for extracting alchemical data from `Amber <http://ambermd.org>`_ output files.

Most of the file parsing parts are inherited from
`alchemical-analysis`_.

.. _alchemical-analysis: https://github.com/MobleyLab/alchemical-analysis

"""

import os
import re
import logging

import pandas as pd
import numpy as np

from .util import anyopen

logger = logging.getLogger("alchemlyb.parsers.Amber")

# TODO: perhaps move constants elsewhere?
# these are the units we need for dealing with Amber, which uses
# kcal/mol for energies  http://ambermd.org/Questions/units.html
# (kB in kcal/molK)
k_b = 1.9872041e-3


def convert_to_pandas(file_datum):
    """Convert the data structure from numpy to pandas format"""
    data_dic = {}
    data_dic["dHdl"] = []
    data_dic["lambdas"] = []
    data_dic["time"] = []
    for frame_index, frame_dhdl in enumerate(file_datum.gradients):
        data_dic["dHdl"].append(frame_dhdl)
        data_dic["lambdas"].append(file_datum.clambda)
        # here we need to convert dt to ps unit from ns
        frame_time = file_datum.t0 + (frame_index + 1) * file_datum.dt * 1000
        data_dic["time"].append(frame_time)
    df = pd.DataFrame(data_dic["dHdl"], columns=["dHdl"],
                      index=pd.Float64Index(data_dic["time"], name='time'))
    df["lambdas"] = data_dic["lambdas"][0]
    df = df.reset_index().set_index(['time'] + ['lambdas'])
    return df


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


class SectionParser(object):
    """
    A simple parser to extract data values from sections.
    """

    def __init__(self, filename):
        """Opens a file according to its file type."""
        self.filename = filename
        try:
            self.fileh = anyopen(self.filename, 'r')
        except Exception as ex:  # pragma: no cover
            logging.exception("ERROR: cannot open file %s" % filename)
        self.lineno = 0

    def skip_lines(self, nlines):
        """Skip a given number of files."""
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

    def extract_section(self, start, end, fields, limit=None, extra='',
                        debug=False):
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
                lines.append(line.rstrip('\n'))
        line = ''.join(lines)
        result = []
        for field in fields:
            match = re.search(r' %s\s+=\s+(\*+|%s|\d+)'
                              % (field, _FP_RE), line)
            if match:
                value = match.group(1)
                # FIXME: assumes fields are only integers or floats
                if '*' in value:  # Fortran format overflow
                    result.append(float('Inf'))
                # NOTE: check if this is a sufficient test for int
                elif '.' not in value and re.search(r'\d+', value):
                    result.append(int(value))
                else:
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


class FEData(object):
    """A simple struct container to collect data from individual files."""

    __slots__ = ['clambda', 't0', 'dt', 'T', 'gradients',
                 'component_gradients', 'mbar_energies',
                 'have_mbar', 'mbar_lambdas', 'mbar_lambda_idx']

    def __init__(self):
        self.clambda = -1.0
        self.t0 = -1.0
        self.dt = -1.0
        self.T = -1.0
        self.gradients = []
        self.component_gradients = []
        self.mbar_energies = []
        self.have_mbar = False
        self.mbar_lambdas = []
        self.mbar_lambda_idx = -1


def file_validation(outfile):
    """validate the energy output file """
    file_datum = FEData()
    invalid = False
    with SectionParser(outfile) as secp:
        line = secp.skip_lines(5)
        if not line:
            logging.warning('  WARNING: file does not contain any useful data, '
                            'ignoring file')
            invalid = True
        if not secp.skip_after('^   2.  CONTROL  DATA  FOR  THE  RUN'):
            logging.warning('  WARNING: no CONTROL DATA found, ignoring file')
            invalid = True
        ntpr, = secp.extract_section('^Nature and format of output:', '^$',
                                     ['ntpr'])
        nstlim, dt = secp.extract_section('Molecular dynamics:', '^$',
                                          ['nstlim', 'dt'])
        T, = secp.extract_section('temperature regulation:', '^$',
                                  ['temp0'])
        if not T:
            logging.error('ERROR: Non-constant temperature MD not '
                          'currently supported')
            invalid = True
        clambda, = secp.extract_section('^Free energy options:', '^$',
                                        ['clambda'], '^---')
        if clambda is None:
            logging.warning('  WARNING: no free energy section found, ignoring file')
            invalid = True

        mbar_ndata = 0
        have_mbar, mbar_ndata = secp.extract_section('^FEP MBAR options:',
                                                      '^$',
                                                      ['ifmbar',
                                                        'bar_intervall'],
                                                      '^---')
        if have_mbar:
            mbar_ndata = int(nstlim / mbar_ndata)
            mbar_lambdas = _process_mbar_lambdas(secp)
            file_datum.mbar_lambdas = mbar_lambdas
            clambda_str = '%6.4f' % clambda

            if clambda_str not in mbar_lambdas:
                logging.warning('\n Warning: lamba %s not contained in set of '
                                'MBAR lambas: %s\nNot using MBAR.' %
                                (clambda_str, ', '.join(mbar_lambdas)))

                have_mbar = False
            else:
                mbar_nlambda = len(mbar_lambdas)
                mbar_lambda_idx = mbar_lambdas.index(clambda_str)
                file_datum.mbar_lambda_idx = mbar_lambda_idx

                for _ in range(mbar_nlambda):
                    file_datum.mbar_energies.append([])

        if not secp.skip_after('^   3.  ATOMIC '):
            logging.warning('  WARNING: no ATOMIC section found, ignoring file\n')
            invalid = True

        t0, = secp.extract_section('^ begin time', '^$', ['coords'])
        if not secp.skip_after('^   4.  RESULTS'):
            logging.warning('  WARNING: no RESULTS section found, ignoring file\n')
            invalid = True
    if invalid:
        return False
    file_datum.clambda = clambda
    file_datum.t0 = t0
    file_datum.dt = dt
    file_datum.T = T
    file_datum.have_mbar = have_mbar
    return file_datum


def extract_u_nk(outfile, T):
    """Return reduced potentials `u_nk` from Amber outputfile.

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

    """
    beta = 1/(k_b * T)

    file_datum = file_validation(outfile)
    if not file_validation(outfile):   # pragma: no cover
        return None
    if not file_datum.have_mbar:
        raise Exception('ERROR: No MBAR energies found! Cannot parse file.')
    with SectionParser(outfile) as secp:
        line = secp.skip_lines(5)
        high_E_cnt = 0
        for line in secp:
            if line.startswith('MBAR Energy analysis'):
                mbar = secp.extract_section('^MBAR', '^ ---', file_datum.mbar_lambdas,
                                            extra=line)

                if any_none(mbar):
                    continue

                E_ref = mbar[file_datum.mbar_lambda_idx]

                for lmbda, E in enumerate(mbar):
                    if E > 0.0:
                        high_E_cnt += 1

                    file_datum.mbar_energies[lmbda].append(beta * (E - E_ref))

        if high_E_cnt:
            logger.warning('\n WARNING: %i MBAR energ%s > 0.0 kcal/mol' %
                           (high_E_cnt, 'ies are' if high_E_cnt > 1 else 'y is'))

        time = [file_datum.t0 + (frame_index + 1) * file_datum.dt * 1000
                for frame_index in range(len(file_datum.mbar_energies[0]))]

    return pd.DataFrame(file_datum.mbar_energies,
                        columns=pd.MultiIndex.from_arrays([time, np.repeat(file_datum.clambda, len(time))],
                                                          names=['time', 'lambdas']),
                        index=np.array(file_datum.mbar_lambdas, dtype=np.float64)).T


def extract_dHdl(outfile, T):
    """Return gradients ``dH/dl`` from Amber TI outputfile.

    Parameters
    ----------
    outfile : str
        Path to Amber .out file to extract data from.
    T : float
        Temperature in Kelvin at which the simulations were performed

    Returns
    -------
    dH/dl : Series
        dH/dl as a function of time for this lambda window.
    """
    beta = 1/(k_b * T)

    file_datum = file_validation(outfile)
    if not file_validation(outfile):
        return None
    finished = False
    comps = []
    with SectionParser(outfile) as secp:
        line = secp.skip_lines(5)
        nensec = 0
        nenav = 0
        old_nstep = -1
        old_comp_nstep = -1
        high_E_cnt = 0
        in_comps = False
        for line in secp:
            if 'DV/DL, AVERAGES OVER' in line:
                in_comps = True
            if line.startswith(' NSTEP'):
                if in_comps:
                    # CHECK the result
                    result = secp.extract_section('^ NSTEP', '^ ---',
                                                  ['NSTEP'] + DVDL_COMPS,
                                                  extra=line)
                    if result[0] != old_comp_nstep and not any_none(result):
                        comps.append([float(E) for E in result[1:]])
                        nenav += 1
                        old_comp_nstep = result[0]
                    in_comps = False
                else:
                    nstep, dvdl = secp.extract_section('^ NSTEP', '^ ---',
                                                       ['NSTEP', 'DV/DL'],
                                                       extra=line)
                    if nstep != old_nstep and dvdl is not None \
                            and nstep is not None:
                        file_datum.gradients.append(dvdl)
                        nensec += 1
                        old_nstep = nstep
            if line == '   5.  TIMINGS\n':
                finished = True
                break
    if not finished:  # pragma: no cover
        logging.warning('  WARNING: prematurely terminated run')
    if not nensec:  # pragma: no cover
        logging.warning('  WARNING: File %s does not contain any DV/DL data',
                        outfile)
    logging.info('%i data points, %i DV/DL averages', nensec, nenav)
    # at this step we get info stored in the FEData object for a given amber out file
    file_datum.component_gradients.extend(comps)
    # convert file_datum to the pandas format to make it identical to alchemlyb output format
    df = convert_to_pandas(file_datum)
    df['dHdl'] *= beta
    return df


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
