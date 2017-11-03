"""Parsers for extracting alchemical data from amber output files.
Most of the file parsing part are inheriting from alchemical-analysis  
Change the final format to pandas to be consistent with the alchemlyb format
"""

import os
import re
import pandas as pd
import numpy as np
import logging 

logger = logging.getLogger("alchemlyb.parsers.Amber")

def convert_to_pandas(file_datum):
    """Convert the data structure from numpy to pandas format"""
    data_dic = {}
    data_dic["dHdl"] = []
    data_dic["lambdas"] = []
    data_dic["time"] = []
    for frame_index, frame_dhdl in enumerate(file_datum.gradients):
        data_dic["dHdl"].append(frame_dhdl)
        data_dic["lambdas"].append(file_datum.clambda)
        #here we need to convert dt to ps unit from ns 
        frame_time = file_datum.t0 + (frame_index + 1) * file_datum.dt*1000
        data_dic["time"].append(frame_time)
    df = pd.DataFrame(data_dic["dHdl"], columns=["dHdl"], index =pd.Float64Index(data_dic["time"], name='time'))
    df["lambdas"] = data_dic["lambdas"][0]
    df = df.reset_index().set_index(['time'] + ['lambdas'])
    return df

DVDL_COMPS = ['BOND', 'ANGLE', 'DIHED', '1-4 NB', '1-4 EEL', 'VDWAALS',
              'EELEC', 'RESTRAINT']
_FP_RE = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
_MAGIC_CMPR = {
    '\x1f\x8b\x08': ('gzip', 'GzipFile'),  # last byte is compression method
    '\x42\x5a\x68': ('bz2', 'BZ2File')
}

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
        yield it.next()

class SectionParser(object):
    """
    A simple parser to extract data values from sections.
    """
    def __init__(self, filename):
        """Opens a file according to its file type."""
        self.filename = filename
        with open(filename, 'r') as f:
            magic = f.read(3)   # NOTE: works because all 3-byte headers
        try:
            method = _MAGIC_CMPR[magic]
        except KeyError:
            open_it = open
        else:
            open_it = getattr(__import__(method[0]), method[1])
        try:
            self.fileh = open_it(self.filename, 'r')
            self.filesize = os.stat(self.filename).st_size
        except Exception as ex:
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
        for line in self:
            match = re.search(pattern, line)
            if match:
                break
        return self.fileh.tell() != self.filesize

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
                if '*' in value:            # Fortran format overflow
                    result.append(float('Inf') )
                # NOTE: check if this is a sufficient test for int
                elif '.' not in value and re.search(r'\d+', value):
                    result.append(int(value))
                else:
                    result.append(float(value))
            else:                       # section may be incomplete
                result.append(None)
        return result

    def __iter__(self):
        return self

    def next(self):
        """Read next line of the filehandle and check for EOF."""
        self.lineno += 1
        curr_pos = self.fileh.tell()
        if curr_pos == self.filesize:
            raise StopIteration
        # NOTE: can't mix next() with seek()
        return self.fileh.readline()
    #make compatible with python 3.6
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
                 'component_gradients']

    def __init__(self):
        self.clambda = -1.0
        self.t0 = -1.0
        self.dt = -1.0
        self.T = -1.0
        self.gradients = []
        self.component_gradients = []

def file_validation(outfile):
    """validate the energy output file """
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

        if not secp.skip_after('^   3.  ATOMIC '):
            logging.warning('  WARNING: no ATOMIC section found, ignoring file\n')
            invalid = True

        t0, = secp.extract_section('^ begin time', '^$', ['coords'])
        if not secp.skip_after('^   4.  RESULTS'):
            logging.warning('  WARNING: no RESULTS section found, ignoring file\n')
            invalid = True
    if invalid:
        return False
    file_datum = FEData()
    file_datum.clambda = clambda
    file_datum.t0 = t0
    file_datum.dt = dt
    file_datum.T = T
    return file_datum

def extract_dHdl(outfile):
    """Return gradients `dH/dl` from Amebr TI outputfile
    Parameters
    ----------
    outfile : str
        Path to Amber .out file to extract data from.

    Returns
    -------
    dH/dl : Series
        dH/dl as a function of time for this lambda window.
    """
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
                    #CHECK the result
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
    if not finished:
        logging.warning('  WARNING: prematurely terminated run')
    if not nensec:
        logging.warning('  WARNING: File %s does not contain any DV/DL data\n' %
              outfile)
    logging.info('%i data points, %i DV/DL averages' % (nensec, nenav))
    #at this step we get info stored in the FEData object for a given amber out file
    file_datum.component_gradients.extend(comps)
    #convert file_datum to the pandas format to make it identical to alchemlyb output format
    df = convert_to_pandas(file_datum)        
    return df
