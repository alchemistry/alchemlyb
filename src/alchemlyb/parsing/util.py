"""Collection of utilities used by many parsers.

"""
import os
from os import PathLike
from typing import IO, Optional
import bz2
import gzip
import zipfile

def bz2_open(filename, mode):
    mode += 't' if mode in ['r','w','a','x'] else ''
    return bz2.open(filename, mode)

def gzip_open(filename, mode):
    mode += 't' if mode in ['r','w','a','x'] else ''
    return gzip.open(filename, mode)

def anyopen(datafile: PathLike | IO, mode='r', compression=None):
    """Return a file stream for file or stream, even if compressed.

    Supports files compressed with bzip2 (.bz2), gzip (.gz), and zip (.zip)
    compression schemes. The appropriate extension must be present for
    the function to properly handle the file.

    If giving a stream for `datafile`, then you must specify `compression` if 
    the stream is compressed.

    Parameters
    ----------
    datafile : PathLike | IO
        Path to file to use, or an open IO stream. If an IO stream, use
        `compression` to specify the type of compression used, if any.
    mode : str
        Mode for stream; usually 'r' or 'w'.
    compression : str
        Use to specify compression.
        Must be one of 'bz2', 'gz', 'zip'.

    Returns
    -------
    stream : stream
        Open stream for reading.

    """
    # opener for each type of file
    extensions = {'.bz2': bz2_open,
                  '.gz': gzip_open,
                  '.zip': zipfile.ZipFile}

    # compression selections available
    compressions = {'bz2': bz2_open,
                    'gz': gzip_open,
                    'zip': zipfile.ZipFile}

    # if `datafile` is a stream
    if hasattr(datafile, 'read'):
        # if no compression specified, just pass the stream through
        if compression is None:
            return datafile
        elif compression in compressions:
            decompressor = compressions[compression]
            return decompressor(datafile, mode=mode)

    # otherwise, treat as a file
    ext = os.path.splitext(datafile)[1]

    if ext in extensions:
       opener = extensions[ext]

    else:
        opener = open

    return opener(datafile, mode)
