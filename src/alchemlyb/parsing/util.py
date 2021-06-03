"""Collection of utilities used by many parsers.

"""
import os
import bz2
import gzip
import zipfile

def bz2_open(filename, mode):
    mode += 't' if mode in ['r','w','a','x'] else ''
    return bz2.open(filename, mode)

def gzip_open(filename, mode):
    mode += 't' if mode in ['r','w','a','x'] else ''
    return gzip.open(filename, mode)

def anyopen(filename, mode='r'):
    """Return a file stream for filename, even if compressed.

    Supports files compressed with bzip2 (.bz2), gzip (.gz), and zip (.zip)
    compression schemes. The appropriate extension must be present for
    the function to properly handle the file.

    Parameters
    ----------
    filename : str
        Path to file to use.
    mode : str
        Mode for stream; usually 'r' or 'w'.

    Returns
    -------
    stream : stream
        Open stream for reading.

    """
    # opener for each type of file
    extensions = {'.bz2': bz2_open,
                  '.gz': gzip_open,
                  '.zip': zipfile.ZipFile}

    ext = os.path.splitext(filename)[1]

    if ext in extensions:
       opener= extensions[ext]

    else:
        opener = open

    return opener(filename, mode)
