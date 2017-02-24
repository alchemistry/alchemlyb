"""Collection of utilities used by many parsers.

"""
import os
from bz2 import BZ2File
from gzip import GzipFile
from zipfile import ZipFile

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
    extensions = {'.bz2': BZ2File,
                  '.gz': GzipFile,
                  '.zip': ZipFile}

    ext = os.path.splitext(filename)[1]

    if ext in extensions:
       opener = extensions[ext] 

    else:
        opener = open

    return opener(filename, 'r')
