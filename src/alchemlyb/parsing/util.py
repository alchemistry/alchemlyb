"""Collection of utilities used by many parsers.

"""
import os
import bz2
import gzip
import zipfile

# need to do some backflips to support Python 2 bz2 behavior
# bz2 in Python 2 doesn't have an open function, and in Python 3
# the BZ2File class only does binary mode
try:
    bz2.open
except AttributeError:
    bz2_open = bz2.BZ2File
else:
    def bz2_open(filename, mode):
        mode += 't' if mode in ['r','w','a','x'] else ''
        return bz2.open(filename, mode)

# similar changes need to be made for gzip
# gzip in Python 2 assumes text mode when 'r' is
# specified and in Python 3 gzip assumes binary mode when
# 'r' is specified
try:
    gzip.compress
except AttributeError:
    gzip_open = gzip.open
else:
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
