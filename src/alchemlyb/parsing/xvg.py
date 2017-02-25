# -*- encoding: utf-8 -*-
# Copyright (c) 2009-2012 Oliver Beckstein <orbeckst@gmail.com>
# Released under the BSD-3 clause license

"""
Very Simple xmgrace XVG file format
===================================

Gromacs produces graphs in the `xmgrace`_ ("xvg") format. These are
simple multi-column data files. The class :class:`XVG` encapsulates
access to such files and adds a number of methods to access the data
(as NumPy arrays), compute aggregates, or quickly plot it.

.. _xmgrace: http://plasma-gate.weizmann.ac.il/Grace/

The :class:`XVG` class is useful beyond reading xvg files. With the
*array* keyword or the :meth:`XVG.set` method one can load data from
an array instead of a file. The array should be simple "NXY" data
(typically: first column time or position, further columns scalar
observables). The data should be a NumPy :class:`numpy.ndarray` array
``a`` with :attr:`~numpy.ndarray.shape` ``(M, N)`` where *M*-1 is the
number of observables and *N* the number of observations, e.g.the
number of time points in a time series. ``a[0]`` is the time or
position and ``a[1:]`` the *M*-1 data columns.


Data selection
~~~~~~~~~~~~~~

Plotting from :class:`XVG` is fairly flexible as one can always pass
the *columns* keyword to select which columns are to be
plotted. Assuming that the data contains ``[t, X1, X2, X3]``, then one
can

1) plot all observable columns (X1 to X3) against t::

     xvg.plot()

2) plot only X2 against t::

     xvg.plot(columns=[0,2])

3) plot X2 and X3 against t::

     xvg.plot(columns=[0,2,3])

4) plot X1 against X3::

     xvg.plot(columns=[2,3])




Classes and functions
---------------------

.. autoclass:: XVG
   :members:

"""


from __future__ import with_statement

import os, errno
import logging

import numpy

from .util import anyopen


logger = logging.getLogger("alchemlyb.parsing.xvg")

class XVG(object):
    """Class that represents the numerical data in a grace xvg file.

    All data must be numerical. :const:`NAN` and :const:`INF` values are
    supported via python's :func:`float` builtin function.

    The :attr:`~XVG.array` attribute can be used to access the the
    array once it has been read and parsed. The :attr:`~XVG.ma`
    attribute is a numpy masked array (good for plotting).

    .. Note::

       - Only simple XY or NXY files are currently supported, *not*
         Grace files that contain multiple data sets separated by '&'.
       - Any kind of formatting (i.e. :program:`xmgrace` commands) is discarded.
    """

    def __init__(self, filename=None, names=None, array=None, stride=1):
        """Initialize the class from a xvg file.

        :Arguments:
              *filename*
                    is the xvg file; it can only be of type XY or
                    NXY. If it is supplied then it is read and parsed
                    when :attr:`XVG.array` is accessed.
              *names*
                    optional labels for the columns (currently only
                    written as comments to file); string with columns
                    separated by commas or a list of strings
              *array*
                    read data from *array* (see :meth:`XVG.set`)
              *stride*
                    Only read every *stride* line of data [1].

        """
        self.__array = None           # cache for array (BIG) (used by XVG.array)
        self.__cache = {}             # cache for computed results

        if filename is not None:
            self.filename = filename  # note: reading data from file is delayed until required
        if names is None:
            self.names = []
        else:
            try:
                self.names = names.split(',')
            except AttributeError:
                self.names = names
        self.stride = stride

        if array is not None:
            self.set(array)

    def read(self, filename=None):
        """Read and parse xvg file *filename*."""
        self.filename = filename
        self.parse()

    def write(self, filename):
        """Write array to xvg file *filename* in NXY format."""
        with open(filename, 'w') as xvg:
            xvg.write("# xmgrace compatible NXY data file\n"
                      "# Written by xvg.XVG()\n")
            xvg.write("# :columns: {0!r}\n".format(self.names))
            for xyy in self.array.T:
                # quick and dirty ascii output... does NOT work with streams
                xyy.tofile(xvg, sep=" ", format="%-8s")
                xvg.write('\n')

    @property
    def array(self):
        """Represent xvg data as a (cached) numpy array.

        The array is returned with column-first indexing, i.e. for a data file with
        columns X Y1 Y2 Y3 ... the array a will be a[0] = X, a[1] = Y1, ... .
        """
        if self.__array is None:
            self.parse()
        return self.__array

    @property
    def ma(self):
        """Represent data as a masked array.

        The array is returned with column-first indexing, i.e. for a data file with
        columns X Y1 Y2 Y3 ... the array a will be a[0] = X, a[1] = Y1, ... .

        inf and nan are filtered via :func:`numpy.isfinite`.
        """
        a = self.array
        return numpy.ma.MaskedArray(a, mask=numpy.logical_not(numpy.isfinite(a)))

    @property
    def mean(self):
        """Mean value of all data columns."""
        return self.array[1:].mean(axis=1)

    @property
    def std(self):
        """Standard deviation from the mean of all data columns."""
        return self.array[1:].std(axis=1)

    @property
    def min(self):
        """Minimum of the data columns."""
        return self.array[1:].min(axis=1)

    @property
    def max(self):
        """Maximum of the data columns."""
        return self.array[1:].max(axis=1)

    def parse(self, stride=None):
        """Read and cache the file as a numpy array.

        Store every *stride* line of data; if ``None`` then the class default is used.

        The array is returned with column-first indexing, i.e. for a data file with
        columns X Y1 Y2 Y3 ... the array a will be a[0] = X, a[1] = Y1, ... .
        """
        if stride is None:
            stride = self.stride
        irow  = 0  # count rows of data
        # cannot use numpy.loadtxt() because xvg can have two types of 'comment' lines
        with anyopen(self.filename) as xvg:
            rows = []
            ncol = None
            for lineno, line in enumerate(xvg):
                line = line.strip()
                if line.startswith(('#', '@')) or len(line) == 0:
                    continue
                if line.startswith('&'):
                    raise NotImplementedError('{0!s}: Multi-data not supported, only simple NXY format.'.format(self.filename))
                # parse line as floats
                try:
                    row = map(float, line.split())
                except:
                    logger.error("%s: Cannot parse line %d: %r",
                                      self.filename, lineno+1, line)
                    raise
                # check for same number of columns as in previous step
                if ncol is not None and len(row) != ncol:
                    errmsg = "{0!s}: Wrong number of columns in line {1:d}: {2!r}".format(self.filename, lineno+1, line)
                    logger.error(errmsg)
                    raise IOError(errno.ENODATA, errmsg, self.filename)
                # finally: a good line
                if irow % stride == 0:
                    ncol = len(row)
                    rows.append(row)
                irow += 1
        try:
            self.__array = numpy.array(rows).transpose()    # cache result
        except:
            logger.error("%s: Failed reading XVG file, possibly data corrupted. "
                              "Check the last line of the file...", self.filename)
            raise
        finally:
            del rows     # try to clean up as well as possible as it can be massively big

    def set(self, a):
        """Set the *array* data from *a* (i.e. completely replace).

        No sanity checks at the moment...
        """
        self.__array = numpy.asarray(a)




