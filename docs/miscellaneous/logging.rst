.. _logging_section:

Logging
=======

In **alchemlyb**, we use :mod:`loguru` for logging. By default, the
:mod:`loguru` will print the logging information into the
`sys.stderr <https://docs.python.org/3/library/sys.html#sys.stderr>`_.

Print to the stderr
-------------------

If you want to customise the printing to the `stderr`, you could remove the
existing sink first ::

    from loguru import logger
    logger.remove()

Then add other custom sink ::

    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

The loguru logger is compatible with the :mod:`logging` module of the Python
standard library and can easily be
`configured to log to a *logging* handler <https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging>`_.


Save to a file
--------------

Alternatively, one could save to a file simply with ::

    from loguru import logger
    logger.add("file_{time}.log")

See for `configured to log to a file <https://loguru.readthedocs.io/en/stable/overview.html#easier-file-logging-with-rotation-retention-compression>`_
for more explanation.
