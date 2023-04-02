.. _logging_section:

Logging
=======

In **alchemlyb**, we use :mod:`loguru` for logging. By default, the
:mod:`loguru` will print the logging information into the `stderr`.

Print to the stderr
-------------------

If you want to customise the printing to the `stderr`, you could remove the
existing sink first ::

    from loguru import logger
    logger.remove()

Then add other custom sink ::

    logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

Save to a file
--------------

Alternatively, one could save to a file simply with ::

    from loguru import logger
    logger.add("file_{time}.log")

