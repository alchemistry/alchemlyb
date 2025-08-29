Installing alchemlyb
====================

*alchemlyb* is available via the ``pip`` and ``conda`` package
managers and can easily be installed with all its
dependencies. Alternatively, it can also be directly installed from
source 


``conda`` installation
----------------------

|install_with_conda| |anaconda_package| |platforms| |last_updated|

	    
The easiest way to keep track of all dependencies is to **install**
*alchemlyb* as a `conda`_ package from the `conda-forge (alchemlyb)`_
channel ::

  conda install -c conda-forge alchemlyb 


You can later **update** your installation with ::

  conda update -c conda-forge alchemlyb

Managing the ``JAX`` Dependency
-------------------------------

By default, the conda installation of alchemlyb includes the JAX library for accelerated performance. If you encounter dependency conflicts or prefer a leaner installation, you can manage this dependency.

To install without JAX, specify the core build. This version offers the basic functionality without the JAX-based acceleration. ::

	conda install -c conda-forge 'alchemlyb=*=core_*'

To explicitly install with JAX, you can specify the jax build. This is useful to ensure you're getting the accelerated version. ::

	conda install -c conda-forge 'alchemlyb=*=jax_*'

``pip`` installation
--------------------

**Install** via ``pip`` from `PyPi (alchemlyb)`_ ::

  pip install alchemlyb

**Update** with ::

  pip install --update alchemlyb



Installing from source
----------------------

To install from source, first clone the source code repository
https://github.com/alchemistry/alchemlyb from GitHub with ::

    git clone https://github.com/alchemistry/alchemlyb.git

and then install with ``pip`` ::

    cd alchemlyb
    pip install .



.. _`PyPi (alchemlyb)`: https://pypi.org/project/alchemlyb/
.. _`conda`: https://conda.io/
.. _`conda-forge (alchemlyb)`: https://anaconda.org/conda-forge/alchemlyb
    
.. |install_with_conda| image:: https://anaconda.org/conda-forge/alchemlyb/badges/installer/conda.svg
   :alt: install with conda
   :target: https://conda.anaconda.org/conda-forge

.. |anaconda_package| image:: https://anaconda.org/conda-forge/alchemlyb/badges/version.svg
   :alt: anaconda package
   :target: https://anaconda.org/conda-forge/alchemlyb

.. |platforms| image:: https://anaconda.org/conda-forge/alchemlyb/badges/platforms.svg
   :alt: platforms
   :target: https://anaconda.org/conda-forge/alchemlyb

.. |last_updated| image:: https://anaconda.org/conda-forge/alchemlyb/badges/latest_release_date.svg
   :alt: last updated
   :target: https://anaconda.org/conda-forge/alchemlyb

