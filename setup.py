#! /usr/bin/python
"""Setuptools-based setup script for alchemlyb.

For a basic installation just type the command::

  python setup.py install

"""

from setuptools import setup, find_packages

import versioneer

setup(name='alchemlyb',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='the simple alchemistry library',
      author='David Dotson',
      author_email='dotsdl@gmail.com',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        ],
      packages=find_packages('src'),
      package_dir={'': 'src'},
      license='BSD',
      long_description=open('README.rst').read(),
      tests_require = ['pytest', 'alchemtest'],
      install_requires=['numpy', 'pandas>=0.23.0', 'pymbar', 'scipy', 'scikit-learn']
      )
