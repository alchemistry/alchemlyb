#! /usr/bin/python
"""Setuptools-based setup script for alchemlyb.

For a basic installation just type the command::

  python setup.py install

"""

from setuptools import setup, find_packages

setup(name='alchemlyb',
      version='0.1.0-dev',
      description='the simple alchemistry library',
      author='David Dotson',
      author_email='dotsdl@gmail.com',
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
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
      dependency_links=['http://github.com/alchemistry/alchemtest/tarball/master#egg=alchemtest-0.1.0-dev']
      install_requires=['numpy', 'pandas', 'pymbar', 'scipy', 'scikit-learn']
      )
