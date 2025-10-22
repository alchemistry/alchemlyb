# alchemlyb: the simple alchemistry library

[![DOI](https://joss.theoj.org/papers/10.21105/joss.06934/status.svg)](https://doi.org/10.21105/joss.06934)[![Zenodo DOI](https://zenodo.org/badge/68669096.svg)](https://zenodo.org/badge/latestdoi/68669096) [![Documentation](https://readthedocs.org/projects/alchemlyb/badge/?version=latest)](http://alchemlyb.readthedocs.io/en/latest/) [![Build Status](https://github.com/alchemistry/alchemlyb/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/alchemistry/alchemlyb/actions/workflows/ci.yaml) [![Code coverage](https://codecov.io/gh/alchemistry/alchemlyb/branch/master/graph/badge.svg)](https://codecov.io/gh/alchemistry/alchemlyb) [![anaconda package](https://anaconda.org/conda-forge/alchemlyb/badges/version.svg)](https://anaconda.org/conda-forge/alchemlyb)

**alchemlyb** makes alchemical free energy calculations easier to do by leveraging the full power and flexibility of the PyData stack [Wu2024](#wu2024). It includes:

1. Parsers for extracting raw data from output files of common molecular dynamics engines such as [GROMACS](http://www.gromacs.org/), [AMBER](http://ambermd.org/), [NAMD](http://www.ks.uiuc.edu/Research/namd/) and [other simulation codes](https://alchemlyb.readthedocs.io/en/latest/parsing.html).

2. Subsamplers for obtaining uncorrelated samples from timeseries data (including extracting independent, equilibrated samples [Chodera2016](#chodera2016) as implemented in the [pymbar](https://pymbar.readthedocs.io/) package).

3. Estimators for obtaining free energies directly from this data, using best-practices approaches for multistate Bennett acceptance ratio (MBAR) [Shirts2008](#shirts2008) and BAR (from [pymbar](https://pymbar.readthedocs.io/)) and thermodynamic integration (TI).

## How to Cite

If you use this software in your research, please cite it as follows:

Wu, Z., Dotson, D. L., Alibay, I., Allen, B. K., Barhaghi, M. S., Hénin, J., Joseph, T. T., Kenney, I. M., Lee, H., Li, H., Lim, V., Liu, S., Marson, D., Merz, P. T., Schlaich, A., Mobley, D., Shirts, M. R., Beckstein, O. (2024). **alchemlyb: the simple alchemistry library.** *Journal of Open Source Software*, **9** (101), 6934. [https://doi.org/10.21105/joss.06934](https://doi.org/10.21105/joss.06934)

## Documentation

The documentation is hosted on [Read the Docs](https://alchemlyb.readthedocs.io/en/latest/).

## Installation

**Install** via `pip` from [PyPi (alchemlyb)](https://pypi.org/project/alchemlyb):

```bash
pip install alchemlyb
```

or as a `conda` package from the [conda-forge (alchemlyb)](https://anaconda.org/conda-forge/alchemlyb) channel:

```bash
conda install -c conda-forge alchemlyb
```

**Update** with `pip`:

```bash
pip install --update alchemlyb
```

or with `conda` run:

```bash
conda update -c conda-forge alchemlyb
```

to get the latest released version.

## Getting involved

Contributions of all kinds are very welcome.

If you have questions or want to discuss alchemlyb please post in the [alchemlyb Discussions](https://github.com/alchemistry/alchemlyb/discussions).

If you have bug reports or feature requests then please get in touch with us through the [Issue Tracker](https://github.com/alchemistry/alchemlyb/issues).

We also welcome code contributions: have a look at our [Developer Guide](https://github.com/alchemistry/alchemlyb/wiki/Developer-Guide). Open an issue with the proposed fix or change in the [Issue Tracker](https://github.com/alchemistry/alchemlyb/issues) and submit a pull request against the [alchemistry/alchemlyb](https://github.com/alchemistry/alchemlyb) GitHub repository.

## References

- <a id="shirts2008">Shirts, M.R.</a>, and Chodera, J.D. (2008). Statistically optimal analysis of samples from multiple equilibrium states. The Journal of Chemical Physics 129, 124105.
- <a id="chodera2016">Chodera, J.D</a>. (2016). A Simple Method for Automated Equilibration Detection in Molecular Simulations. Journal of Chemical Theory and Computation 12, 1799–1805.
- <a id="wu2024">Wu, Z.</a>, Dotson, D. L., Alibay, I., Allen, B. K., Barhaghi, M. S., Hénin, J., Joseph, T. T., Kenney, I. M., Lee, H., Li, H., Lim, V., Liu, S., Marson, D., Merz, P. T., Schlaich, A., Mobley, D., Shirts, M. R., Beckstein, O. (2024). **alchemlyb: the simple alchemistry library.** *Journal of Open Source Software*, **9** (101), 6934. [https://doi.org/10.21105/joss.06934](https://doi.org/10.21105/joss.06934)
