---
title: 'alchemlyb: the simple alchemistry library'
tags:
  - Python
  - alchemistry
  - molecular dynamics
  - free energy
authors:
  - name: Zhiyi Wu
    orcid: 0000-0002-7615-7851
    equal-contrib: true
    affiliation: 1
  - name: David L. Dotson
    orcid: 0000-0001-5879-2942
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Michael R. Shirts
    orcid: 0000-0003-3249-1097
    affiliation: 3
  - name: Oliver Beckstein
    orcid: 0000-0003-1340-0831
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 4
affiliations:
 - name: Exscientia, Oxford, UK
   index: 1
 - name: Datryllic LLC, Phoenix, AZ, USA
   index: 2
 - name: University of Colorado Boulder, Boulder, Colorado, USA
   index: 3
 - name: Arizona State University, Tempe, Arizona, USA
   index: 4

date: 31 May 2024
bibliography: paper.bib

---

# Summary

*alchemlyb* is an open-source Python software package for the analysis of alchemical free energy calculations, an integral part of computational chemistry and biology, most notably in the field of drug discovery.
Its functionality covers contains individual function-based building blocks for all aspects of a full typical free energy analysis workflow, starting with the extraction of raw data from the output of diverse molecular dynamics (MD) packages, moving on to data preprocessing tasks such as decorrelation of time series, using various estimators to derive free energy estimates from simulation samples, and finally providing quality analysis tools for data convergence checking and visualization.

*alchemlyb* also contains high-level end-to-end workflows that combine multiple building blocks into a user-friendly analysis pipeline from the initial data input stage to the final result derivation. This workflow functionality enhances accessibility by enabling researchers from diverse scientific backgrounds, and not solely computational chemistry specialists, to utilize *alchemlyb* effectively.


# Statement of need

In the pharmaceutical sector, computational chemistry techniques are integral for evaluating potential drug compounds based on their protein binding affinity [@deng2009computations].
Notably, absolute binding free energy calculations between proteins and ligands or relative binding affinity of ligands to the same protein are routinely employed for this purpose [@merz2010drug].
The resultant estimates of these free energies are essential for understanding binding affinity throughout various stages of drug discovery, such as hit identification and lead optimization [@merz2010drug].
Other free energies extracted from simulations are useful in solution thermodynamics, chemical engineering, environmental science, and material science.

Molecular dynamics (MD) packages such as GROMACS [@pronk2013gromacs], AMBER [@case2014ff14sb], NAMD [@phillips2020scalable], and GOMC [@cummings2021open] are used to run free energy simulations and many of these packages also contain tools for the subsequent processing of simulation data into free energies.
However, there are no standard output formats and analysis tools implement different algorithms for the different stages of the free energy data processing pipeline.
Therefore, it is very difficult to analyze data from different MD packages in a consistent manner.
Furthermore, the native analysis tools do not always implement current best practices [@klimovich2015guidelines,@Mey2020aa] or are out of date
Overall, the coupling between data generation and analysis in most MD packages hinders seamless collaboration and comparison of results across packages.

*alchemlyb* addresses this problem by focusing only on the data analysis with the goal to provide a unified interface for working with free energy data.
In an initial step data are read from the native MD package file formats and then organized into a common standard data structure, a *pandas* `Dataframe` [@mckinney-proc-scipy-2010] (https://pandas.pydata.org).
Functions are provided for pre-processing data by subsampling or decorrelation.
Statistical mechanical estimators are available to derive free energy quantities; these estimators are implemented as classes with the same API as estimators in scikit-learn [@scikitlearn2011,@sklearn2013api] (https://scikit-learn.org).
Overall, *alchemlyb* implements modular building blocks to simplify the process of extracting crucial thermodynamic insights from molecular simulations in a uniform manner.

*alchemlyb* succeeds the widely-used but now deprecated [alchemical-analysis.py](https://github.com/MobleyLab/alchemical-analysis) tool [@klimovich2015guidelines], which combined pre-processing, free energy estimation, and plotting in a single script. 
`alchemical-analysis.py` was not thoroughly tested and hard to integrate into modern workflows due to its monolithic design. 
*alchemlyb* improves over its predecessor with a modular, function based design and thorough testing of all components using continuous integration.
Thus, *alchemlyb* is primarily a library that enables users to easily use well-tested building blocks within their own tools while additionally providing examples of complete end-to-end workflows.
This innovation enables consistent processing of free energy data from diverse MD packages, facilitating streamlined comparison and combination of results.

Notably, *alchemlyb*'s robust and user-friendly nature has led to its integration into other automated workflow libraries such as BioSimSpace [@hedges2023suite] or MDPOW [@fan2020aa], demonstrating its accessibility and usability within broader scientific workflows and reinforcing its position as a versatile tool in the field of computational chemistry.


# Implementation

Free energy differences are fundamental to understand many different processes at the molecular scale, ranging from the binding of drug molecules to their receptor proteins or nucleic acids through the partitioning of molecules into different solvents or phases to the stability of crystals and biomolecules.
The calculation of free energy differences is challenging but in the specific case of binding free energies and partitioning free energies, alchemical free energy calculations have emerged as a de-facto standard approach whereby non-physical intermediate states are introduced to bridge between the physical end states of the process of interest, such as the drug bound to the receptor and the drug free in solution [@Mey2020aa].
In stratified ("windowed") alchemical free energy calculations, the system's interaction (its Hamiltonian) is continuously transformed as a function of a vector of $N$ alchemical parameters $\vec{\lambda}=(\lambda_1, \lambda_2, \dots, \lambda_N)$, so that in general $\vec{\lambda}=(0, 0, \dots, 0)$ indicates the initial physically realizable state and $\vec{\lambda} = (1, 1, \dots, 1)$ the final physically realizable state, while any intermediate configurations are non-physical but required for converging the calculations. 
At each $\vec{\lambda}$-value (or "window"), the system configurations are sampled in the relevant thermodynamic ensemble, typically using molecular dynamics (MD) or Monte Carlo (MC) simulations and relevant quantities are calculated and stored for each sampled conformation.
Estimators are then applied to these quantities to yield free energy differences between states and thus between the final and initial state, which equals the desired physical free energy difference.

## Core design principles

*alchemlyb* is a Python library that seeks to make doing alchemical free energy calculations easier and less error prone. 
It includes functionality for parsing data from file formats of widely used simulation packages, subsampling these data, and fitting these data with an estimator to obtain free energies. 
Functions are simple in usage and pure in scope, and can be chained together to build customized analyses of data while estimators are implemented as a classes that follow the tried-and-tested scikit-learn API.
General and robust workflows following best practices are also provided, which can be used as reference implementations and examples.

First and foremost, scientific code must be correct and we try to ensure this requirement by following best software engineering practices during development, close to full test coverage of all code in the library (currently 99%), and providing citations to published papers for included algorithms. 
We use a curated, public data set (*alchemtest* (https://github.com/alchemistry/alchemtest)) for automated testing.

The guiding design principles are summarized as:

1. Use functions when possible, classes only when necessary (or for estimators, see (2)).
2. For estimators, mimic the object-oriented scikit-learn API as much as possible.
3. Aim for a consistent interface throughout, e.g. all parsers take similar inputs and yield a common set of outputs, using the `pandas.DataFrame` as the underlying data structure.
4. Have *all* functionality tested.

*alchemlyb* is published under the open source BSD-3 clause license.

## Library structure

*alchemlyb* offers specific parsers in `alchemlyb.parsing` to load raw free energy data from various MD packages (GROMACS [@pronk2013gromacs], AMBER [@case2014ff14sb], NAMD [@phillips2020scalable], and GOMC [@cummings2021open]).
The raw data are converted into a standard format as a `pandas.DataFrame` and converted from the energy of the software to units of $k T$ where $k = 1.380649 \times 10^{-23}\,\text{J}\,\text{K}^{-1}$ is Boltzmann's constant and $T$ is the temperature at which the simulation was performed.
Metadata such as $T$ and the energy unit are stored in DataFrame attributes and propagated through *alchemlyb*, which enables seamless unit conversion with functions in the `alchemlyb.postprocessing` module.
Two types of free energy data are considered: Hamiltonian gradients (`dHdl`, $dH/d\lambda$) at all lambda states, suitable for thermodynamic integration (TI) estimators [@kirkwood1935statistical], and reduced potential energy differences between lambda states (`u_nk`, $u_{nk}$), which are used for free energy perturbation (FEP) estimators [@zwanzig1954high].

Both types of estimators assume uncorrelated samples, which requires subsampling of the raw data.
The `alchemlyb.preprocessing.subsampling` module provides tools for data subsampling based on autocorrelation times [@chodera2007use,@Chodera2016aa] as well as simple slicing of the `dHdl` and `u_nk` DataFrames.

The two major classes of commonly used estimators are implemented in `alchemlyb.estimators`.
Unlike other components  of *alchemlyb* that are implemented as pure functions, estimators are implemented as classes and follow the well-known scikit-learn API [@sklearn2013api] where instantiation sets the parameters (e.g., `estimator = MBAR(maximum_iterations=10000)`) and calling of the `fit()` method (e.g., `estimator.fit(u_nk)`) applies the estimator to the data and populates output attributes of the class; these results attributes are customarily indicated with a trailing underscore (e.g., `estimator.delta_f_` for the matrix of free energy differences between all states). 
In *alchemlyb*, TI [@paliwal2011benchmark] and TI with Gaussian quadrature [@gusev2023active] estimators are implemented in the TI category of estimators (module `alchemlyb.estimators.TI`).
FEP category estimators (module `alchemlyb.estimators.FEP`) include Bennett Acceptance Ratio (BAR) [@bennett1976efficient] and Multistate BAR (MBAR) [@shirts2008statistically], which are implemented in the `pymbar` package [@shirts2008statistically] (https://github.com/choderalab/pymbar) and called from *alchemlyb*.

To evaluate the accuracy of the free energy estimate, *alchemlyb* offers a range of assessment tools.
The error of the TI method is correlated with the average curvature [@pham2011identifying], while the error of FEP estimators depends on the overlap in sampled energy distributions [@pohorille2010good].
*alchemlyb*  the smoothness of the integrand for TI estimators and the overlap matrix for FEP estimators.
The accumulated samples should be collected from equilibrated simulations and *alchemlyb* contains tools for assessing (`alchemlyb.convergence`) and plotting (`alchemlyb.visualisation`) the convergence of the free energy estimate as a function of simulation time [@yang2004free] and means to compute the "fractional equilibration time" [@fan2020aa] to detect potentially un-equilibrated data.

*alchemlyb* offers all these tools as a library for users to customize each stage of the analysis (Figure 1).

![The building blocks of *alchemlyb*. Raw data from simulation packages are parsed into common data structures depending on the free energy quantities, pre-processed, and processed with a free energy estimator. The resulting free energy differences are analyzed for convergence and plotted for quality assessment.](Fig1.pdf)


## Workflows

The building blocks are sufficient to compute free energies from alchemical free energy simulations and assess their reliability.
*alchemlyb* also provides a structure to combined the building blocks into full end-to-end workflows (module `alchemlyb.workflows`).
As an example, the `ABFE` workflow for absolute binding free energy estimation reads in the raw input data and performs decorrelation, estimation, and quality plotting of the estimate.
It can directly estimate quantities such as solvation free energies and makes it easy to calculate more complex quantities such as absolute binding free energies (as the difference between the solvation free energy of the ligand in water and the solvation free energy of the ligand in the protein's binding pocket).



# Acknowledgements

Some work on alchemlyb was supported by grants from the  National Institutes of Health (Award No R01GM118772 to O.B.) and the National Science Foundation (award ACI-1443054 to O.B.).

# Author contributions

D.D. and O.B. designed the project. Z.W., D.D., contributed to the new features. Z.W., D.D., O.B. maintain the codebase. Z.W., M.R.S, O.B. wrote the manuscript.


# References


