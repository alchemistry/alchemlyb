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
    equal-contrib: true
    affiliation: "2, 3"
  - name: Irfan Alibay
    orcid: 0000-0001-5787-9130
    affiliation: 13
  - name: Jérôme Hénin
    orcid: 0000-0003-2540-4098
    affiliation: 4
  - name: Thomas T. Joseph
    orcid: 0000-0003-1323-3244
    affiliation: 5
  - name: Ian M. Kenney
    orcid: 0000-0002-9749-8866
    affiliation: 2
  - name: Hyungro Lee
    orcid: 0000-0002-4221-7094
    affiliation: 11
  - name: Haoxi Li
    orcid: 0009-0004-8369-1042
    affiliation: 6
  - name: Victoria Lim
    orcid: 0000-0003-4030-9312
    affiliation: 9
  - name: Domenico Marson
    orcid: 0000-0003-1839-9868
    affiliation: 7
  - name: Alexander Schlaich
    orcid: 0000-0002-4250-363X
    affiliation: 8
  - name: David Mobley
    orcid: 0000-0002-1083-5533
    affiliation: 9
  - name: Michael R. Shirts
    orcid: 0000-0003-3249-1097
    affiliation: 10
  - name: Oliver Beckstein
    orcid: 0000-0003-1340-0831
    corresponding: true
    affiliation: "2,12"
affiliations:
 - name: Exscientia, Oxford, UK
   index: 1
 - name: Department of Physics, Arizona State University, Tempe, Arizona, USA
   index: 2
 - name: Datryllic LLC, Phoenix, Arizona, USA (present affiliation)
   index: 3
 - name: Université Paris Cité, CNRS, Laboratoire de Biochimie Théorique, Paris, France
   index: 4
 - name: Department of Anesthesiology and Critical Care, Perelman School of Medicine, University of Pennsylvania, Philadelphia, Pennsylvania, USA
   index: 5
 - name: UNC Eshelman School of Pharmacy, University of North Carolina, Chapel Hill, NC, USA.
   index: 6
 - name: Molecular Biology and Nanotechnology Laboratory (MolBNL@UniTS), DEA, University of Trieste, Trieste, Italy
   index: 7
 - name: Stuttgart Center for Simulation Science (SC SimTech) & Institute for Computational Physics, University of Stuttgart, 70569 Stuttgart, Germany 
   index: 8
 - name: Departments of Pharmaceutical Sciences and Chemistry, University of California Irvine, Irvine, California, USA
   index: 9
 - name: University of Colorado Boulder, Boulder, Colorado, USA
   index: 10
 - name: Pacific Northwest National Laboratory, Richland, Washington, USA
   index: 11
 - name: Center for Biological Physics, Arizona State University, Tempe, AZ, USA
   index: 12
 - name: Open Free Energy, Open Molecular Software Foundation, Davis, 95616 California, United States
   index: 13

date: 31 May 2024
bibliography: paper.bib

---

# Summary

*alchemlyb* is an open-source Python software package for the analysis of alchemical free energy calculations, an important method in computational chemistry and biology, most notably in the field of drug discovery.
Its functionality contains individual composable building blocks for all aspects of a full typical free energy analysis workflow, starting with the extraction of raw data from the output of diverse molecular simulation packages, moving on to data preprocessing tasks such as decorrelation of time series, using various estimators to derive free energy estimates from simulation samples, and finally providing quality analysis tools for data convergence checking and visualization.
*alchemlyb* also contains high-level end-to-end workflows that combine multiple building blocks into a user-friendly analysis pipeline from the initial data input stage to the final results. This workflow functionality enhances accessibility by enabling researchers from diverse scientific backgrounds, and not solely computational chemistry specialists, to use *alchemlyb* effectively.


# Statement of need

In the pharmaceutical sector, computational chemistry techniques are integral for evaluating potential drug compounds based on their protein binding affinity [@deng2009computations].
Notably, absolute binding free energy calculations between proteins and ligands or relative binding affinity of ligands to the same protein are routinely employed for this purpose [@merz2010drug].
The resultant estimates of these free energies are essential for understanding binding affinity throughout various stages of drug discovery, such as hit identification and lead optimization [@merz2010drug].
Other free energies extracted from simulations are useful in solution thermodynamics, chemical engineering, environmental science, and material science [@Schlaich2015aa].

Molecular simulation packages such as [GROMACS](https://www.gromacs.org/) [@Abraham2015aa], [Amber](https://ambermd.org/) [@Case2005uq], [NAMD](https://www.ks.uiuc.edu/Research/namd/) [@phillips2020scalable], [LAMMPS](https://lammps.org/) [@Thompson2022aa], and [GOMC](https://gomc-wsu.org/) [@Nejahi2021aa] are used to run free energy simulations and many of these packages also contain tools for the subsequent processing of simulation data into free energies.
However, there are no standard output formats and analysis tools implement different algorithms for the different stages of the free energy data processing pipeline.
Therefore, it is very difficult to analyze data from different simulation packages in a consistent manner.
Furthermore, the native analysis tools do not always implement current best practices [@klimovich2015guidelines; @Mey2020aa] or are out of date.
Overall, the coupling between data generation and analysis in most simulation packages hinders seamless collaboration and comparison of results across different implementations of data generation for free energy calculations.

*alchemlyb* addresses this problem by focusing only on the data analysis portion of this process with the goal to provide a unified interface for working with free energy data generated from different software packages.
In an initial step data are read from the native package file formats and then organized into a common standard data structure, organized as a [*pandas*](https://pandas.pydata.org) `DataFrame` [@mckinney-proc-scipy-2010].
Functions are provided for pre-processing data by subsampling or decorrelation.
Statistical mechanical estimators are available to extract free energies and thermodynamic expectations as well associated metrics of quality; these estimators are implemented as classes with the same API as estimators in [scikit-learn](https://scikit-learn.org) [@scikitlearn2011; @sklearn2013api].
*alchemlyb* implements modular building blocks to simplify the process of extracting crucial thermodynamic insights from molecular simulations in a uniform manner.

*alchemlyb* succeeds the widely-used but now deprecated [`alchemical-analysis.py` tool](https://github.com/MobleyLab/alchemical-analysis) [@klimovich2015guidelines], which combined pre-processing, free energy estimation, and plotting in a single script.
`alchemical-analysis.py` was not thoroughly tested and hard to integrate into modern workflows due to its monolithic design, and only supported outdated Python 2.
*alchemlyb* improves over its predecessor with a modular, function based design and thorough testing of all components using continuous integration.
Thus, *alchemlyb* is a library that enables users to easily use well-tested building blocks within their own tools while additionally providing examples of complete end-to-end workflows.
This innovation enables consistent processing of free energy data from diverse simulation packages, facilitating streamlined comparison and combination of results.

Notably, *alchemlyb*'s robust and user-friendly nature has led to its integration into other automated workflow libraries such as BioSimSpace [@Hedges2019aa] or MDPOW [@fan2020aa], demonstrating its accessibility and usability within broader scientific workflows and reinforcing its position as a versatile tool in the field of computational chemistry.


# Implementation

Free energy differences are fundamental to understand many different processes at the molecular scale, ranging from the binding of drug molecules to their receptor proteins or nucleic acids through the partitioning of molecules into different solvents or phases to the stability of crystals and biomolecules.
The calculation of such transfer free energies involves constructing two end states where a target molecule interacts with different environments.
For example, in a solvation free energy calculation, at one state (the coupled state) it interacts with a solvent (in the case of hydration free energies, water), and the other where the ligand has no intermolecular interactions (the decoupled state), mimicking the transfer of a ligand at infinite dilution in the solvent at one end of the process and then ligand in the gas phase at the other.
The solvation free energy is then obtained by calculating the free energy difference between these two end states.
To achieve this, it is crucial to ensure sufficient overlap in phase space between the coupled and decoupled states, a condition often challenging to achieve.

Stratified alchemical free energy calculations have emerged as a de-facto standard approach whereby non-physical intermediate states are introduced to bridge between the physical end states of the process [@Mey2020aa].
In such free energy calculations, overlapping states are created by the introduction of a parameter $\lambda$ that continuously connects the functional form (the Hamiltonian of the system) of the two end-states, resulting in a series of intermediate states each with a different $\lambda$ value between 0 and 1 and with the physically realizable end states at $\lambda=0$ and $\lambda=1$.
In general, $N$ alchemical parameters are used to describe the alchemical transformation with a parameter vector $\vec{\lambda}=(\lambda_1, \lambda_2, \dots, \lambda_N)$, so that $\vec{\lambda}=(0, 0, \dots, 0)$ indicates the initial and $\vec{\lambda} = (1, 1, \dots, 1)$ the final state.
The intermediate states are non-physical but required for converging the calculations. 
At each $\vec{\lambda}$-value (or "window"), the system configurations are sampled in the relevant thermodynamic ensemble, typically using Molecular Dynamics (MD) or Monte Carlo (MC) simulations, while generating and accumulating free energy data discussed below.
Estimators are then applied to these data to compute free energy differences between states, including the difference between the final and initial state, thus yielding the desired free energy difference of the physical process of interest.

## Core design principles

*alchemlyb* is a Python library that seeks to make doing alchemical free energy calculations easier and less error prone. 
It includes functionality for parsing data from file formats of widely used simulation packages, subsampling these data, and fitting these data with an estimator to obtain free energies. 
Functions are simple in usage and pure in scope, and can be chained together to build customized analyses of data while estimators are implemented as classes that follow the tried-and-tested scikit-learn API [@sklearn2013api].
General and robust workflows following best practices are also provided, which can be used as reference implementations and examples.

First and foremost, scientific code must be correct and we try to ensure this requirement by following best software engineering practices during development, close to full test coverage of all code in the library (currently 99%), and providing citations to published papers for included algorithms. 
We use a curated, public data set ([*alchemtest*](https://github.com/alchemistry/alchemtest)) for automated testing; code in *alchemtest* is published under the open source BSD-3 clause license while all data are included under an [open license](https://opendefinition.org/licenses/#recommended-conformant-licenses) such as [CC0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain) or [CC-BY](http://opendefinition.org/licenses/cc-by/) (attribution required).

The guiding design principles are summarized as:

1. Use functions when possible, classes only when necessary (or for estimators, see (2)).
2. For estimators, mimic the object-oriented scikit-learn API as much as possible.
3. Aim for a consistent interface throughout, e.g. all parsers take similar inputs and yield a common set of outputs, using the `pandas.DataFrame` as the underlying data structure.
4. Have *all* functionality tested.

*alchemlyb* supports recent versions of Python 3 and follows the [SPEC 0 (Minimum Supported Dependencies)](https://scientific-python.org/specs/spec-0000/) Scientific Python Ecosystem Coordination community standard for deciding on when to drop support for older versions of Python and dependencies.
Releases are numbered following the [Semantic Versioning 2.0.0](https://semver.org/) standard of MAJOR.MINOR.PATCHLEVEL, which ensures that users immediately understand if a release may break backwards compatibility (increase of the major version), adds new features (increase of minor version), or only contains bug fixes or other changes that do not directly affect users.
All code is published under the open source BSD-3 clause license.

## Library structure

*alchemlyb* offers specific parsers in `alchemlyb.parsing` to load raw free energy data from various molecular simulation packages ([GROMACS](https://www.gromacs.org/) [@Abraham2015aa], [Amber](https://ambermd.org/) [@Case2005uq], [NAMD](https://www.ks.uiuc.edu/Research/namd/) [@phillips2020scalable], and [GOMC](https://gomc-wsu.org/) [@Nejahi2021aa]) and provides a general structure for implementing parsers for other packages that are not yet supported.
The raw data are converted into a standard format as a `pandas.DataFrame` [@mckinney-proc-scipy-2010] and converted from the energy of the software to units of $k T$ where $k = 1.380649 \times 10^{-23}\,\text{J}\,\text{K}^{-1}$ is Boltzmann's constant and $T$ is the temperature at which the simulation was performed.
Metadata such as $T$ and the energy unit are stored in `DataFrame` attributes and propagated through *alchemlyb*, which enables seamless unit conversion with functions in the `alchemlyb.postprocessing` module.
Two types of free energy data are considered: Hamiltonian gradients (`dHdl`, $dH/d\lambda$) at all lambda states, suitable for thermodynamic integration (TI) estimators [@kirkwood1935statistical], and reduced potential energy differences between lambda states (`u_nk`, $u_{nk}$), which are used for free energy perturbation (FEP) estimators [@zwanzig1954high].

Both types of estimators assume uncorrelated samples in order to give unbiased estimates of the uncertainties, which requires subsampling of the raw data.
The `alchemlyb.preprocessing.subsampling` module provides tools for data subsampling based on autocorrelation times [@chodera2007use; @Chodera2016aa] as well as simple slicing of the `dHdl` and `u_nk` DataFrames.

The two major classes of commonly used estimators are implemented in `alchemlyb.estimators`.
Unlike other components  of *alchemlyb* that are implemented as pure functions, estimators are implemented as classes and follow the well-known scikit-learn API [@sklearn2013api] where instantiation sets the parameters (e.g., `estimator = MBAR(maximum_iterations=10000)`) and calling of the `fit()` method (e.g., `estimator.fit(u_nk)`) applies the estimator to the data and populates output attributes of the class; these results attributes are customarily indicated with a trailing underscore (e.g., `estimator.delta_f_` for the matrix of free energy differences between all states). 
In *alchemlyb*, TI [@paliwal2011benchmark] and TI with Gaussian quadrature [@gusev2023active] estimators are implemented in the TI category of estimators (module `alchemlyb.estimators.TI`).
FEP category estimators (module `alchemlyb.estimators.FEP`) include Bennett Acceptance Ratio (BAR) [@bennett1976efficient] and Multistate BAR (MBAR) [@shirts2008statistically], which are implemented in the [*pymbar*](https://github.com/choderalab/pymbar) package [@shirts2008statistically] and called from *alchemlyb*.

To evaluate the accuracy of the free energy estimate, *alchemlyb* offers a range of assessment tools.
The error of the TI method is correlated with the average curvature [@pham2011identifying], while the error of FEP estimators depends on the overlap in sampled energy distributions [@pohorille2010good].
*alchemlyb* creates visualizations of the smoothness of the integrand for TI estimators and the overlap matrix for FEP estimators, which can be qualitatively and quantitatively analyzed to determine the degree of overlap between simulated alchemical states, and suggest whether additional simulations should be run.
For statistical validity, the accumulated samples should be collected from equilibrated simulations and *alchemlyb* contains tools for assessing (`alchemlyb.convergence`) and plotting (`alchemlyb.visualisation`) the convergence of the free energy estimate as a function of simulation time [@yang2004free] and means to compute the "fractional equilibration time" [@fan2020aa] to detect potentially un-equilibrated data.

*alchemlyb* offers all these tools as a library for users to customize each stage of the analysis (\autoref{fig:buildingblocks}).

![The building blocks of *alchemlyb*. Raw data from simulation packages are parsed into common data structures depending on the free energy quantities, pre-processed, and processed with a free energy estimator. The resulting free energy differences are analyzed for convergence and plotted for quality assessment.\label{fig:buildingblocks}](Fig1.pdf)


## Workflows

The building blocks are sufficient to compute free energies from alchemical free energy simulations and assess their reliability.
This functionality is used, for example, by the Streamlined Alchemical Free Energy Perturbation (SAFEP) analysis scripts [@Salari2018; @santiagomcrae2023].

*alchemlyb* also provides a structure to combine the building blocks into full end-to-end workflows (module `alchemlyb.workflows`).
As an example, the `ABFE` workflow for absolute binding free energy estimation reads in the raw input data and performs decorrelation, estimation, and quality plotting of the estimate.
It can directly estimate quantities such as solvation free energies and makes it easy to calculate more complex quantities such as absolute binding free energies (as the difference between the solvation free energy of the ligand in water and the solvation free energy of the ligand in the protein's binding pocket).


# Acknowledgements

Some work on alchemlyb was supported by grants from the  National
Institutes of Health (Award No R01GM118772 to O.B., R35GM148236 to
D.M., K08GM139031 to T.T.J.) and the National Science Foundation (award ACI-1443054 to O.B.). A.S. acknowledges funding by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy - EXC 2075 – 390740016 and support by the Stuttgart Center for Simulation Science (SimTech).
The sponsors were not involved in any aspects of the research or the writing of the manuscript.

Shuai Liu, Travis Jensen, Bryce Allen, Dominik Wille, Mohammad S. Barhaghi, and Pascal Merz contributed code to *alchemlyb*.

# Author contributions

D.L.D., M.R.S., D.M., and O.B. designed the project. Z.W., D.L.D., I.A., J.H., T.T.J., I.M.K., H.L., H.L., V.L., D.M., A.S. contributed to new features. Z.W., D.L.D., O.B. maintained the code base. Z.W., D.L.D., M.R.S, A.S., O.B. wrote the manuscript.

# References


