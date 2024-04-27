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

date: 24 April 2024
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
Furthermore, the native analysis tools do not always implement current best practices [@klimovich2015guidelines,@mey2020bestpractices] or are out of date.
Overall, the lack of coupling between data generation and analysis in most MD packages hinders seamless collaboration and comparison of results across different implementations of data generation for free energy calculations.

*alchemlyb* addresses this problem by focusing only the data analysis portion of this process with the goal to provide a unified interface for working with free energy data generated from different MD packages.
In an initial step, data are read from the native MD package file formats and then organized into a common standard data structure, organized as a pandas Dataframe.
Additional functions enable subsampling or decorrelation of data and applying statistical mechanical estimators to extract the free energies and thermodynamic expectations as well associated metrics of quality.
*alchemlyb* implements these workflows using modular building blocks to simplify the process of extracting crucial thermodynamic insights from molecular simulations in a uniform manner.

*alchemlyb* succeeds the widely-used but now deprecated [alchemical-analysis.py](https://github.com/MobleyLab/alchemical-analysis) tool [@klimovich2015guidelines], which combined pre-processing, free energy estimation, and plotting in a single script. 
`alchemical-analysis.py` was not thoroughly tested and hard to integrate into modern workflows due to its monolithic design, as well as remaining in python 2. 
*alchemlyb* improves over its predecessor with a modular, function based design and thorough testing of all components using continuous integration.
Thus, *alchemlyb* is a library that enables users to easily use well-tested building blocks with in their own tools while additionally providing examples of complete end-to-end workflows.
This innovation enables consistent processing of free energy data from diverse MD packages, facilitating streamlined comparison and combination of results.

Notably, *alchemlyb*'s robust and user-friendly nature has led to its integration into other automated workflow libraries such as BioSimSpace [@hedges2023suite].
This further enhances its accessibility and usability within broader scientific workflows, reinforcing its position as a versatile and essential tool in the field of computational chemistry.


# Implementation

Transfer free energies, key physical property often computed by computational chemists, involves constructing two end states where a target molecule interacts with different environments  For example, in a solvation free energy calculation, at one state (the coupled state) it interacts with a solvent (in the case hydration free energies, water), and the other where the ligand has no intermolecular interactions (the decoupled state), mimicking the transfer of a ligand at infinite dilution in the solvent at one end of the process and then ligand in the gas phase at the other.
The solvation free energy is then obtained by calculating the free energy difference between these two end states.
To achieve this, it is crucial to ensure sufficient overlap in phase space between the coupled and decoupled states, a condition often challenging to achieve.
The creation of overlapping states is facilitated by introducing a parameter `lambda` ($\lambda$) that continuously connects the functional form of the two end-states, resulting in a series of intermediate states each with a  $\lambda$ value ranging from 0 to 1 (inclusive) are simulated. MD engines simulate the system at these states at these intermediate alchemical states, generating and accumulating free energy data discussed below.

*alchemlyb* offers specific parsers designed to load raw free energy data from various MD engines, converting them into standard `pandas` `DataFrames`.
Two types of free energy data are considered: Hamiltonian gradients (`dHdl`, $dH/d\lambda$) at all lambda states, suitable for thermodynamic integration (TI) estimators [@kirkwood1935statistical], and reduced potential energy differences between lambda states (`u_nk`, $u_{nk}$), which are used for free energy perturbation (FEP) estimators [@zwanzig1954high].

In *alchemlyb*, TI [@paliwal2011benchmark] and TI with Gaussian quadrature [@gusev2023active] estimators are implemented in the TI category of estimators.
FEP category estimators include Bennett Acceptance Ratio (BAR) [@bennett1976efficient] and Multistate BAR (MBAR) [@shirts2008statistically].
These estimators assume uncorrelated samples in order to give unbiased estimates of the uncertainties, and *alchemlyb* provides tools for data resampling based on autocorrelation times [@chodera2007use].

To evaluate the accuracy of the free energy estimate, *alchemlyb* offers a range of assessment tools.
The error of the TI method is correlated with the average curvature [@pham2011identifying], while the error of FEP estimators depends on the overlap in sampled energy distributions [@pohorille2010good].
*alchemlyb* creates visualizations of the smoothness of the integrand for TI estimators and the overlap matrix for FEP estimators, which can be qualitatively and quantitatively analyzed to determine the degree of overlap between simulated alchemical states, and suggest whether additional simulations should be run. 
For statistical validity, the accumulated samples should be collected from equilibrated simulations, an *alchemlyb* thus also has tools for plotting the convergence of the free energy estimate as a function of simulation time [@yang2004free] to detect the presence of potentially un-equilibrated data.

*alchemlyb* offers all these tools as a library for users to customize each stage of the analysis (Figure 1).
Additionally, *alchemlyb* provides an automated end-to-end workflow that carries out all stages of the analysing, reading in the raw input data and performs decorrelation, estimation, and quality plotting of the estimates.
This workflow allows for the estimation of quantities such as solvation free energy with minimal code.
Moreover, this facilitates more complex calculations, such as absolute binding free energy, which is the free energy difference between the solvation free energy of the ligand in water and the solvation free energy of the ligand in the protein's binding pocket. 

![The building blocks of *alchemlyb*](Fig1.pdf)


# Acknowledgements

Some work on alchemlyb was supported by grants from the  National Institutes of Health (Award No R01GM118772 to O.B.) and the National Science Foundation (award ACI-1443054 to O.B.).

# Author contributions

D.D. and O.B. designed the project. Z.W., D.D., contributed to the new features. Z.W., D.D., O.B. maintain the codebase. Z.W., M.R.S, O.B. wrote the manuscript.


# References


