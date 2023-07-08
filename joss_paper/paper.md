---
title: 'Alchemlyb: The Simple Alchemistry Library'
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
  - name: David Dotson
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
affiliations:
 - name: Exscientia, Oxford, UK.
   index: 1
 - name: Institution Name, Country
   index: 2

date: 99 August 2023
bibliography: paper.bib

---

# Summary

alchemlyb is a dedicated open-source software package tailored for the analysis of alchemical free energy calculations, an integral part of computational chemistry and biology, notably in the field of drug discovery. The software spans a wide range of functions, starting with the extraction of raw data from Molecular Dynamics (MD) engines, moving on to data preprocessing tasks such as decorrelation, using various estimators to derive free energy estimates, and finally providing quality analysis tools for data convergence checking.

A distinctive attribute of alchemlyb is its streamlined, end-to-end analysis process reminiscent of the now-discontinued alchemical analysis workflow. This user-friendly workflow facilitates navigation through the entire analysis pipeline, from the initial data input stage to the final result derivation, enabling researchers from diverse scientific backgrounds, and not solely computational chemistry specialists, to utilize alchemlyb effectively.

# Statement of need

In the pharmaceutical sector, computational chemistry techniques, particularly relative/absolute binding free energy calculations, are regularly employed to rank potential drug compounds based on their protein-binding affinity. These calculations produce free energy data, which alchemlyb expertly processes to offer free energy estimates. These estimates provide critical insights into the binding affinity at various stages of drug discovery, such as hit identification and lead optimization. alchemlyb's unique capacity to cater to this need has cemented its role as an invaluable asset in computational chemistry.

Moreover, within the realm of computational research, different MD engines, including GROMACS, AMBER, OpenMM, and NAMD, have their distinct sets of tools for conducting free energy calculations. This diversity complicates the research process, as data from different engines necessitate unique processing and analysis methods.

The solution to this complication comes in the form of alchemlyb, providing a unified, engine-agnostic analysis workflow. This allows for consistent analysis of free energy data from different MD engines, making it possible for researchers to compare and combine results from various engines in a more streamlined manner.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from XXXXX during the genesis of this project.

# References