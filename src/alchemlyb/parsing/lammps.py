""" Parsers for extracting alchemical data from [LAMMPS](https://docs.lammps.org/Manual.html) output files.

For clarity, we would like to distinguish the difference between $\lambda$ and $\lambda'$. We refer to $\lambda$ as 
the potential scaling of the equilibrated system, so that when this value is changed, the system undergoes another equilibration 
step. One the otherhand, $\lambda'$ is the value used to scaled the potentials for the configurations of the system equilibrated 
for $\lambda$. The value of $\lambda'$ is used in two instances. First, in thermodynamic integration (TI), values of $\lambda'$ 
that are very close to $\lambda$ can be used to calculate the derivative. This is needed because LAMMPS does not compute 
explicit derivatives, although one should check whether they can derive an explicit expression, they cannot for changes of 
$\lambda'$ in the soft Lennard-Jones (LJ) potential.

The parsers featured in this module are constructed to parse LAMMPS output files output using the 
[`fix ave/time command`](https://docs.lammps.org/fix_ave_time.html), containing data for given potential energy values (an 
approximation of the Hamiltonian) at specified values of $\lambda$ and $\lambda'$, $U_{\lambda,\lambda'}$. Because generating 
the input files can be combersome, functions have been included to generate the appropriate sections. If a linear approximation 
can be made to calculate $U_{\lambda,\lambda'}$ from $U_{\lambda}$ in post-processing, we recommend using 
:func:`alchemlyb.parsing.generate_input_linear_approximation()`. If a linear approximation cannot be made (such as changing 
$\lambda$ in the soft-LJ potential) we recommend running a loop over all values of $\lambda$ saving frames spaced to be 
independent samples, and an output file with small perturbations with $\lambda'$ to calculate the derivative for TI in 
post-processing. This is achieved with `alchemlyb.parsing.generate_traj_input()`. After this first simulation, we then 
recommend the files needed for MBAR are generated using the [rerun](https://docs.lammps.org/rerun.html) feature in LAMMPS. 
Breaking up the computation like this will allow one to add additional points to their MBAR analysis without repeating the 
points from an initial simulation. Generating the file for a rerun is achieved with 
:func:`alchemlyb.parsing.generate_rerun_mbar()`. Notice that the output files do not contain the header information expected 
in LAMMPS as that is system specific and left to the user.

Note that in LAMMPS, [fix adapt/fep](https://docs.lammps.org/fix_adapt_fep.html) changes $\lambda$ and 
[compute fep](https://docs.lammps.org/compute_fep.html) changes $\lambda'$.

.. versionadded:: 1.0.0

"""

import os
import warnings
import numpy as np
import pandas as pd
import glob

from . import _init_attrs
from ..postprocessors.units import R_kJmol, kJ2kcal

k_b = R_kJmol * kJ2kcal


def _isfloat(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def generate_input_linear_approximation(
    parameter,
    parameter_range,
    parameter_change,
    pair_style,
    types_solvent,
    types_solute,
    output_file=None,
    parameter2=None,
    parameter2_value=None,
    pair_style2=None,
):
    """Outputs the section of a LAMMPS input file that separates the Coulomb, nonbonded, and bond/angle/torsional contributions
    of the solute and solvent. As long as the parameter being changed is linearly dependent on the potential energy, these files for
    each value of the parameter can be used for thermodynamic integration (TI) or multi-state Bennett acceptance ratio (MBAR).

    The input data file for this script should be an equilibrated frame in the NPT ensemble. Notice that the input file contains
    the following keywords that you might replace with the values for your simulation using `sed`: TEMP, PRESS

    Parameters
    ----------
    parameter : str
        Parameter being varied, see table in `compute fep <https://docs.lammps.org/compute_fep.html>`_ for the options in
        your pair-potential
    parameter_range : list[float]
        Range of parameter values to be changed where the first value should be the value with which the system has been
        equilibrated.
    parameter_change : float
        The size of the step between parameter values. Take care that number of points needed to traverse the given range
        should result in an integer, otherwise LAMMPS will not end at the desired value.
    pair_style : str
        String with LAMMPS pair style being altered
    types_solvent : str
        String defining atom types in the solvent (with no spaces, e.g., *4)
    types_solute : str
        String defining atom types in the solute (with no spaces, e.g., 5*9)
    output_file : str, default=None
        File name and path for optional output file
    parameter2 : str, default=None
        Parameter that has been varied and is set to another value in this simulation, e.g., lambda when the Coulomb potential
        is set to zero. Using this feature avoids complications with writing the pair potential information in the data file.
        See table in `compute fep <https://docs.lammps.org/compute_fep.html>`_ for the options in your pair-potential
    pair_style2 : str, default=None
        String with LAMMPS pair style for ``parameter2``
    parameter2_value : float, default=None
        Value to set ``parameter2``

    Returns
    -------
    file : list[str]
        List of strings representing lines in a file

    """
    nblocks = (parameter_range[1] - parameter_range[0]) / parameter_change
    if nblocks % 1 > 0:
        raise ValueError(
            "The number of steps needed to traverse the parameter range, {}, with a step size of, {} is not an integer".format(
                parameter_range, parameter_change
            )
        )
    else:
        nblocks = int(nblocks)

    if any(
        [x is not None for x in [parameter2, pair_style2, parameter2_value]]
    ) and not all([x is not None for x in [parameter2, pair_style2, parameter2_value]]):
        raise ValueError(
            (
                f"If any values for 'parameter2' are provided, all must be provided: parameter2={parameter2}, "
                + f"parameter2_value={parameter2_value}, pair_style2={pair_style2}"
            )
        )
    name1 = "-".join([pair_style.replace("/", "-"), parameter])
    file = [
        "\n# Variables and System Conditions\n",
        "variable freq equal 1000 # Consider changing\n",
        "variable runtime equal 1000000\n",
        f"variable delta equal {parameter_change} \n",
        f"variable nblocks equal {nblocks} \n",
        f"variable paramstart equal {parameter_range[0]}\n",
        "variable TK equal TEMP\n",
        "variable PBAR equal PRESS\n",
        "fix 1 all npt temp ${TK} ${TK} 1.0 iso ${PBAR} ${PBAR} # Change dampening factors according to your system\n",
        "thermo ${freq}\n",
        "\n# Group atoms\n",
        f"group solute type {types_solute}\n",
        f"group solvent type {types_solvent}\n",
        "\n# Set-up Loop\n",
        "variable runid loop 0 ${nblocks} pad\n",
        "    label runloop1\n",
        "\n# Adjust param for the box and equilibrate\n",
        "    variable param equal v_paramstart-v_runid*v_delta\n",
        '    if "${runid} == 0" then &\n',
        '        "jump SELF skipequil"\n',
        "    variable param0 equal v_paramstart-(v_runid-1)*v_delta\n",
        "    variable paramramp equal ramp(v_param0,v_param)\n",
        "    fix ADAPT all adapt/fep ${freq} &\n",
        f"        pair {pair_style} {parameter} {types_solute} {types_solvent} v_paramramp\n",
        "    thermo_style custom v_vstep v_time v_paramramp temp press pe evdwl enthalpy\n",
        "    run ${runtime} # Run Ramp\n",
        "    thermo_style custom v_vstep v_time v_param temp press pe evdwl enthalpy\n",
        "    run ${runtime} # Run Equil\n",
        "\n    label skipequil\n\n",
        f"    write_data files/npt_{name1}_" + "${param}.data\n",
        "\n    # Initialize computes\n",
        "    ## Compute PE for contributions for bonds, angles, dihedrals, and impropers\n",
        "    compute pe_solute_bond solute pe/atom bond angle dihedral improper # PE from nonpair/noncharged intramolecular interactions\n",
        "    compute pe_solute_1 solute reduce sum c_pe_solute_bond\n",
        "    compute pe_solvent_bond solvent pe/atom bond angle dihedral improper # PE from nonpair/noncharged intramolecular interactions\n",
        "    compute pe_solvent_1 solvent reduce sum c_pe_solvent_bond\n",
        "\n    ## Compute PE for contributions for pair and charges\n",
        "    compute pe_solute_2 solute group/group solute pair yes kspace no\n",
        "    compute pe_solute_3 solute group/group solute pair no kspace yes\n",
        "    compute pe_solvent_2 solvent group/group solvent pair yes kspace no\n",
        "    compute pe_solvent_3 solvent group/group solvent pair no kspace yes\n",
        "    compute pe_inter_2 solute group/group solvent pair yes kspace no\n",
        "    compute pe_inter_3 solute group/group solvent pair no kspace yes\n",
        "    thermo_style custom v_vstep v_time v_param temp press pe evdwl enthalpy &\n",
        "        c_pe_solute_1 c_pe_solute_2 c_pe_solute_3 c_pe_solvent_1 c_pe_solvent_2 c_pe_solvent_3 c_pe_inter_2 c_pe_inter_3\n",
        "    fix FEPout all ave/time ${freq} 1 ${freq} v_vstep v_time v_param v_tinst v_pinst v_pe v_evdwl v_enthalpy &\n",
        "        c_pe_solute_1 c_pe_solute_2 c_pe_solute_3 c_pe_solvent_1 c_pe_solvent_2 c_pe_solvent_3 c_pe_inter_2 c_pe_inter_3 &\n",
        f"        file files/linear_{name1}_" + "${param}.txt\n",
        "\n    run ${runtime}\n\n",
        "    uncompute pe_solute_bond\n",
        "    uncompute pe_solute_1\n",
        "    uncompute pe_solvent_bond\n",
        "    uncompute pe_solvent_1\n",
        "    uncompute pe_solute_2\n",
        "    uncompute pe_solute_3\n",
        "    uncompute pe_solvent_2\n",
        "    uncompute pe_solvent_3\n",
        "    uncompute pe_inter_2\n",
        "    uncompute pe_inter_3\n",
        '    if "${runid} != 0" then &\n',
        '        "unfix ADAPT"\n',
        "    unfix FEPout\n",
        "\n    next runid\n",
        "    jump SELF runloop1\n",
        "write_data npt.data nocoeff\n",
    ]

    if parameter2 is not None:
        name2 = "-".join([pair_style2.replace("/", "-"), parameter2])
        file2 = [
            "\n# Set Previous Change\n",
            f"variable param2 equal {parameter2_value}\n",
            "fix ADAPT2 all adapt/fep 1 &\n",
            f"    pair {pair_style2} {parameter2} {types_solute} {types_solvent} v_param2\n",
        ]
        file[13:13] = file2
        file[-1:-1] = "unfix ADAPT2\n"
        ind = [ii for ii, x in enumerate(file) if "fix FEPout" in x][0]
        file[ind] = (
            "    fix FEPout all ave/time ${freq} 1 ${freq} v_vstep v_time v_param v_param2 v_tinst v_pinst v_pe v_evdwl v_enthalpy &\n"
        )
        file[ind + 2] = (
            f"        file files/linear_{name1}_"
            + "${param}_"
            + f"{name2}_{parameter2_value}.txt\n"
        )
        ind = [ii for ii, x in enumerate(file) if "write_data files/npt" in x][0]
        file[ind] = (
            f"    write_data files/npt_{name1}_"
            + "${param}_"
            + f"{name2}_{parameter2_value}.data\n"
        )

    if output_file is not None:
        with open(output_file, "w") as f:
            for line in file:
                f.write(line)

    return file


def generate_traj_input(
    parameter,
    parameter_range,
    parameter_change,
    pair_style,
    types_solvent,
    types_solute,
    del_parameter=0.01,
    output_file=None,
    parameter2=None,
    parameter2_value=None,
    pair_style2=None,
    del_parameter2=None,
):
    """Outputs the section of a LAMMPS input file that loops over the values of parameter being changed (e.g., lambda)
    Small perturbations in the potential energy are also output so that the derivative can be calculated for thermodynamic
    integration. Trajectories are produces so that files for MBAR analysis may be generated in post-processing.

    The input data file for this script should be an equilibrated frame in the NPT ensemble. Notice that the input file contains
    the following keywords that you might replace with the values for your simulation using `sed`: TEMP, PRESS

    Parameters
    ----------
    parameter : str
        Parameter being varied, see table in `compute fep <https://docs.lammps.org/compute_fep.html>`_ for the options in
        your pair-potential
    parameter_range : list[float]
        Range of parameter values to be changed where the first value should be the value with which the system has been
        equilibrated.
    parameter_change : float
        The size of the step between parameter values. Take care that number of points needed to traverse the given range
        should result in an integer, otherwise LAMMPS will not end at the desired value.
    pair_style : str
        String of LAMMPS pair style being changes
    types_solvent : str
        String defining atom types in the solvent (not spaces)
    types_solute : str
        String defining atom types in the solute (not spaces)
    del_parameter : float, default=0.1
        Change used to calculate the forward and backward difference used to compute the derivative through a central difference
        approximation.
    output_file : str, default=None
        File name and path for optional output file
    parameter2 : str, default=None
        Parameter that has been varied and is set to another value in this simulation, e.g., lambda when the Coulomb potential
        is set to zero. Using this feature avoids complications with writing the pair potential information in the data file.
        See table in `compute fep <https://docs.lammps.org/compute_fep.html>`_ for the options in your pair-potential
    pair_style2 : str, default=None
        String with LAMMPS pair style being set for ``parameter2``
    parameter2_value : float, default=None
        Value to set ``parameter2``
    del_parameter2 : float, default=None
        Change used to calculate the forward and backward difference used to compute the derivative through a central difference
        approximation for parameter2.

    Returns
    -------
    file : list[str]
        List of strings representing lines in a file

    """
    nblocks = (parameter_range[1] - parameter_range[0]) / parameter_change
    if nblocks % 1 > 0:
        raise ValueError(
            f"The number of steps needed to traverse the parameter range, {parameter_range}, with a step size of, {parameter_change} is not an integer"
        )
    else:
        nblocks = int(nblocks)

    if any(
        [
            x is not None
            for x in [parameter2, pair_style2, parameter2_value, del_parameter2]
        ]
    ) and not all(
        [
            x is not None
            for x in [parameter2, pair_style2, parameter2_value, del_parameter2]
        ]
    ):
        raise ValueError(
            (
                f"If any values for 'parameter2' are provided, all must be provided: parameter2={parameter2}, "
                + f"parameter2_value={parameter2_value}, pair_style2={pair_style2}, del_parameter2={del_parameter2}"
            )
        )
    name1 = "-".join([pair_style.replace("/", "-"), parameter])
    file = [
        "\n# Variables and System Conditions\n",
        "variable freq equal 1000 # Consider changing\n",
        "variable runtime equal 1000000\n",
        f"variable delta equal {parameter_change} \n",
        f"variable nblocks equal {nblocks} \n",
        f"variable deltacdm equal {del_parameter} # delta used in central different method for derivative in TI\n",
        f"variable paramstart equal {parameter_range[0]}\n",
        "variable TK equal TEMP\n",
        "variable PBAR equal PRESS\n",
        "fix 1 all npt temp ${TK} ${TK} 1.0 iso ${PBAR} ${PBAR} # Change dampening factors according to your system\n",
        "thermo ${freq}\n",
        "\n# Set-up Loop\n",
        "variable nblocks equal 1/v_delta",
        "variable runid loop 0 ${nblocks} pad\n",
        "    label runloop1\n",
        "\n# Adjust param for the box and equilibrate\n",
        "    variable param equal v_paramstart-v_runid*v_delta\n",
        '    if "${runid} == 0" then &\n',
        '        "jump SELF skipequil"\n',
        "    variable param0 equal v_paramstart-(v_runid-1)*v_delta\n",
        "    variable paramramp equal ramp(v_param0,v_param)\n",
        "    fix ADAPT all adapt/fep ${freq} &\n",
        f"        pair {pair_style} {parameter} {types_solute} {types_solvent} v_paramramp\n",
        "    thermo_style custom v_vstep v_time v_paramramp temp press pe evdwl enthalpy\n",
        "    run ${runtime} # Run Ramp\n",
        "    thermo_style custom v_vstep v_time v_param temp press pe evdwl enthalpy\n",
        "    run ${runtime} # Run Equil\n",
        "\n    label skipequil\n\n",
        f"    write_data files/npt_{name1}_" + "${param}.data\n",
        "\n    # Initialize computes\n",
        "    thermo_style custom v_vstep v_time v_param temp press pe evdwl enthalpy\n",
        "    variable deltacdm2 equal -v_deltacdm\n",
        "    compute FEPdb all fep ${TK} &\n",
        f"        pair {pair_style} {parameter} {types_solute} {types_solvent} v_deltacdm2\n",
        "    compute FEPdf all fep ${TK} &\n",
        f"        pair {pair_style} {parameter} {types_solute} {types_solvent} v_deltacdm\n",
        "    fix FEPout all ave/time ${freq} 1 ${freq} v_vstep v_time v_param v_deltacdm v_tinst v_pinst v_pe v_evdwl v_enthalpy &\n",
        f"        c_FEPdb[1] c_FEPdf[1] file files/ti_{name1}_" + "${param}.txt\n",
        "\n    dump TRAJ all custom ${freq} "
        + f"files/dump_{name1}_"
        + "${param}.lammpstrj id mol type element xu yu zu\n",
        "\n    run ${runtime}\n\n",
        "    uncompute FEPdb\n",
        "    uncompute FEPdf\n",
        '    if "${runid} != 0" then &\n',
        '        "unfix ADAPT"\n',
        "    unfix FEPout\n",
        "    undump TRAJ\n",
        "\n    next runid\n",
        "    jump SELF runloop1\n",
        "write_data npt.data nocoeff\n",
    ]

    if parameter2 is not None:
        name2 = "-".join([pair_style2.replace("/", "-"), parameter2])
        file[6:6] = (f"variable delta2cdm equal {del_parameter2}\n",)
        file2 = [
            "\n# Set Previous Change\n",
            f"variable param2 equal {parameter2_value}\n",
            "fix ADAPT2 all adapt/fep 1 &\n",
            f"    pair {pair_style2} {parameter2} {types_solute} {types_solvent} v_param2\n",
            "variable delta2cdm2 equal -v_delta2cdm\n",
            "compute FEP2db all fep ${TK} &\n",
            f"    pair {pair_style2} {parameter2} {types_solute} {types_solvent} v_delta2cdm2\n",
            "compute FEP2df all fep ${TK} &\n",
            f"    pair {pair_style2} {parameter2} {types_solute} {types_solvent} v_delta2cdm\n",
        ]
        file[11:11] = file2
        file[-1:-1] = "unfix ADAPT2\n"
        file[-1:-1] = "uncompute FEP2db\n"
        file[-1:-1] = "uncompute FEP2df\n"
        ind = [ii for ii, x in enumerate(file) if "write_data files/npt" in x][0]
        file[ind] = (
            f"    write_data files/npt_{name1}_"
            + "${param}_"
            + f"{name2}_{parameter2_value}.data\n"
        )
        ind = [ii for ii, x in enumerate(file) if "fix FEPout" in x][0]
        file[ind] = (
            "    fix FEPout all ave/time ${freq} 1 ${freq} v_vstep v_time v_param v_deltacdm v_param2 v_delta2cdm v_tinst v_pinst v_pe v_evdwl v_enthalpy &\n"
        )
        file[ind + 1] = (
            f"        c_FEPdb[1] c_FEPdf[1] c_FEP2db[1] c_FEP2df[1] file files/ti_{name1}_"
            + "${param}_"
            + f"{name2}_{parameter2_value}.txt\n"
        )
        file[ind + 2] = (
            "\n    dump TRAJ all custom ${freq} "
            + f"files/dump_{name1}_"
            + "${param}_"
            + f"{name2}_{parameter2_value}.lammpstrj id mol type element xu yu zu\n"
        )

    if output_file is not None:
        with open(output_file, "w") as f:
            for line in file:
                f.write(line)

    return file


def generate_rerun_mbar(
    parameter_value,
    parameter,
    parameter_range,
    parameter_change,
    pair_style,
    types_solvent,
    types_solute,
    output_file=None,
    parameter2=None,
    pair_style2=None,
    parameter2_value=None,
):
    """Outputs the section of a LAMMPS input file that reruns trajectories for different lambda values and calculates
    the potential energy for all other lambda values with this set of configurations.

    Parameters
    ----------
    parameter_value : float
        Value of parameter being varied (e.g., lambda)
    parameter : str
        Parameter being varied, see table in `compute fep <https://docs.lammps.org/compute_fep.html>`_ for the options in
        your pair-potential
    parameter_range : list[float]
        Range of parameter values to be changed where the first value should be the value with which the system has been
        equilibrated.
    parameter_change : float
        The size of the step between parameter values. Take care that number of points needed to traverse the given range
        should result in an integer, otherwise lammps will not end at the desired value.
    pair_style : str
        String of LAMMPS pair style being changes
    types_solvent : str
        String defining atom types in the solvent (not spaces)
    types_solute : str
        String defining atom types in the solute (not spaces)
    output_file : str, default=None
        File name and path for optional output file
    parameter2 : str, default=None
        Parameter that has been varied and is set to another value in this simulation, e.g., lambda for the coulombic potential
        is set to zero. Using this feature avoids complicaitons with writing the pair potential information in the data file.
        See table in `compute fep <https://docs.lammps.org/compute_fep.html>`_ for the options in your pair-potential
    pair_style2 : str, default=None
        String with LAMMPS pair style being set for ``parameter2``
    parameter2_value : float, default=None
        Value to set ``parameter2``

    Returns
    -------
    file : list[str]
        List of strings representing lines in a file

    """
    nblocks = (parameter_range[1] - parameter_range[0]) / parameter_change
    if nblocks % 1 > 0:
        raise ValueError(
            "The number of steps needed to traverse the parameter range, {}, with a step size of, {} is not an integer".format(
                parameter_range, parameter_change
            )
        )
    else:
        nblocks = int(nblocks)

    if any(
        [x is not None for x in [parameter2, pair_style2, parameter2_value]]
    ) and not all([x is not None for x in [parameter2, pair_style2, parameter2_value]]):
        raise ValueError(
            (
                f"If any values for 'parameter2' are provided, all must be provided: parameter2={parameter2}, "
                + f"parameter2_value={parameter2_value}, pair_style2={pair_style2}"
            )
        )

    if np.isclose(parameter_range[0], 0):
        prec = int(np.abs(int(np.log10(np.abs(parameter_change)))))
    else:
        prec = max(
            int(np.abs(int(np.log10(np.abs(parameter_range[0]))))),
            int(np.abs(int(np.log10(np.abs(parameter_change))) + 1)),
        )
    name1 = "-".join([pair_style.replace("/", "-"), parameter])
    file = [
        "\n# Variables and System Conditions\n",
        f"variable param equal {parameter_value}\n",
        "variable freq equal 1000 # Consider changing\n",
        "variable runtime equal 1000000\n",
        f"variable delta equal {parameter_change}\n",
        "variable TK equal TEMP\n",
        "\nthermo ${freq}\n",
        f"read_data files/npt_{name1}_" + "${param}.data\n",
        "\n# Initialize computes\n",
    ]
    if parameter2 is not None:
        file2 = [
            "\n# Set Previous Change\n",
            "variable param2 equal {parameter2_value}\n",
            "fix ADAPT2 all adapt/fep 1 &\n",
            f"    pair {pair_style2} {parameter2} {types_solute} {types_solvent} v_param2\n",
        ]
        file[8:8] = file2
        name2 = "-".join([pair_style2.replace("/", "-"), parameter2])
        ind = [ii for ii, x in enumerate(file) if "read_data files/npt" in x][0]
        file[ind] = (
            f"read_data files/npt_{name1}_"
            + "${param}_"
            + f"{name2}_{parameter2_value}.data\n"
        )

    for i in range(nblocks):
        value2 = parameter_range[0] + parameter_change * i
        delta = value2 - parameter_value
        tmp = "variable delta{0:0d}".format(i) + " equal {0:." + str(prec) + "f}\n"
        tmp = [
            tmp.format(delta),
            "compute FEP{0:03d} all fep ".format(i) + "${TK} &\n",
            f"    pair {pair_style} {parameter} {types_solute} {types_solvent} v_delta{i}\n",
            "variable param{0:03d} equal v_param+v_delta{0:0d}\n".format(i),
            "fix FEPout{0:03d} all".format(i)
            + " ave/time ${freq} 1 ${freq} "
            + "v_time v_param v_param{0:03d} &\n".format(i),
            "    c_FEP{0:03d}[1] c_FEP{0:03d}[2] c_FEP{0:03d}[3]".format(i)
            + f" file files/mbar_{name1}"
            + "_${param}_${param"
            + str("{0:03d}".format(i))
            + "}.txt\n\n",
        ]
        if parameter2 is not None:
            ind = [ii for ii, x in enumerate(tmp) if "fix FEPout" in x][0]
            tmp[ind : ind + 2] = [
                "fix FEPout{0:03d} all".format(i)
                + " ave/time ${freq} 1 ${freq} "
                + "v_time v_param v_param{0:03d} v_param2 &\n".format(i),
                "    c_FEP{0:03d}[1] c_FEP{0:03d}[2] c_FEP{0:03d}[3]".format(i)
                + f" file files/mbar_{name1}"
                + "_${param}_${param"
                + str("{0:03d}".format(i))
                + "}_"
                + "{}_{}.txt\n\n".format(name2, parameter2_value),
            ]
        file.extend(tmp)

    if parameter2 is not None:
        file.append(
            f"\nrerun files/dump_{name1}_"
            + "${param}_"
            + f"{name2}_{parameter2_value}.lammpstrj "
            + "every ${freq} dump xu yu zu\n\n"
        )
    else:
        file.append(
            f"\nrerun files/dump_{name1}"
            + "_${param}.lammpstrj every ${freq} dump xu yu zu\n\n"
        )

    if output_file is not None:
        with open(output_file, "w") as f:
            for line in file:
                f.write(line)

    return file


def _get_bar_lambdas(fep_files, indices=[2, 3]):
    """Retrieves all lambda values from FEP filenames.

    Parameters
    ----------
    fep_files: str or list of str
        Path(s) to fepout files to extract data from.
    indices : list[int], default=[1,2]
        In provided file names, using underscore as a separator, these indices mark the part of the filename
        containing the lambda information.

    Returns
    -------
    lambda_values : list
        List of tuples lambda values contained in the file.
    lambda_pairs : list
        List of tuples containing two floats, lambda and lambda'.

    """

    def tuple_from_filename(filename, separator="_", indices=[2, 3]):
        name_array = ".".join(os.path.split(filename)[-1].split(".")[:-1]).split(
            separator
        )
        if not _isfloat(name_array[indices[0]]):
            raise ValueError(
                f"Entry, {indices[0]} in filename cannot be converted to float: {name_array[indices[0]]}"
            )
        if not _isfloat(name_array[indices[1]]):
            raise ValueError(
                f"Entry, {indices[1]} in filename cannot be converted to float: {name_array[indices[1]]}"
            )
        return (float(name_array[indices[0]]), float(name_array[indices[1]]))

    def lambda2_from_filename(filename, separator="_", index=-1):
        name_array = ".".join(os.path.split(filename)[-1].split(".")[:-1]).split(
            separator
        )
        if not _isfloat(name_array[index]):
            raise ValueError(
                f"Entry, {index} in filename cannot be converted to float: {name_array[index]}"
            )
        return float(name_array[index])

    lambda_pairs = [tuple_from_filename(y, indices=indices) for y in fep_files]
    if len(indices) == 3:
        lambda2 = list(
            set([lambda2_from_filename(y, index=indices[2]) for y in fep_files])
        )
        if len(lambda2) > 1:
            raise ValueError(
                "More than one value of lambda2 is present in the provided files."
                f" Restrict filename input to one of: {lambda2}"
            )
    else:
        lambda2 = None

    lambda_values = sorted(list(set([x for y in lambda_pairs for x in y])))
    check_float = [x for x in lambda_values if not _isfloat(x)]
    if check_float:
        raise ValueError(
            "Lambda values must be convertible to floats: {}".format(check_float)
        )
    if [x for x in lambda_values if float(x) < 0]:
        raise ValueError("Lambda values must be positive: {}".format(lambda_values))

    # check that all needed lamba combinations are present
    lamda_dict = {x: [y[1] for y in lambda_pairs if y[0] == x] for x in lambda_values}

    # Check for MBAR content
    missing_combinations_mbar = []
    missing_combinations_bar = []
    for lambda_value, lambda_array in lamda_dict.items():
        missing_combinations_mbar.extend(
            [(lambda_value, x) for x in lambda_values if x not in lambda_array]
        )

    if missing_combinations_mbar:
        warnings.warn(
            "The following combinations of lambda and lambda prime are missing for MBAR analysis: {}".format(
                missing_combinations_mbar
            )
        )
    else:
        return lambda_values, lambda_pairs, lambda2

    # Check for BAR content
    missing_combinations_bar = []
    extra_combinations_bar = []
    lambda_values.sort()
    for ind, (lambda_value, lambda_array) in enumerate(lamda_dict.items()):
        if ind == 0:
            tmp_array = [lambda_values[ind], lambda_values[ind + 1]]
        elif ind == len(lamda_dict) - 1:
            tmp_array = [lambda_values[ind - 1], lambda_values[ind]]
        else:
            tmp_array = [
                lambda_values[ind - 1],
                lambda_values[ind],
                lambda_values[ind + 1],
            ]

        missing_combinations_bar.extend(
            [(lambda_value, x) for x in tmp_array if x not in lambda_array]
        )
        extra_combinations_bar.extend(
            [(lambda_value, x) for x in lambda_array if x not in tmp_array]
        )

    if missing_combinations_bar:
        raise ValueError(
            "BAR calculation cannot be performed without the following lambda-lambda prime combinations: {}".format(
                missing_combinations_bar
            )
        )
    if extra_combinations_bar:
        warnings.warn(
            "The following combinations of lambda and lambda prime are extra and being discarded for BAR analysis: {}".format(
                extra_combinations_bar
            )
        )
        lambda_pairs = [x for x in lambda_pairs if x not in extra_combinations_bar]

    return lambda_values, lambda_pairs, lambda2


@_init_attrs
def extract_u_nk(
    fep_files,
    T,
    columns_lambda1=[2, 3],
    column_u_nk=4,
    column_lambda2=None,
    indices=[1, 2],
    units="real",
    vdw_lambda=1,
):
    """This function will go into alchemlyb.parsing.lammps

    Each file is imported as a data frame where the columns kept are either:
        [0, columns_lambda1[0] columns_lambda1[1], column_u_nk]
    or if columns_lambda2 is not None:
        [0, columns_lambda1[0] columns_lambda1[1], column_lambda2, column_u_nk]

    Parameters
    ----------
    filenames : str
        Path to fepout file(s) to extract data from. Filenames and paths are
        aggregated using [glob](https://docs.python.org/3/library/glob.html). For example, "/path/to/files/something_*_*.txt".
    temperature : float
        Temperature in Kelvin at which the simulation was sampled.
    columns_lambda1 : list[int]
        Indices for columns (column number minus one) representing (1) the lambda at which the system is equilibrated and (2) the lambda used
        in the computation of the potential energy.
    column_u_nk : int, default=4
        Index for the column (column number minus one) representing the potential energy
    column_lambda2 : int
        Index for column (column number minus one) for the unchanging value of lambda for another potential.
        If ``None`` then we do not expect two lambda values being varied.
    indices : list[int], default=[1,2]
        In provided file names, using underscore as a separator, these indices mark the part of the filename
        containing the lambda information for :func:`alchemlyb.parsing._get_bar_lambdas`. If ``column_lambda2 != None``
        this list should be of length three, where the last value represents the invariant lambda.
    units : str, default="real"
        Unit system used in LAMMPS calculation. Currently supported: "real" and "lj"
    vdw_lambda : int, default=1
        In the case that ``column_lambda2 is not None``, this integer represents which lambda represents vdw interactions.

    Results
    -------
    u_nk_df : pandas.Dataframe
        Dataframe of potential energy for each alchemical state (k) for each frame (n).
        Note that the units for timestamps are not considered in the calculation.

        Attributes

        - temperature in K
        - energy unit in kT

    """

    # Collect Files
    files = glob.glob(fep_files)
    if not files:
        raise ValueError(f"No files have been found that match: {fep_files}")

    if units == "real":
        beta = 1 / (k_b * T)
    elif units == "lj":
        beta = 1 / T
    else:
        raise ValueError(
            f"LAMMPS unit type, {units}, is not supported. Supported types are: real and lj"
        )

    if len(columns_lambda1) != 2:
        raise ValueError(
            f"Provided columns for lambda1 must have a length of two, columns_lambda1: {columns_lambda1}"
        )
    if not np.all([isinstance(x, int) for x in columns_lambda1]):
        raise ValueError(
            f"Provided column for columns_lambda1 must be type int. columns_lambda1: {columns_lambda1}, type: {[type(x) for x in columns_lambda1]}"
        )
    if column_lambda2 is not None and not isinstance(column_lambda2, int):
        raise ValueError(
            f"Provided column for u_nk must be type int. column_u_nk: {column_lambda2}, type: {type(column_lambda2)}"
        )
    if not isinstance(column_u_nk, int):
        raise ValueError(
            f"Provided column for u_nk must be type int. column_u_nk: {column_u_nk}, type: {type(column_u_nk)}"
        )

    lambda_values, _, lambda2 = _get_bar_lambdas(files, indices=indices)

    if column_lambda2 is None:
        u_nk = pd.DataFrame(columns=["time", "fep-lambda"] + lambda_values)
        lc = len(lambda_values)
        col_indices = [0] + list(columns_lambda1) + [column_u_nk]
    else:
        u_nk = pd.DataFrame(columns=["time", "coul-lambda", "vdw-lambda"])
        lc = len(lambda_values) ** 2
        col_indices = [0] + list(columns_lambda1) + [column_lambda2, column_u_nk]

    for file in files:
        if not os.path.isfile(file):
            raise ValueError("File not found: {}".format(file))

        data = pd.read_csv(file, sep=" ", comment="#")
        lx = len(data.columns)
        if [False for x in col_indices if x > lx]:
            raise ValueError(
                "Number of columns, {}, is less than index: {}".format(lx, col_indices)
            )
        data = data.iloc[:, col_indices]
        if column_lambda2 is None:
            data.columns = ["time", "fep-lambda", "fep-lambda2", "u_nk"]
            lambda1_col, lambda1_2_col = "fep-lambda", "fep-lambda2"
            columns_a = ["time", "fep-lambda"]
            columns_b = lambda_values
        else:
            columns_a = ["time", "coul-lambda", "vdw-lambda"]
            if vdw_lambda == 1:
                data.columns = [
                    "time",
                    "vdw-lambda",
                    "vdw-lambda2",
                    "coul-lambda",
                    "u_nk",
                ]
                lambda1_col, lambda1_2_col = "vdw-lambda", "vdw-lambda2"
                columns_b = [(lambda2, x) for x in lambda_values]
            elif vdw_lambda == 2:
                data.columns = [
                    "time",
                    "coul-lambda",
                    "coul-lambda2",
                    "vdw-lambda",
                    "u_nk",
                ]
                lambda1_col, lambda1_2_col = "coul-lambda", "coul-lambda2"
                columns_b = [(x, lambda2) for x in lambda_values]
            else:
                raise ValueError(
                    f"'vdw_lambda must be either 1 or 2, not: {vdw_lambda}'"
                )

        for lambda1 in list(data[lambda1_col].unique()):
            tmp_df = data.loc[data[lambda1_col] == lambda1]

            for lambda12 in list(tmp_df[lambda1_2_col].unique()):
                tmp_df2 = tmp_df.loc[tmp_df[lambda1_2_col] == lambda12]

                lr = tmp_df2.shape[0]
                if u_nk[u_nk[lambda1_col] == lambda1].shape[0] == 0:
                    u_nk = pd.concat(
                        [
                            u_nk,
                            pd.concat(
                                [
                                    tmp_df2[columns_a],
                                    pd.DataFrame(
                                        np.zeros((lr, lc)),
                                        columns=columns_b,
                                    ),
                                ],
                                axis=1,
                            ),
                        ],
                        axis=0,
                        sort=False,
                    )

                column_name = lambda_values[
                    [ii for ii, x in enumerate(lambda_values) if float(x) == lambda12][
                        0
                    ]
                ]
                if column_lambda2 is not None:
                    column_name = (
                        (lambda2, column_name)
                        if vdw_lambda == 1
                        else (column_name, lambda2)
                    )
                if u_nk.loc[u_nk[lambda1_col] == lambda1, column_name][0] != 0:
                    raise ValueError(
                        "Energy values already available for lambda, {}, lambda', {}.".format(
                            lambda1, lambda12
                        )
                    )

                if (
                    u_nk.loc[u_nk[lambda1_col] == lambda1, column_name].shape[0]
                    != tmp_df2["u_nk"].shape[0]
                ):
                    raise ValueError(
                        "Number of energy values in file, {}, N={}, inconsistent with previous files of length, {}.".format(
                            file,
                            tmp_df2["u_nk"].shape[0],
                            u_nk.loc[u_nk[lambda1_col] == lambda1, column_name].shape[
                                0
                            ],
                        )
                    )

                u_nk.loc[u_nk[lambda1_col] == lambda1, column_name] = (
                    beta * tmp_df2["u_nk"]
                )

    if column_lambda2 is None:
        u_nk.set_index(["time", "fep-lambda"], inplace=True)
    else:
        u_nk.set_index(["time", "coul-lambda", "vdw-lambda"], inplace=True)

    return u_nk


@_init_attrs
def extract_dHdl(
    fep_files,
    T,
    column_lambda1=2,
    column_dlambda1=3,
    column_lambda2=None,
    column_dlambda2=None,
    columns_derivative1=[10, 11],
    columns_derivative2=[12, 13],
    index=-1,
    units="real",
):
    """This function will go into alchemlyb.parsing.lammps

    Each file is imported as a data frame where the columns kept are either:
        [0, column_lambda, column_dlambda1, columns_derivative[0], columns_derivative[1]]
    or if columns_lambda2 is not None:
        [
            0, column_lambda, column_dlambda1, column_lambda2, column_dlambda2,
            columns_derivative1[0], columns_derivative1[1], columns_derivative2[0], columns_derivative2[1]
        ]

    Parameters
    ----------
    filenames : str
        Path to fepout file(s) to extract data from. Filenames and paths are
        aggregated using [glob](https://docs.python.org/3/library/glob.html). For example, "/path/to/files/something_*_*.txt".
    temperature : float
        Temperature in Kelvin at which the simulation was sampled.
    column_lambda1 : int, default=2
        Index for column (column number minus one) representing the lambda at which the system is equilibrated.
    column_dlambda1 : int, default=3
        Index for column (column number minus one) for the change in lambda.
    column_lambda2 : int, default=None
        Index for column (column number minus one) for a second value of lambda.
        If this array is ``None`` then we do not expect two lambda values.
    column_dlambda2 : int, default=None
        Index for column (column number minus one) for the change in lambda2.
    columns_derivative : list[int], default=[10,11]
        Indices for columns (column number minus one) representing the lambda at which to find the forward
        and backward distance.
    index : int, default=-1
        In provided file names, using underscore as a separator, this index marks the part of the filename
        containing the lambda information for :func:`alchemlyb.parsing._get_ti_lambdas`.
    units : str, default="real"
        Unit system used in LAMMPS calculation. Currently supported: "real" and "lj"

    Results
    -------
    dHdl : pandas.Dataframe
        Dataframe of potential energy for each alchemical state (k) for each frame (n).
        Note that the units for timestamps are not considered in the calculation.

        Attributes

        - temperature in K or dimensionless
        - energy unit in kT

    """

    # Collect Files
    files = glob.glob(fep_files)
    if not files:
        raise ValueError("No files have been found that match: {}".format(fep_files))

    if units == "real":
        beta = 1 / (k_b * T)
    elif units == "lj":
        beta = 1 / T
    else:
        raise ValueError(
            "LAMMPS unit type, {}, is not supported. Supported types are: real and lj".format(
                units
            )
        )

    if not isinstance(column_lambda1, int):
        raise ValueError(
            "Provided column_lambda1 must be type 'int', instead: {}".format(
                type(column_lambda1)
            )
        )
    if column_lambda2 is not None and not isinstance(column_lambda2, int):
        raise ValueError(
            "Provided column_lambda2 must be type 'int', instead: {}".format(
                type(column_lambda2)
            )
        )
    if not isinstance(column_dlambda1, int):
        raise ValueError(
            "Provided column_dlambda1 must be type 'int', instead: {}".format(
                type(column_dlambda1)
            )
        )
    if column_dlambda2 is not None and not isinstance(column_dlambda2, int):
        raise ValueError(
            "Provided column_dlambda2 must be type 'int', instead: {}".format(
                type(column_dlambda2)
            )
        )

    if len(columns_derivative1) != 2:
        raise ValueError(
            "Provided columns for derivative values must have a length of two, columns_derivative1: {}".format(
                columns_derivative1
            )
        )
    if not np.all([isinstance(x, int) for x in columns_derivative1]):
        raise ValueError(
            "Provided column for columns_derivative1 must be type int. columns_derivative1: {}, type: {}".format(
                columns_derivative1, type([type(x) for x in columns_derivative1])
            )
        )
    if len(columns_derivative2) != 2:
        raise ValueError(
            "Provided columns for derivative values must have a length of two, columns_derivative2: {}".format(
                columns_derivative2
            )
        )
    if not np.all([isinstance(x, int) for x in columns_derivative2]):
        raise ValueError(
            "Provided column for columns_derivative1 must be type int. columns_derivative1: {}, type: {}".format(
                columns_derivative2, type([type(x) for x in columns_derivative2])
            )
        )

    if column_lambda2 is None:
        dHdl = pd.DataFrame(columns=["time", "fep-lambda", "fep"])
        col_indices = [0, column_lambda1, column_dlambda1] + list(columns_derivative1)
    else:
        dHdl = pd.DataFrame(
            columns=["time", "coul-lambda", "vdw-lambda", "coul", "vdw"]
        )
        col_indices = (
            [0, column_lambda2, column_lambda1, column_dlambda1, column_dlambda2]
            + list(columns_derivative1)
            + list(columns_derivative2)
        )

    for file in files:
        if not os.path.isfile(file):
            raise ValueError("File not found: {}".format(file))

        data = pd.read_csv(file, sep=" ", comment="#")
        lx = len(data.columns)
        if [False for x in col_indices if x > lx]:
            raise ValueError(
                "Number of columns, {}, is less than index: {}".format(lx, col_indices)
            )

        data = data.iloc[:, col_indices]
        if column_lambda2 is None:
            data.columns = ["time", "fep-lambda", "dlambda", "dU_back", "dU_forw"]
            data["fep"] = (data.dU_forw - data.dU_back) / (2 * data.dlambda)
            data.drop(columns=["dlambda", "dU_back", "dU_forw"], inplace=True)
            dHdl = pd.concat([dHdl, data], axis=0, sort=False)
        else:
            data.columns = [
                "time",
                "coul-lambda",
                "vdw-lambda",
                "dlambda_vdw",
                "dlambda_coul",
                "dU_back_vdw",
                "dU_forw_vdw",
                "dU_back_coul",
                "dU_forw_coul",
            ]
            data["coul"] = (data.dU_forw_coul - data.dU_back_coul) / (
                2 * data.dlambda_coul
            )
            data["vdw"] = (data.dU_forw_vdw - data.dU_back_vdw) / (2 * data.dlambda_vdw)
            data.drop(
                columns=[
                    "dlambda_vdw",
                    "dlambda_coul",
                    "dU_back_coul",
                    "dU_forw_coul",
                    "dU_back_vdw",
                    "dU_forw_vdw",
                ],
                inplace=True,
            )

    if column_lambda2 is None:
        dHdl.set_index(["time", "fep-lambda"], inplace=True)
        dHdl.mul({"fep": beta})
    else:
        dHdl.set_index(["time", "coul-lambda", "vdw-lambda"], inplace=True)
        dHdl.mul({"coul": beta, "vdw": beta})

    return dHdl
