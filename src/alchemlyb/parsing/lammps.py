""" Parsers for extracting alchemical data from [LAMMPS](https://docs.lammps.org/Manual.html) output files.

For clarity, we would like to distinguish the difference between $\lambda$ and $\lambda'$. We refer to $\lambda$ as 
the potential scaling of the equilibrated system, so that when this value is changed, the system undergoes another equilibration 
step. One the other hand, $\lambda'$ is the value used to scaled the potentials for the configurations of the system equilibrated 
for $\lambda$. The value of $\lambda'$ is used in two instances. First, in thermodynamic integration (TI), values of $\lambda'$ 
that are very close to $\lambda$ can be used to calculate the derivative. This is needed because LAMMPS does not compute 
explicit derivatives, although one should check whether they can derive an explicit expression, they cannot for changes of 
$\lambda'$ in the soft Lennard-Jones (LJ) potential.

The parsers featured in this module are constructed to parse LAMMPS output files output using the 
[`fix ave/time command`](https://docs.lammps.org/fix_ave_time.html), containing data for given potential energy values (an 
approximation of the Hamiltonian) at specified values of $\lambda$ and $\lambda'$, $U_{\lambda,\lambda'}$. Because generating 
the input files can be cumbersome, functions have been included to generate the appropriate sections. If a linear approximation 
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
from scipy import constants

from . import _init_attrs
from ..postprocessors.units import R_kJmol, kJ2kcal

def _isfloat(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def beta_from_units(T, units):
    """Output value of beta from temperature and units.

    Parameters
    ----------
    T : float
        Temperature that the system was run with
    units : str
        LAMMPS style unit

    Returns
    -------
    beta : float
        Value of beta used to scale the potential energy.
        
    Raises
    ------
    ValueError
        If unit string is not recognized.
        
    .. versionadded:: 1.??
    """
    if units == "real": # E in kcal/mol, T in K
        beta = 1 / (R_kJmol * kJ2kcal * T)
    elif units == "lj": # Nondimensional E and T scaled by epsilon
        beta = 1 / T
    elif units == "metal": # E in eV, T in K
        beta = 1 / (R_kJmol * kJ2kcal * T) # NoteHere!!!!
    elif units == "si": # E in J, T in K
        beta = 1 / (
            constants.R * T * 
            constants.physical_constants["electron volt"][0]
        )
    elif units == "cgs": # E in ergs, T in K
        beta = 1 / (constants.R * T * 1e-7)
    elif units == "electron": # E in Hartrees, T in K
        beta = 1 / (
            constants.R * T * 
            constants.physical_constants["Hartree energy"][0]
        )
    elif units == "micro": # E in epicogram-micrometer^2/microsecond^2, T in K
        beta = 1 / (constants.R * T * 1e-15)
    elif units == "nano": # E in attogram-nanometer^2/nanosecond^2, T in K
        beta = 1 / (constants.R * T * 1e-21)
    else:
        raise ValueError(
            "LAMMPS unit type, {}, is not supported. Supported types are: real and lj".format(
                units
            )
        )
        
    return beta


def _tuple_from_filename(filename, separator="_", indices=[2, 3], prec=4):
    """ Pull a tuple representing the lambda values used, as defined by the filenames.

    Parameters
    ----------
    filename : str
        Filename and path
    separator : str, default="_"
        Separator used to breakup the filename. The choice in ``indices`` is dependent on this choice.
    indices : list, default=[2, 3]
        Indices used to pull :math:`\lambda` and :math:`\lambda'`
    prec : int, default=4
        Number of decimal points in the output.

    Returns
    -------
    tuple[float]
        Tuple of lambda values

    .. versionadded:: 1.??
    """
    
    name_array = ".".join(os.path.split(filename)[-1].split(".")[:-1]).split(separator)
    if not _isfloat(name_array[indices[0]]):
        raise ValueError(
            f"Entry, {indices[0]} in filename cannot be converted to float: {name_array[indices[0]]}"
        )
    if not _isfloat(name_array[indices[1]]):
        raise ValueError(
            f"Entry, {indices[1]} in filename cannot be converted to float: {name_array[indices[1]]}"
        )
    return (round(float(name_array[indices[0]]), prec), round(float(name_array[indices[1]]), prec))

def _lambda_from_filename(filename, separator="_", index=-1, prec=4):
    """ Pull the :math:`\lambda'` value, as defined by the filenames.
    
    Here :math:`\lambda'` is the scaling value applied to a configuration that is equilibrated to
    a different value of :math:`\lambda`.

    Parameters
    ----------
    filename : str
        Filename and path
    separator : str, default="_"
        Separator used to breakup the filename. The choice in ``index`` is dependent on this choice.
    index : list, default=1
        Index used to pull :math:`\lambda'`
    prec : int, default=4
        Number of decimal points in the output.

    Returns
    -------
    float
        Lambda prime value

    .. versionadded:: 1.??
    """
    name_array = ".".join(os.path.split(filename)[-1].split(".")[:-1]).split(separator)
    if not _isfloat(name_array[index]):
        raise ValueError(
            f"Entry, {index} in filename cannot be converted to float: {name_array[index]}"
        )
    return round(float(name_array[index]), prec)

def _get_bar_lambdas(fep_files, indices=[2, 3], prec=4, force=False):
    """Retrieves all lambda values from FEP filenames.

    Parameters
    ----------
    fep_files: str or list of str
        Path(s) to fepout files to extract data from.
    indices : list[int], default=[1,2]
        In provided file names, using underscore as a separator, these indices mark the part of the filename
        containing the lambda information.
    prec : int, default=4
        Number of decimal places defined used in ``round()`` function.
    force : bool, default=False
        If ``True`` the dataframe will be created, even if not all lambda and lambda prime combinations are available.

    Returns
    -------
    lambda_values : list
        List of tuples lambda values contained in the file.
    lambda_pairs : list
        List of tuples containing two floats, lambda and lambda'.

    .. versionadded:: 1.??
    """

    lambda_pairs = [_tuple_from_filename(y, indices=indices, prec=prec) for y in fep_files]
    if len(indices) == 3:
        lambda2 = list(
            set([_lambda_from_filename(y, index=indices[2], prec=prec) for y in fep_files])
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
    if [x for x in lambda_values if round(float(x), prec) < 0]:
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

    if missing_combinations_bar and not force:
        raise ValueError(
            "BAR calculation cannot be performed without the following lambda-lambda prime combinations: {}".format(
                missing_combinations_bar
            )
        )
    if extra_combinations_bar and not force:
        warnings.warn(
            "The following combinations of lambda and lambda prime are extra and being discarded for BAR analysis: {}".format(
                extra_combinations_bar
            )
        )
        lambda_pairs = [x for x in lambda_pairs if x not in extra_combinations_bar]

    return lambda_values, lambda_pairs, lambda2


@_init_attrs
def extract_u_nk_from_u_n(
    fep_files, 
    T, 
    column_lambda, 
    column_u_cross, 
    dependence=lambda x : (x), 
    units="real", 
    index=-1, 
    prec=4, 
):
    """ Produce u_nk from files containing u_n given a separable dependence on lambda.

    Parameters
    ----------
    filenames : str
        Path to fepout file(s) to extract data from. Filenames and paths are
        aggregated using [glob](https://docs.python.org/3/library/glob.html). For example, "/path/to/files/something_*.txt".
    temperature : float
        Temperature in Kelvin at which the simulation was sampled.
    columns_lambda : int
        Indices for columns (file column number minus one) representing the lambda at which the system is equilibrated
    column_cross : int
        Index for the column (file column number minus one) representing the potential energy of the cross interactions 
        between the solute and solvent.
    dependence : func, default=`lambda x : (x)`
        Dependence of changing variable on the potential energy, which must be separable.
    index : int, default=-1
        In provided file names, using underscore as a separator, these indices mark the part of the filename
        containing the lambda information for :func:`alchemlyb.parsing._get_bar_lambdas`. If ``column_lambda2 != None``
        this list should be of length three, where the last value represents the invariant lambda.
    units : str, default="real"
        Unit system used in LAMMPS calculation. Currently supported: "real" and "lj"
    prec : int, default=4
        Number of decimal places defined used in ``round()`` function.
        
    Returns
    -------
    u_nk_df : pandas.Dataframe
        Dataframe of potential energy for each alchemical state (k) for each frame (n).
        Note that the units for timestamps are not considered in the calculation.

        Attributes

        - temperature in K
        - energy unit in kT
            
    .. versionadded:: 1.??
    """
    # Collect Files
    files = glob.glob(fep_files)
    if not files:
        raise ValueError(f"No files have been found that match: {fep_files}")

    beta = beta_from_units(T, units)

    if not isinstance(column_lambda, int):
        raise ValueError(
            f"Provided column for lambda must be type int. column_u_lambda: {column_lambda}, type: {type(column_lambda)}"
        )
    if not isinstance(column_u_cross, int):
        raise ValueError(
            f"Provided column for u_cross must be type int. column_u_cross: {column_u_cross}, type: {type(column_u_cross)}"
        )

    lambda_values = list(
            set([_lambda_from_filename(y, index=index, prec=prec) for y in files])
        )

    u_nk = pd.DataFrame(columns=["time", "fep-lambda"] + lambda_values)
    lc = len(lambda_values)
    col_indices = [0, column_lambda, column_u_cross]

    for file in files:
        if not os.path.isfile(file):
            raise ValueError("File not found: {}".format(file))

        data = pd.read_csv(file, sep=" ", comment="#", header=None)
        lx = len(data.columns)
        if [False for x in col_indices if x > lx]:
            raise ValueError(
                "Number of columns, {}, is less than index: {}".format(lx, col_indices)
            )
        data = data.iloc[:, col_indices]
        data.columns = ["time", "fep-lambda", "u_cross"]
        lambda1_col = "fep-lambda"
        data[[lambda1_col]] = data[[lambda1_col]].apply(
            lambda x: round(x, prec)
        )

        for lambda1 in list(data[lambda1_col].unique()):
            tmp_df = data.loc[data[lambda1_col] == lambda1]
            lr = tmp_df.shape[0]
            for lambda12 in lambda_values:
                if u_nk[u_nk[lambda1_col] == lambda1].shape[0] == 0:
                    u_nk = pd.concat(
                        [
                            u_nk,
                            pd.concat(
                                [
                                    tmp_df[["time", "fep-lambda"]],
                                    pd.DataFrame(
                                        np.zeros((lr, lc)),
                                        columns=lambda_values,
                                    ),
                                ],
                                axis=1,
                            ),
                        ],
                        axis=0,
                        sort=False,
                    )

                if u_nk.loc[u_nk[lambda1_col] == lambda1, lambda12][0] != 0:
                    raise ValueError(
                        "Energy values already available for lambda, {}, lambda', {}.".format(
                            lambda1, lambda12
                        )
                    )

                u_nk.loc[u_nk[lambda1_col] == lambda1, lambda12] = (
                    beta * tmp_df["u_cross"] * (dependence(lambda12) / dependence(lambda1) - 1)
                )

                if lambda1 == lambda12 and u_nk.loc[u_nk[lambda1_col] == lambda1, lambda12][0] != 0:
                    raise ValueError(f"The difference in PE should be zero when lambda = lambda', {lambda1} = {lambda12}," \
                        " Check that the 'column_u_n' was defined correctly.")

    u_nk.set_index(["time", "fep-lambda"], inplace=True)

    return u_nk


@_init_attrs
def extract_u_nk(
    fep_files,
    T,
    columns_lambda1=[1,2],
    column_u_nk=3,
    column_lambda2=None,
    indices=[1, 2],
    units="real",
    vdw_lambda=1,
    prec=4,
    force=False,
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
    columns_lambda1 : list[int], default=[1,2]
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
    prec : int, default=4
        Number of decimal places defined used in ``round()`` function.
    force : bool, default=False
        If ``True`` the dataframe will be created, even if not all lambda and lambda prime combinations are available.
        
    Results
    -------
    u_nk_df : pandas.Dataframe
        Dataframe of potential energy for each alchemical state (k) for each frame (n).
        Note that the units for timestamps are not considered in the calculation.

        Attributes

        - temperature in K
        - energy unit in kT

    .. versionadded:: 1.??
    """

    # Collect Files
    files = glob.glob(fep_files)
    if not files:
        raise ValueError(f"No files have been found that match: {fep_files}")

    beta = beta_from_units(T, units)

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
            f"Provided column for lambda must be type int. column_lambda2: {column_lambda2}, type: {type(column_lambda2)}"
        )
    if not isinstance(column_u_nk, int):
        raise ValueError(
            f"Provided column for u_nk must be type int. column_u_nk: {column_u_nk}, type: {type(column_u_nk)}"
        )

    lambda_values, _, lambda2 = _get_bar_lambdas(files, indices=indices, prec=prec, force=force)

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

        data = pd.read_csv(file, sep=" ", comment="#", header=None)
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
            data[[lambda1_col, lambda1_2_col]] = data[[lambda1_col, lambda1_2_col]].apply(
                lambda x: round(x, prec)
            )
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
            data[columns_a[1:]+[lambda1_2_col]] = data[columns_a[1:]+[lambda1_2_col]].apply(
                lambda x: round(x, prec)
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
                    
                column_list = [ii for ii, x in enumerate(lambda_values) if round(float(x), prec) == lambda12]
                if not column_list:
                    raise ValueError("Lambda values found in files do not align with those in the filenames. " \
                        "Check that 'columns_lambda' are defined correctly.")
                else:
                    column_name = lambda_values[column_list[0]]
                    
                if column_lambda2 is not None:
                    column_name = (
                        (lambda2, column_name)
                        if vdw_lambda == 1
                        else (column_name, lambda2)
                    )
                    
                if u_nk.loc[u_nk[lambda1_col] == lambda1, column_name][0] != abs(0):
                    raise ValueError(
                        "Energy values already available for lambda, {}, lambda', {}. Check for a duplicate file.".format(
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
                if lambda1 == lambda12 and u_nk.loc[u_nk[lambda1_col] == lambda1, column_name][0] != 0:
                    raise ValueError(f"The difference in PE should be zero when lambda = lambda', {lambda1} = {lambda12}," \
                        " Check that 'column_u_nk' was defined correctly.")

    if column_lambda2 is None:
        u_nk.set_index(["time", "fep-lambda"], inplace=True)
    else:
        u_nk.set_index(["time", "coul-lambda", "vdw-lambda"], inplace=True)

    return u_nk


@_init_attrs
def extract_dHdl_from_u_n(
    fep_files,
    T,
    column_lambda=None,
    column_u_cross=None,
    dependence=lambda x : (1/x),
    units="real",
):
    """Produce dHdl dataframe from sparated contributions of the potential energy.

    Each file is imported as a data frame where the columns are:
        [0, column_lambda, column_solvent, column_solute, column_cross]

    Parameters
    ----------
    filenames : str
        Path to fepout file(s) to extract data from. Filenames and paths are
        aggregated using [glob](https://docs.python.org/3/library/glob.html). For example, "/path/to/files/something_*.txt".
    T : float
        Temperature in Kelvin at which the simulation was sampled.
    columns_lambda : int, default=None
        Indices for columns (file column number minus one) representing the lambda at which the system is equilibrated
    column_u : int, default=None
        Index for the column (file column number minus one) representing the potential energy of the system
    dependence : func, default=`lambda x : (1/x)`
        Transform of lambda needed to convert the potential energy into the derivative of the potential energy with respect to lambda, which must be separable.
        For example, for the LJ potential U = eps * f(sig, r), dU/deps = f(sig, r), so we need a dependence function of 1/eps to convert the 
        potential energy to the derivative with respect to eps.
    units : str, default="real"
        Unit system used in LAMMPS calculation. Currently supported: "real" and "lj"

    Results
    -------
    dHdl : pandas.Dataframe
        Dataframe of the derivative for the potential energy for each alchemical state (k) 
        for each frame (n). Note that the units for timestamps are not considered in the calculation.

        Attributes

        - temperature in K or dimensionless
        - energy unit in kT

    .. versionadded:: 1.??
    """

    # Collect Files
    files = glob.glob(fep_files)
    if not files:
        raise ValueError(f"No files have been found that match: {fep_files}")

    beta = beta_from_units(T, units)

    if not isinstance(column_lambda, int):
        raise ValueError(
            f"Provided column for lambda must be type int. column_lambda: {column_lambda}, type: {type(column_lambda)}"
        )
    if not isinstance(column_u_cross, int):
        raise ValueError(
            f"Provided column for u_cross must be type int. column_u_cross: {column_u_cross}, type: {type(column_u_cross)}"
        )

    dHdl = pd.DataFrame(columns=["time", "fep-lambda", "fep"])
    col_indices = [0, column_lambda, column_u_cross]

    for file in files:
        if not os.path.isfile(file):
            raise ValueError("File not found: {}".format(file))

        data = pd.read_csv(file, sep=" ", comment="#", header=None)
        lx = len(data.columns)
        if [False for x in col_indices if x > lx]:
            raise ValueError(
                "Number of columns, {}, is less than index: {}".format(lx, col_indices)
            )

        data = data.iloc[:, col_indices]
        
        data.columns = ["time", "fep-lambda", "U"]
        data["fep"] = dependence(data.loc[:, "fep-lambda"]) * data.U
        data.drop( columns=["U"], inplace=True)

        dHdl = pd.concat([dHdl, data], axis=0, sort=False)

    dHdl.set_index(["time", "fep-lambda"], inplace=True)
    dHdl.mul({"fep": beta})

    return dHdl


@_init_attrs
def extract_dHdl(
    fep_files,
    T,
    column_lambda1=1,
    column_dlambda1=2,
    column_lambda2=None,
    column_dlambda2=None,
    columns_derivative1=[10, 11],
    columns_derivative2=[12, 13],
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
    units : str, default="real"
        Unit system used in LAMMPS calculation. Currently supported: "real" and "lj"

    Results
    -------
    dHdl : pandas.Dataframe
        Dataframe of the derivative for the potential energy for each alchemical state (k) 
        for each frame (n). Note that the units for timestamps are not considered in the calculation.

        Attributes

        - temperature in K or dimensionless
        - energy unit in kT

    .. versionadded:: 1.??
    """

    # Collect Files
    files = glob.glob(fep_files)
    if not files:
        raise ValueError("No files have been found that match: {}".format(fep_files))

    beta = beta_from_units(T, units)

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

        data = pd.read_csv(file, sep=" ", comment="#", header=None)
        lx = len(data.columns)
        if [False for x in col_indices if x > lx]:
            raise ValueError(
                "Number of columns, {}, is less than index: {}".format(lx, col_indices)
            )

        data = data.iloc[:, col_indices]
        if column_lambda2 is None:
            # dU_back: U(l-dl) - U(l); dU_forw: U(l+dl) - U(l)
            data.columns = ["time", "fep-lambda", "dlambda", "dU_back", "dU_forw"]
            data["fep"] = (data.dU_forw - data.dU_back) / (2 * data.dlambda)
            data.drop(columns=["dlambda", "dU_back", "dU_forw"], inplace=True)
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
        dHdl = pd.concat([dHdl, data], axis=0, sort=False)

    if column_lambda2 is None:
        dHdl.set_index(["time", "fep-lambda"], inplace=True)
        dHdl.mul({"fep": beta})
    else:
        dHdl.set_index(["time", "coul-lambda", "vdw-lambda"], inplace=True)
        dHdl.mul({"coul": beta, "vdw": beta})

    return dHdl


@_init_attrs
def extract_H(
    fep_files,
    T,
    column_lambda1=2,
    column_pe=5,
    column_lambda2=None,
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
    column_pe : int, default=5
        Index for column (column number minus one) representing the potential energy of the system.
    column_lambda2 : int, default=None
        Index for column (column number minus one) for a second value of lambda.
        If this array is ``None`` then we do not expect two lambda values.
    units : str, default="real"
        Unit system used in LAMMPS calculation. Currently supported: "real" and "lj"

    Results
    -------
    H : pandas.Dataframe
        Dataframe of potential energy for each alchemical state (k) for each frame (n).
        Note that the units for timestamps are not considered in the calculation.

        Attributes

        - temperature in K or dimensionless
        - energy unit in kT

    .. versionadded:: 1.??
    """

    # Collect Files
    files = glob.glob(fep_files)
    if not files:
        raise ValueError("No files have been found that match: {}".format(fep_files))

    beta = beta_from_units(T, units)

    if not isinstance(column_lambda1, int):
        raise ValueError(
            "Provided column_lambda1 must be type 'int', instead: {}".format(
                type(column_lambda1)
            )
        )
    if not isinstance(column_pe, int):
        raise ValueError(
            "Provided column_pe must be type 'int', instead: {}".format(
                type(column_pe)
            )
        )
    if column_lambda2 is not None and not isinstance(column_lambda2, int):
        raise ValueError(
            "Provided column_lambda2 must be type 'int', instead: {}".format(
                type(column_lambda2)
            )
        )

    if column_lambda2 is None:
        df_H = pd.DataFrame(columns=["time", "fep-lambda", "U"])
        col_indices = [0, column_lambda1, column_pe]
    else:
        df_H = pd.DataFrame(
            columns=["time", "coul-lambda", "vdw-lambda", "U"]
        )
        col_indices = [0, column_lambda2, column_lambda1, column_pe]
    
    for file in files:
        if not os.path.isfile(file):
            raise ValueError("File not found: {}".format(file))

        data = pd.read_csv(file, sep=" ", comment="#", header=None)
        lx = len(data.columns)
        if [False for x in col_indices if x > lx]:
            raise ValueError(
                "Number of columns, {}, is less than index: {}".format(lx, col_indices)
            )

        data = data.iloc[:, col_indices]
        if column_lambda2 is None:
            data.columns = ["time", "fep-lambda", "U"]
        else:
            data.columns = [
                "time",
                "coul-lambda",
                "vdw-lambda",
                "U",
            ]
        df_H = pd.concat([df_H, data], axis=0, sort=False)


    if column_lambda2 is None:
        df_H.set_index(["time", "fep-lambda"], inplace=True)
    else:
        df_H.set_index(["time", "coul-lambda", "vdw-lambda"], inplace=True)
    df_H.mul({"U": beta})

    return df_H