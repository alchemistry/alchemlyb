r""" Parsers for extracting alchemical data from `LAMMPS <https://docs.lammps.org/Manual.html>`_ output files.

For clarity, we would like to distinguish the difference between :math:`\lambda` and :math:`\lambda'`. We refer to :math:`\lambda` as 
the potential scaling of the equilibrated system, so that when this value is changed, the system undergoes another equilibration 
step. One the other hand, :math:`\lambda'` is the value used to scaled the potentials for the configurations of the system equilibrated 
for :math:`\lambda`. The value of :math:`\lambda'` is used in two instances. First, in thermodynamic integration (TI), values of :math:`\lambda'` 
that are very close to :math:`\lambda` can be used to calculate the derivative. This is needed because LAMMPS does not compute 
explicit derivatives, although one should check whether they can derive an explicit expression, they cannot for changes of 
:math:`\lambda'` in the soft Lennard-Jones (LJ) potential.

The parsers featured in this module are constructed to parse LAMMPS output files output using the 
`fix ave/time command <https://docs.lammps.org/fix_ave_time.html>`_, containing data for given potential energy values (an 
approximation of the Hamiltonian) at specified values of :math:`\lambda` and :math:`\lambda'`, :math:`U_{\lambda,\lambda'}`. Note that in 
LAMMPS, `fix adapt/fep <https://docs.lammps.org/fix_adapt_fep.html>`_ changes :math:`\lambda` and 
`compute fep <https://docs.lammps.org/compute_fep.html>`_ changes :math:`\lambda'`.

.. versionadded:: 2.4.1

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

    Supported types are: cgs, electron, lj. metal, micro, nano, real, si

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

    .. versionadded:: 2.4.1

    """
    if units == "real":  # E in kcal/mol, T in K
        beta = 1 / (R_kJmol * kJ2kcal * T)
    elif units == "lj":  # Nondimensional E and T scaled by epsilon
        beta = 1 / T
    elif units == "metal":  # E in eV, T in K
        beta = 1 / (constants.R * T / constants.eV / constants.Avogadro)
    elif units == "si":  # E in J, T in K
        beta = 1 / (constants.R * T / constants.Avogadro)
    elif units == "cgs":  # E in ergs, T in K
        beta = 1 / (constants.R * T / constants.Avogadro * 1e7)
    elif units == "electron":  # E in Hartrees, T in K
        beta = 1 / (
            constants.R
            * T
            / constants.Avogadro
            / constants.physical_constants["Hartree energy"][0]
        )
    elif units == "micro":  # E in picogram-micrometer^2/microsecond^2, T in K
        beta = 1 / (constants.R * T / constants.Avogadro * 1e15)
    elif units == "nano":  # E in attogram-nanometer^2/nanosecond^2, T in K
        beta = 1 / (constants.R * T / constants.Avogadro * 1e21)
    else:
        raise ValueError(
            "LAMMPS unit type, {}, is not supported. Supported types are: cgs, electron,"
            " lj. metal, micro, nano, real, si".format(units)
        )

    return beta


def energy_from_units(units):
    """Output conversion factor for pressure * volume to LAMMPS energy units

    Supported types are: cgs, electron, lj. metal, micro, nano, real, si

    Parameters
    ----------
    units : str
        LAMMPS style unit

    Returns
    -------
    conversion_factor : float
        Conversion factor for pressure * volume to LAMMPS energy units

    Raises
    ------
    ValueError
        If unit string is not recognized.

    .. versionadded:: 2.4.1

    """
    if units == "real":  # E in kcal/mol, Vol in Å^3, pressure in atm
        scaling_factor = (
            constants.atm * constants.angstrom**3 / 1e3 * kJ2kcal * constants.N_A
        )
    elif (
        units == "lj"
    ):  # Nondimensional E scaled by epsilon, vol in sigma^3, pressure in epsilon / sigma^3
        scaling_factor = 1
    elif units == "metal":  # E in eV, vol in Å^3, pressure in bar
        scaling_factor = constants.bar * constants.angstrom**3 / constants.eV
    elif units == "si":  # E in J, vol in m^3, pressure in Pa
        scaling_factor = 1
    elif units == "cgs":  # E in ergs, vol in cm^3, pressure in dyne/cm^2
        scaling_factor = 1
    elif units == "electron":  # E in Hartrees, vol in Bohr^3, pressure in Pa
        Hartree2J = constants.physical_constants["Hartree energy"][0]
        Bohr2m = constants.physical_constants["Bohr radius"][0]
        scaling_factor = Bohr2m**3 / Hartree2J
    elif units == "micro":
        # E in picogram-micrometer^2/microsecond^2, vol in um^3, pressure in picogram/(micrometer-microsecond^2)
        scaling_factor = 1
    elif units == "nano":
        # E in attogram-nanometer^2/nanosecond^2, vol in nm^3, pressure in attogram/(nanometer-nanosecond^2)
        scaling_factor = 1
    else:
        raise ValueError(
            "LAMMPS unit type, {}, is not supported. Supported types are: cgs, electron,"
            " lj. metal, micro, nano, real, si".format(units)
        )

    return scaling_factor


def _tuple_from_filename(filename, separator="_", indices=[2, 3], prec=4):
    r"""Pull a tuple representing the lambda values used, as defined by the filenames.

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

    .. versionadded:: 2.4.1

    """
    filename = filename.replace(".bz2", "").replace(".gz", "")
    name_array = ".".join(os.path.split(filename)[-1].split(".")[:-1]).split(separator)
    try:
        value1 = float(name_array[indices[0]])
    except ValueError:
        raise ValueError(
            f"Entry, {indices[0]} in filename cannot be converted to float: {name_array[indices[0]]}"
        )

    try:
        value2 = float(name_array[indices[1]])
    except ValueError:
        raise ValueError(
            f"Entry, {indices[1]} in filename cannot be converted to float: {name_array[indices[1]]}"
        )

    return (
        round(value1, prec),
        round(value2, prec),
    )


def _lambda_from_filename(filename, separator="_", index=-1, prec=4):
    r"""Pull the :math:`\lambda'` value, as defined by the filenames.

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

    .. versionadded:: 2.4.1

    """
    filename = filename.replace(".bz2", "").replace(".gz", "")
    name_array = ".".join(os.path.split(filename)[-1].split(".")[:-1]).split(separator)
    try:
        value = float(name_array[index])
    except:
        raise ValueError(
            f"Entry, {index} in filename cannot be converted to float: {name_array[index]}"
        )
    return round(value, prec)


def _get_bar_lambdas(fep_files, indices=[2, 3], prec=4, force=False):
    """Retrieves all lambda values from FEP filenames.

    Parameters
    ----------
    fep_files: str or list of str
        Path(s) to fepout files to extract data from.
    indices : list[int], default=[1,2]
        In provided file names, using underscore as a separator, these indices mark the part of the filename
        containing the lambda information. If three values, implies a value of lambda2 is present.
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
    lambda2 : float
        Value of lambda2 that is held constant.

    .. versionadded:: 2.5.0

    """

    lambda_pairs = [
        _tuple_from_filename(y, indices=indices, prec=prec) for y in fep_files
    ]
    if len(indices) == 3:
        lambda2 = list(
            set(
                [
                    _lambda_from_filename(y, index=indices[2], prec=prec)
                    for y in fep_files
                ]
            )
        )
        if len(lambda2) > 1:
            raise ValueError(
                "More than one value of lambda2 is present in the provided files."
                f" Restrict filename input to one of: {lambda2}"
            )
        else:
            lambda2 = lambda2[0]
    else:
        lambda2 = None

    lambda_values = sorted(list(set([x for y in lambda_pairs for x in y])))

    if [x for x in lambda_values if round(float(x), prec) < 0]:
        raise ValueError("Lambda values must be positive: {}".format(lambda_values))

    # check that all needed lamba combinations are present
    lambda_dict = {x: [y[1] for y in lambda_pairs if y[0] == x] for x in lambda_values}

    # Check for MBAR content
    missing_combinations_mbar = []
    missing_combinations_bar = []
    for lambda_value, lambda_array in lambda_dict.items():
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
    for ind, (lambda_value, lambda_array) in enumerate(lambda_dict.items()):
        if ind == 0:
            tmp_array = [lambda_values[ind], lambda_values[ind + 1]]
        elif ind == len(lambda_dict) - 1:
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
    column_U,
    column_U_cross,
    dependence=lambda x: (x),
    index=-1,
    units="real",
    prec=4,
    ensemble="nvt",
    pressure=None,
    column_volume=4,
):
    """Produce u_nk from files containing u_n given a separable dependence on lambda.

    Parameters
    ----------
    fep_files : str or list
        If not a list, a str representing the path to fepout file(s) to extract data from. Filenames and paths are
        aggregated using `glob <https://docs.python.org/3/library/glob.html>`_. For example, "/path/to/files/something_*.txt".
    T : float
        Temperature in Kelvin at which the simulation was sampled.
    columns_lambda : int
        Indices for columns (file column number minus one) representing the lambda at which the system is equilibrated
    column_U : int
        Index for the column (file column number minus one) representing the potential energy of the system.
    column_U_cross : int
        Index for the column (file column number minus one) representing the potential energy of the cross interactions
        between the solute and solvent.
    dependence : func, default=`lambda x : (x)`
        Dependence of changing variable on the potential energy, which must be separable.
    index : int, default=-1
        In provided file names, using underscore as a separator, these indices mark the part of the filename
        containing the lambda information for :func:`alchemlyb.parsing._lambda_from_filename`. If ``column_lambda2 != None``
        this list should be of length three, where the last value represents the invariant lambda.
    units : str, default="real"
        Unit system used in LAMMPS calculation. Currently supported: "cgs", "electron", "lj". "metal", "micro", "nano",
        "real", "si"
    prec : int, default=4
        Number of decimal places defined used in ``round()`` function.
    ensemble : str, default="nvt"
        Ensemble from which the given data was generated. Either "nvt" or "npt" is supported where values from NVT are
        unaltered, while those from NPT are corrected
    pressure : float, default=None
        The pressure of the system in the NPT ensemble in units of energy / volume, where the units of energy and volume
        are as recorded in the LAMMPS dump file.
    column_volume : int, default=4
        The column for the volume in a LAMMPS dump file.

    Returns
    -------
    u_nk_df : pandas.Dataframe
        Dataframe of potential energy for each alchemical state (k) for each frame (n).
        Note that the units for timestamps are not considered in the calculation.

        Attributes

        - temperature in K
        - energy unit in kT

    .. versionadded:: 2.4.1

    """
    # Collect Files
    if isinstance(fep_files, list):
        files = fep_files
    else:
        files = glob.glob(fep_files)
    if not files:
        raise ValueError(f"No files have been found that match: {fep_files}")

    if ensemble == "npt":
        if pressure is None or not isinstance(pressure, float) or pressure < 0:
            raise ValueError(
                "In the npt ensemble, a pressure must be provided in the form of a positive float"
            )
    elif ensemble != "nvt":
        raise ValueError("Only ensembles of nvt or npt are supported.")
    else:
        if pressure is not None:
            raise ValueError(
                "There is no volume correction in the nvt ensemble, the pressure value will not be used."
            )

    beta = beta_from_units(T, units)

    if not isinstance(column_lambda, int):
        raise ValueError(
            f"Provided column for lambda must be type int. column_u_lambda: {column_lambda}, type: {type(column_lambda)}"
        )
    if not isinstance(column_U_cross, int):
        raise ValueError(
            f"Provided column for `U_cross` must be type int. column_U_cross: {column_U_cross}, type: {type(column_U_cross)}"
        )
    if not isinstance(column_U, int):
        raise ValueError(
            f"Provided column for `U` must be type int. column_U: {column_U}, type: {type(column_U)}"
        )

    lambda_values = list(
        set([_lambda_from_filename(y, index=index, prec=prec) for y in files])
    )
    lambda_values = sorted(lambda_values)

    u_nk = pd.DataFrame(columns=["time", "fep-lambda"] + lambda_values)
    lc = len(lambda_values)
    col_indices = [0, column_lambda, column_U, column_U_cross]
    columns = ["time", "fep-lambda", "U", "U_cross"]
    if ensemble == "npt":
        col_indices.append(column_volume)
        columns.append("volume")

    for file in files:
        if not os.path.isfile(file):
            raise ValueError("File not found: {}".format(file))

        tmp_data = pd.read_csv(file, sep=" ", comment="#", header=None)
        ind = [x for x in col_indices if x > len(tmp_data.columns)]
        if len(ind) > 0:
            raise ValueError(
                "Number of columns, {}, is less than indices: {}".format(
                    len(tmp_data.columns), ind
                )
            )
        data = tmp_data.iloc[:, col_indices]
        data.columns = columns
        lambda1_col = "fep-lambda"
        data.loc[:, [lambda1_col]] = data[[lambda1_col]].apply(lambda x: round(x, prec))

        for lambda1 in list(data[lambda1_col].unique()):
            tmp_df = data.loc[data[lambda1_col] == lambda1]

            lr = tmp_df.shape[0]
            for lambda12 in lambda_values:
                if u_nk[u_nk[lambda1_col] == lambda1].shape[0] == 0:
                    tmp_u_nk = pd.concat(
                        [
                            tmp_df[["time", "fep-lambda"]],
                            pd.DataFrame(
                                np.zeros((lr, lc)),
                                columns=lambda_values,
                            ),
                        ],
                        axis=1,
                    )
                    u_nk = (
                        pd.concat([u_nk, tmp_u_nk], axis=0, sort=False)
                        if len(u_nk) != 0
                        else tmp_u_nk
                    )

                if u_nk.loc[u_nk[lambda1_col] == lambda1, lambda12][0] != 0:
                    raise ValueError(
                        "Energy values already available for lambda, {}, lambda', {}.".format(
                            lambda1, lambda12
                        )
                    )

                u_nk.loc[u_nk[lambda1_col] == lambda1, lambda12] = beta * (
                    tmp_df["U_cross"] * (dependence(lambda12) / dependence(lambda1) - 1)
                    + tmp_df["U"]
                )
                if ensemble == "npt":
                    u_nk.loc[u_nk[lambda1_col] == lambda1, lambda12] += (
                        beta * pressure * tmp_df["volume"] * energy_from_units(units)
                    )

    u_nk.set_index(["time", "fep-lambda"], inplace=True)
    u_nk.name = "u_nk"

    return u_nk


@_init_attrs
def extract_u_nk(
    fep_files,
    T,
    columns_lambda1=[1, 2],
    column_dU=4,
    column_U=3,
    column_lambda2=None,
    indices=[1, 2],
    units="real",
    vdw_lambda=1,
    ensemble="nvt",
    pressure=None,
    column_volume=6,
    prec=4,
    force=False,
):
    """Return reduced potentials `u_nk` from LAMMPS dump file(s).

    Each file is imported as a data frame where the columns kept are either::

        [0, columns_lambda1[0] columns_lambda1[1], column_U, column_dU]

    or if columns_lambda2 is not None::

        [0, columns_lambda1[0] columns_lambda1[1], column_lambda2, column_U, column_dU]

    If the simulation took place in the NPT ensemble, column_volume is appended to the end
    of this list.

    Parameters
    ----------
    fep_files : str or list
        If not a list of filenames, represents the path to fepout file(s) to extract data from. Filenames and paths are
        aggregated using `glob <https://docs.python.org/3/library/glob.html>`_. For example, "/path/to/files/something_*_*.txt".
    T : float
        Temperature in Kelvin at which the simulation was sampled.
    columns_lambda1 : list[int], default=[1,2]
        Indices for columns (column number minus one) representing (1) the lambda at which the system is equilibrated and (2) the lambda used
        in the computation of the potential energy.
    column_dU : int, default=4
        Index for the column (column number minus one) representing the difference in potential energy between lambda states
    column_U : int, default=3
        Index for the column (column number minus one) representing the potential energy
    column_lambda2 : int
        Index for column (column number minus one) for the unchanging value of lambda for another potential.
        If ``None`` then we do not expect two lambda values being varied.
    indices : list[int], default=[1,2]
        In provided file names, using underscore as a separator, these indices mark the part of the filename
        containing the lambda information for :func:`alchemlyb.parsing._get_bar_lambdas`. If ``column_lambda2 != None``
        this list should be of length three, where the last value represents the invariant lambda.
    units : str, default="real"
        Unit system used in LAMMPS calculation. Currently supported: "cgs", "electron", "lj". "metal", "micro", "nano",
        "real", "si"
    vdw_lambda : int, default=1
        In the case that ``column_lambda2 is not None``, this integer represents which lambda represents vdw interactions.
    ensemble : str, default="nvt"
        Ensemble from which the given data was generated. Either "nvt" or "npt" is supported where values from NVT are
        unaltered, while those from NPT are corrected
    pressure : float, default=None
        The pressure of the system in the NPT ensemble in units of energy / volume, where the units of energy and volume
        are as recorded in the LAMMPS dump file.
    column_volume : int, default=4
        The column for the volume in a LAMMPS dump file.
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

    .. versionadded:: 2.4.1

    """

    # Collect Files
    if isinstance(fep_files, list):
        files = fep_files
    else:
        files = glob.glob(fep_files)

    if not files:
        raise ValueError(f"No files have been found that match: {fep_files}")

    if ensemble == "npt":
        if pressure is None or not isinstance(pressure, float) or pressure < 0:
            raise ValueError(
                "In the npt ensemble, a pressure must be provided in the form of a positive float"
            )
    elif ensemble != "nvt":
        raise ValueError("Only ensembles of nvt or npt are supported.")
    elif pressure is not None:
        raise ValueError(
            "There is no volume correction in the nvt ensemble, the pressure value will not be used."
        )

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
    if not isinstance(column_dU, int):
        raise ValueError(
            f"Provided column for dU_nk must be type int. column_dU: {column_dU}, type: {type(column_dU)}"
        )
    if not isinstance(column_U, int):
        raise ValueError(
            f"Provided column for U must be type int. column_U: {column_U}, type: {type(column_U)}"
        )

    lambda_values, _, lambda2 = _get_bar_lambdas(
        files, indices=indices, prec=prec, force=force
    )

    if column_lambda2 is not None and lambda2 is None:
        raise ValueError(
            "If column_lambda2 is defined, the length of `indices` should be 3 indicating the value of the "
            "second value of lambda held constant."
        )

    # Set-up u_nk and column names / indices
    if column_lambda2 is None:  # No second lambda state value
        u_nk = pd.DataFrame(columns=["time", "fep-lambda"] + lambda_values)
        lc = len(lambda_values)
        # columns to pull from lammps dump file
        col_indices = [0] + list(columns_lambda1) + [column_U, column_dU]
        # column names from lammps dump file
        columns = ["time", "fep-lambda", "fep-lambda2", "U", "dU_nk"]
        columns_a = ["time", "fep-lambda"]  # u_nk cols 0, 1
        lambda1_col, lambda1_2_col = (
            "fep-lambda",
            "fep-lambda2",
        )  # cols for lambda, lambda'
        columns_b = lambda_values  # u_nk cols > 1
    else:  # There is a frozen, second lambda state
        u_nk = pd.DataFrame(columns=["time", "coul-lambda", "vdw-lambda"])
        lc = len(lambda_values)
        col_indices = (
            [0] + list(columns_lambda1) + [column_lambda2, column_U, column_dU]
        )  # columns to pull from lammps dump file
        if vdw_lambda == 1:
            # column names from lammps dump file
            columns = ["time", "vdw-lambda", "vdw-lambda2", "coul-lambda", "U", "dU_nk"]
            lambda1_col, lambda1_2_col = (
                "vdw-lambda",
                "vdw-lambda2",
            )  # cols for lambda, lambda'
            columns_b = [(lambda2, x) for x in lambda_values]  # u_nk cols > 2
        elif vdw_lambda == 2:
            # column names from lammps dump file
            columns = [
                "time",
                "coul-lambda",
                "coul-lambda2",
                "vdw-lambda",
                "U",
                "dU_nk",
            ]
            lambda1_col, lambda1_2_col = (
                "coul-lambda",
                "coul-lambda2",
            )  # cols for lambda, lambda'
            columns_b = [(x, lambda2) for x in lambda_values]  # u_nk cols > 2
        else:
            raise ValueError(f"'vdw_lambda must be either 1 or 2, not: {vdw_lambda}'")
        columns_a = ["time", "coul-lambda", "vdw-lambda"]  # u_nk cols 0, 1, 2

    if ensemble == "npt":
        col_indices.append(column_volume)
        columns.append("volume")

    # Parse Files
    for file in files:
        if not os.path.isfile(file):
            raise ValueError("File not found: {}".format(file))

        tmp_data = pd.read_csv(file, sep=" ", comment="#", header=None)
        ind = [x for x in col_indices if x >= len(tmp_data.columns)]
        if len(ind) > 0:
            raise ValueError(
                "Number of columns, {}, is less than necessary for indices: {}".format(
                    len(tmp_data.columns), ind
                )
            )
        data = tmp_data.iloc[:, col_indices]
        data.columns = columns

        # Round values of lambda according to ``prec`` variable
        if column_lambda2 is None:
            data.loc[:, [lambda1_col, lambda1_2_col]] = data[
                [lambda1_col, lambda1_2_col]
            ].apply(lambda x: round(x, prec))
        else:
            data.loc[:, columns_a[1:] + [lambda1_2_col]] = data[
                columns_a[1:] + [lambda1_2_col]
            ].apply(lambda x: round(x, prec))

        # Iterate over lambda states (configurations equilibrated at certain lambda value)
        for lambda1 in list(data[lambda1_col].unique()):
            if not np.isnan(lambda1) and lambda1 not in lambda_values:
                raise ValueError(
                    "Lambda value found in a file does not align with those in the filenames."
                    " Check that 'columns_lambda1[0]' or 'prec' are defined correctly. lambda"
                    " file: {}; lambda columns: {}".format(lambda1, lambda_values)
                )
            tmp_df = data.loc[data[lambda1_col] == lambda1]
            # Iterate over evaluated lambda' values at specific lambda state
            for lambda12 in list(tmp_df[lambda1_2_col].unique()):
                column_list = [
                    ii
                    for ii, x in enumerate(lambda_values)
                    if round(float(x), prec) == lambda12
                ]
                if not column_list:
                    raise ValueError(
                        "Lambda value found in a file does not align with those in the filenames. "
                        "Check that 'columns_lambda1[1]' or 'prec' are defined correctly. lambda"
                        " file: {}; lambda columns: {}".format(lambda12, lambda_values)
                    )
                else:
                    column_name = lambda_values[column_list[0]]

                tmp_df2 = tmp_df.loc[tmp_df[lambda1_2_col] == lambda12]

                lr = tmp_df2.shape[0]
                if u_nk[u_nk[lambda1_col] == lambda1].shape[0] == 0:
                    # If u_nk doesn't contain rows for this lambda state,
                    # Create rows with values of zero to populate energies
                    # from lambda' values
                    tmp_df3 = pd.concat(
                        [
                            tmp_df2[columns_a],
                            pd.DataFrame(
                                np.zeros((lr, lc)),
                                columns=columns_b,
                            ),
                        ],
                        axis=1,
                    )
                    u_nk = (  # If u_nk is empty, use this df, else concat
                        pd.concat([u_nk, tmp_df3], axis=0, sort=False)
                        if len(u_nk) != 0
                        else tmp_df3
                    )

                if column_lambda2 is not None:
                    column_name = (
                        (lambda2, column_name)
                        if vdw_lambda == 1
                        else (column_name, lambda2)
                    )

                column_index = list(u_nk.columns).index(column_name)
                row_indices = np.where(u_nk[lambda1_col] == lambda1)[0]

                if u_nk.iloc[row_indices, column_index][0] != 0:
                    raise ValueError(
                        "Energy values already available for lambda, {}, lambda', {}. Check for a duplicate file.".format(
                            lambda1, lambda12
                        )
                    )
                if lambda1 == lambda12 and not np.all(tmp_df2["dU_nk"][0] == 0):
                    raise ValueError(
                        f"The difference in dU should be zero when lambda = lambda', {lambda1} = {lambda12},"
                        " Check that 'column_dU' was defined correctly."
                    )

                if (
                    u_nk.iloc[row_indices, column_index].shape[0]
                    != tmp_df2["dU_nk"].shape[0]
                ):
                    old_length = tmp_df2["dU_nk"].shape[0]
                    stepsize = (
                        u_nk.loc[u_nk[lambda1_col] == lambda1, "time"].iloc[1]
                        - u_nk.loc[u_nk[lambda1_col] == lambda1, "time"].iloc[0]
                    )
                    # Fill in gaps where 'time' is NaN
                    nan_index = np.unique(np.where(tmp_df2["time"].isnull())[0])
                    for index in nan_index:
                        tmp_df2.loc[index, "time"] = (
                            tmp_df2.loc[index - 1, "time"] + stepsize
                        )

                    # Add rows of NaN for timesteps that are missing
                    new_index = pd.Index(
                        list(u_nk["time"].iloc[row_indices]), name="time"
                    )
                    tmp_df2 = tmp_df2.set_index("time").reindex(new_index).reset_index()

                    warnings.warn(
                        "Number of energy values in file, {}, N={}, inconsistent with previous".format(
                            file,
                            old_length,
                        )
                        + " files of length, {}. Adding NaN to row: {}".format(
                            u_nk.iloc[row_indices, column_index].shape[0],
                            np.unique(np.where(tmp_df2.isna())[0]),
                        )
                    )

                # calculate reduced potential u_k = dH + pV + U
                u_nk.iloc[row_indices, column_index] = beta * (
                    tmp_df2["dU_nk"] + tmp_df2["U"]
                )
                if ensemble == "npt":
                    u_nk.iloc[row_indices, column_index] += (
                        beta * pressure * tmp_df2["volume"] * energy_from_units(units)
                    )

    if column_lambda2 is None:
        u_nk.set_index(["time", "fep-lambda"], inplace=True)
    else:
        u_nk.set_index(["time", "coul-lambda", "vdw-lambda"], inplace=True)
    u_nk.name = "u_nk"

    u_nk = u_nk.dropna()

    return u_nk


@_init_attrs
def extract_dHdl_from_u_n(
    fep_files,
    T,
    column_lambda=None,
    column_u_cross=None,
    dependence=lambda x: (1 / x),
    units="real",
    prec=4,
):
    """Produce dHdl dataframe from separated contributions of the potential energy.

    Each file is imported as a dataframe where the columns are:

        [0, column_lambda, column_solvent, column_solute, column_cross]

    Parameters
    ----------
    fep_files : str or list
        If not a list, represents a path to fepout file(s) to extract data from. Filenames and paths are
        aggregated using `glob <https://docs.python.org/3/library/glob.html>`_. For example, "/path/to/files/something_*.txt".
    T : float
        Temperature in Kelvin at which the simulation was sampled.
    columns_lambda : int, default=None
        Indices for columns (file column number minus one) representing the lambda at which the system is equilibrated
    column_u_cross : int, default=None
        Index for the column (file column number minus one) representing the cross interaction potential energy of the system
    dependence : func, default=`lambda x : (1/x)`
        Transform of lambda needed to convert the potential energy into the derivative of the potential energy with respect to lambda, which must be separable.
        For example, for the LJ potential U = eps * f(sig, r), dU/deps = f(sig, r), so we need a dependence function of 1/eps to convert the
        potential energy to the derivative with respect to eps.
    units : str, default="real"
        Unit system used in LAMMPS calculation. Currently supported: "cgs", "electron", "lj". "metal", "micro", "nano",
        "real", "si"
    prec : int, default=4
        Number of decimal places defined used in ``round()`` function.

    Results
    -------
    dHdl : pandas.Dataframe
        Dataframe of the derivative for the potential energy for each alchemical state (k)
        for each frame (n). Note that the units for timestamps are not considered in the calculation.

        Attributes

        - temperature in K or dimensionless
        - energy unit in kT

    .. versionadded:: 2.4.1

    """

    # Collect Files
    if isinstance(fep_files, list):
        files = fep_files
    else:
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
        if [False for x in col_indices if x >= lx]:
            raise ValueError(
                "Number of columns, {}, is less than index: {}".format(lx, col_indices)
            )

        data = data.iloc[:, col_indices]

        data.columns = ["time", "fep-lambda", "U"]
        data["fep-lambda"] = data["fep-lambda"].apply(lambda x: round(x, prec))
        data["fep"] = dependence(data.loc[:, "fep-lambda"]) * data.U
        data.drop(columns=["U"], inplace=True)

        dHdl = pd.concat([dHdl, data], axis=0, sort=False) if len(dHdl) != 0 else data

    dHdl.set_index(["time", "fep-lambda"], inplace=True)
    dHdl = dHdl.mul({"fep": beta})
    dHdl.name = "dH_dl"

    return dHdl


@_init_attrs
def extract_dHdl(
    fep_files,
    T,
    column_lambda1=1,
    column_dlambda1=2,
    column_lambda2=None,
    columns_derivative=[8, 7],
    vdw_lambda=1,
    units="real",
    prec=4,
):
    """Return reduced potentials `dHdl` from LAMMPS dump file(s).

    Each file is imported as a data frame where the columns kept are either::

        [0, column_lambda, column_dlambda1, columns_derivative[0], columns_derivative[1]]

    or if columns_lambda2 is not None::

        [
            0, column_lambda, column_dlambda1, column_lambda2, column_dlambda2,
            columns_derivative[0], columns_derivative[1],
        ]

    Parameters
    ----------
    fep_files : str or list
        If not a list, represents the path to fepout file(s) to extract data from. Filenames and paths are
        aggregated using `glob <https://docs.python.org/3/library/glob.html>`_. For example, "/path/to/files/something_*_*.txt".
    T : float
        Temperature in Kelvin at which the simulation was sampled.
    column_lambda1 : int, default=2
        Index for column (column number minus one) representing the lambda at which the system is equilibrated.
    column_dlambda1 : int, default=3
        Index for column (column number minus one) for the change in lambda.
    column_lambda2 : int, default=None
        Index for column (column number minus one) for a second value of lambda.
        If this array is ``None`` then we do not expect two lambda values.
    columns_derivative : list[int], default=[8, 7]
        Indices for columns (column number minus one) representing the the forward
        and backward derivative respectively.
    vdw_lambda : int, default=1
        In the case that ``column_lambda2 is not None``, this integer represents which lambda represents vdw interactions.
    units : str, default="real"
        Unit system used in LAMMPS calculation. Currently supported: "cgs", "electron", "lj". "metal", "micro", "nano",
        "real", "si"
    prec : int, default=4
        Number of decimal places defined used in ``round()`` function.

    Results
    -------
    dHdl : pandas.Dataframe
        Dataframe of the derivative for the potential energy for each alchemical state (k)
        for each frame (n). Note that the units for timestamps are not considered in the calculation.

        Attributes

        - temperature in K or dimensionless
        - energy unit in kT

    .. versionadded:: 2.4.1

    """

    # Collect Files
    if isinstance(fep_files, list):
        files = fep_files
    else:
        files = glob.glob(fep_files)
    if not files:
        raise ValueError("No files have been found that match: {}".format(fep_files))

    beta = beta_from_units(T, units)

    if not isinstance(column_lambda1, int):
        raise ValueError(
            "Provided column_lambda1 must be type 'int', instead of {}".format(
                type(column_lambda1)
            )
        )
    if column_lambda2 is not None and not isinstance(column_lambda2, int):
        raise ValueError(
            "Provided column_lambda2 must be type 'int', instead of {}".format(
                type(column_lambda2)
            )
        )
    if not isinstance(column_dlambda1, int):
        raise ValueError(
            "Provided column_dlambda1 must be type 'int', instead of {}".format(
                type(column_dlambda1)
            )
        )

    if len(columns_derivative) != 2:
        raise ValueError(
            "Provided columns for derivative values must have a length of two, columns_derivative: {}".format(
                columns_derivative
            )
        )
    if not np.all([isinstance(x, int) for x in columns_derivative]):
        raise ValueError(
            "Provided column for columns_derivative must be type int. columns_derivative: {}, type: {}".format(
                columns_derivative, type([type(x) for x in columns_derivative])
            )
        )

    if column_lambda2 is None:
        dHdl = pd.DataFrame(columns=["time", "fep-lambda", "fep"])
        col_indices = [0, column_lambda1, column_dlambda1] + list(columns_derivative)
    else:
        if vdw_lambda == 1:
            dHdl = pd.DataFrame(columns=["time", "vdw-lambda", "coul-lambda", "vdw"])
        else:
            dHdl = pd.DataFrame(columns=["time", "coul-lambda", "vdw-lambda", "coul"])
        col_indices = [
            0,
            column_lambda1,
            column_lambda2,
            column_dlambda1,
        ] + list(columns_derivative)

    for file in files:
        if not os.path.isfile(file):
            raise ValueError("File not found: {}".format(file))

        data = pd.read_csv(file, sep=" ", comment="#", header=None)
        lx = len(data.columns)
        if [False for x in col_indices if x >= lx]:
            raise ValueError(
                "Number of columns, {}, is less than index: {}".format(lx, col_indices)
            )

        data = data.iloc[:, col_indices]
        if column_lambda2 is None:
            # dU_back: U(l-dl) - U(l); dU_forw: U(l+dl) - U(l)
            data.columns = ["time", "fep-lambda", "dlambda", "dU_forw", "dU_back"]
            data["fep-lambda"] = data["fep-lambda"].apply(lambda x: round(x, prec))
            data["fep"] = (data.dU_forw - data.dU_back) / (2 * data.dlambda)
            data.drop(columns=["dlambda", "dU_back", "dU_forw"], inplace=True)
        else:
            if vdw_lambda == 1:
                columns = [
                    "time",
                    "vdw-lambda",
                    "coul-lambda",
                    "dlambda_vdw",
                    "dU_back_vdw",
                    "dU_forw_vdw",
                ]
                data.columns = columns
                data["vdw"] = (data.dU_forw_vdw - data.dU_back_vdw) / (
                    2 * data.dlambda_vdw
                )
            elif vdw_lambda == 2:
                columns = [
                    "time",
                    "coul-lambda",
                    "vdw-lambda",
                    "dlambda_coul",
                    "dU_back_coul",
                    "dU_forw_coul",
                ]
                data.columns = columns
                data["coul"] = (data.dU_forw_coul - data.dU_back_coul) / (
                    2 * data.dlambda_coul
                )
            data["vdw-lambda"] = data["vdw-lambda"].apply(lambda x: round(x, prec))
            data["coul-lambda"] = data["coul-lambda"].apply(lambda x: round(x, prec))

            data.drop(
                columns=columns[3:],
                inplace=True,
            )
        dHdl = pd.concat([dHdl, data], axis=0, sort=False) if len(dHdl) != 0 else data

    if column_lambda2 is None:
        dHdl.set_index(["time", "fep-lambda"], inplace=True)
        dHdl = dHdl.mul({"fep": beta})
    else:
        dHdl.set_index(["time", "coul-lambda", "vdw-lambda"], inplace=True)
        if vdw_lambda == 1:
            dHdl = dHdl.mul({"vdw": beta})
        elif vdw_lambda == 2:
            dHdl = dHdl.mul({"coul": beta})

    dHdl.name = "dH_dl"

    return dHdl


@_init_attrs
def extract_H(
    fep_files,
    T,
    column_lambda1=1,
    column_pe=5,
    column_lambda2=None,
    units="real",
    ensemble="nvt",
    pressure=None,
    column_volume=6,
):
    """Return reduced potentials Hamiltonian from LAMMPS dump file(s).

    Each file is imported as a data frame where the columns kept are either::

        [0, column_lambda, column_dlambda1, columns_derivative[0], columns_derivative[1]]

    or if columns_lambda2 is not None::

        [
            0, column_lambda, column_dlambda1, column_lambda2, column_dlambda2,
            columns_derivative1[0], columns_derivative1[1]
        ]

    Parameters
    ----------
    fep_files : str or list
        If not a list, represents the path to fepout file(s) to extract data from. Filenames and paths are
        aggregated using `glob <https://docs.python.org/3/library/glob.html>`_. For example, "/path/to/files/something_*_*.txt".
    T : float
        Temperature in Kelvin at which the simulation was sampled.
    column_lambda1 : int, default=1
        Index for column (column number minus one) representing the lambda at which the system is equilibrated.
    column_pe : int, default=5
        Index for column (column number minus one) representing the potential energy of the system.
    column_lambda2 : int, default=None
        Index for column (column number minus one) for a second value of lambda.
        If this array is ``None`` then we do not expect two lambda values.
    units : str, default="real"
        Unit system used in LAMMPS calculation. Currently supported: "cgs", "electron", "lj". "metal", "micro", "nano",
        "real", "si"
    ensemble : str, default="nvt"
        Ensemble from which the given data was generated. Either "nvt" or "npt" is supported where values from NVT are
        unaltered, while those from NPT are corrected
    pressure : float, default=None
        The pressure of the system in the NPT ensemble in units of energy / volume, where the units of energy and volume
        are as recorded in the LAMMPS dump file.
    column_volume : int, default=4
        The column for the volume in a LAMMPS dump file.

    Results
    -------
    H : pandas.Dataframe
        Dataframe of potential energy for each alchemical state (k) for each frame (n).
        Note that the units for timestamps are not considered in the calculation.

        Attributes

        - temperature in K or dimensionless
        - energy unit in kT

    .. versionadded:: 2.4.1

    """

    # Collect Files
    if isinstance(fep_files, list):
        files = fep_files
    else:
        files = glob.glob(fep_files)
    if not files:
        raise ValueError("No files have been found that match: {}".format(fep_files))

    if ensemble == "npt":
        if pressure is None or not isinstance(pressure, float) or pressure < 0:
            raise ValueError(
                "In the npt ensemble, a pressure must be provided in the form of a positive float"
            )
    elif ensemble != "nvt":
        raise ValueError("Only ensembles of nvt or npt are supported.")
    else:
        if pressure is not None:
            raise ValueError(
                "There is no volume correction in the nvt ensemble, the pressure value will not be used."
            )

    beta = beta_from_units(T, units)

    if not isinstance(column_lambda1, int):
        raise ValueError(
            "Provided column_lambda1 must be type 'int', instead of {}".format(
                type(column_lambda1)
            )
        )
    if not isinstance(column_pe, int):
        raise ValueError(
            "Provided column_pe must be type 'int', instead of {}".format(
                type(column_pe)
            )
        )
    if column_lambda2 is not None and not isinstance(column_lambda2, int):
        raise ValueError(
            "Provided column_lambda2 must be type 'int', instead of {}".format(
                type(column_lambda2)
            )
        )

    if column_lambda2 is None:
        columns = ["time", "fep-lambda", "u_n"]
        col_indices = [0, column_lambda1, column_pe]
    else:
        columns = ["time", "coul-lambda", "vdw-lambda", "u_n"]
        col_indices = [0, column_lambda2, column_lambda1, column_pe]

    if ensemble == "npt":
        col_indices.append(column_volume)

    df_H = pd.DataFrame(columns=columns)

    for file in files:
        if not os.path.isfile(file):
            raise ValueError("File not found: {}".format(file))

        data = pd.read_csv(file, sep=" ", comment="#", header=None)
        lx = len(data.columns)
        if [False for x in col_indices if x >= lx]:
            raise ValueError(
                "Number of columns, {}, is less than index: {}".format(lx, col_indices)
            )

        data = data.iloc[:, col_indices]
        if column_lambda2 is None:
            columns = ["time", "fep-lambda", "U"]
        else:
            columns = [
                "time",
                "coul-lambda",
                "vdw-lambda",
                "U",
            ]
        if ensemble == "npt":
            columns.append("volume")
        data.columns = columns
        data["u_n"] = beta * data["U"]
        del data["U"]
        if ensemble == "npt":
            data["u_n"] += beta * pressure * data["volume"] * energy_from_units(units)
            del data["volume"]

        df_H = pd.concat([df_H, data], axis=0, sort=False) if len(df_H) != 0 else data

    if column_lambda2 is None:
        df_H.set_index(["time", "fep-lambda"], inplace=True)
    else:
        df_H.set_index(["time", "coul-lambda", "vdw-lambda"], inplace=True)
    df_H = df_H.mul({"U": beta})

    return df_H
