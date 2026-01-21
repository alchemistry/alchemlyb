r"""
Parsers for extracting alchemical data from `LAMMPS <https://docs.lammps.org/Manual.html>`_ output files.

Use Cases for extract_* Functions
=================================

For clarity, we would like to distinguish the difference between :math:`\lambda` and :math:`\lambda'`. We refer to :math:`\lambda` as
the potential scaling of the equilibrated system, so that when this value is changed, the system undergoes another equilibration
step. On the other hand, :math:`\lambda'` is the value used to scale the potentials for the configurations of the system equilibrated
for :math:`\lambda`. The value of :math:`\lambda'` is used in two instances. First, in thermodynamic integration (TI), values of :math:`\lambda'`
that are very close to :math:`\lambda` can be used to calculate the derivative. This is needed because LAMMPS does not compute
explicit derivatives; although one should check whether explicit expressions can be derived, this is not possible for changes of
:math:`\lambda'` in the soft Lennard-Jones (LJ) potential.

The extract_* functions in this module are designed to handle different aspects of alchemical free energy calculations. Below is an overview of their use cases:

    **extract_u_nk**
      - *Purpose:* Extracts reduced potentials (u_nk) for each alchemical state (k) for each frame (n).
      - *Use Case:* Suitable for MBAR (Multistate Bennett Acceptance Ratio) analysis, where the reduced potentials are required to compute free energy differences across multiple states.
      - *Input Requirements:* Requires columns for timestep, lambda values, potential energy, and optionally volume (for NPT ensemble).

    **extract_dHdl**
      - *Purpose:* Extracts the derivative of the Hamiltonian with respect to lambda (dH/dλ) for each alchemical state.
      - *Use Case:* Used in Thermodynamic Integration (TI) to compute free energy differences by integrating dH/dλ over lambda.
      - *Input Requirements:* Requires columns for timestep, lambda values, lambda derivatives, and derivative values for different components.

    **extract_H**
      - *Purpose:* Extracts the Hamiltonian (potential energy) for each alchemical state.
      - *Use Case:* Provides the raw potential energy data for analysis or validation purposes.
      - *Input Requirements:* Requires columns for timestep, lambda values, and potential energy.

    **extract_u_nk_from_u_n**
      - *Purpose:* Constructs u_nk from files containing u_n given a separable dependence on lambda.
      - *Use Case:* Useful when the dependence of the potential energy on lambda can be expressed as a separable function. This function is provided to reduce the IO cost required if all :math:`\lambda'` must be computed during a simulation.
      - *Input Requirements:* Requires columns for lambda, potential energy, and optionally volume (for NPT ensemble).

File Format Requirements
========================

The parsers featured in this module are constructed to parse LAMMPS output files output using the
`fix ave/time command <https://docs.lammps.org/fix_ave_time.html>`_, containing data for given potential energy values (an
approximation of the Hamiltonian) at specified values of :math:`\lambda` and :math:`\lambda'`, :math:`U_{\lambda,\lambda'}`. Note that in
LAMMPS, `fix adapt/fep <https://docs.lammps.org/fix_adapt_fep.html>`_ changes :math:`\lambda` and
`compute fep <https://docs.lammps.org/compute_fep.html>`_ changes :math:`\lambda'`.

Given the broad flexibility and unstandardized format of LAMMPS output files a user should consider the way they write the output of their
simulation. A user may find the package `generate_alchemical_lammps_inputs <https://github.com/usnistgov/generate_alchemical_lammps_inputs>`_
useful to generate their input scripts. Input files should be space-separated text files produced by LAMMPS `fix ave/time` command, typically
with the following characteristics:

**File Structure:**
- Space-separated columns with no header
- Lines starting with '#' are treated as comments and ignored
- Each row represents a single timestep/frame
- Compressed files (.gz, .bz2) are automatically handled

**Essential Columns:**
The specific column indices depend on the extraction function, but generally include:

For MBAR extraction (`extract_u_nk`):
- Column 0: Timestep/iteration number
- Columns 1-2: Lambda values (λ, λ') defining the alchemical state
- Column 3: Potential energy U at the current lambda state
- Column 4: Potential energy difference dU between lambda states
- Column 6: Volume (for NPT ensemble, optional)

For TI extraction (`extract_dHdl`):
- Column 0: Timestep/iteration number
- Column 1: Lambda value λ
- Column 2: Lambda derivative dλ/dt
- Columns 5,7: Derivative values ∂H/∂λ for different components
- Additional columns may contain volume, pressure, etc.

**Example File Content:**

.. code-block:: text

    # LAMMPS output from fix ave/time
    # Time Lambda1 Lambda2 U dU Volume dHdl1 dHdl2
    100   0.0     0.0     -1234.5  0.0      12345.6  -23.4   15.2
    200   0.0     0.1     -1235.1  -0.6     12346.1  -24.1   15.8
    300   0.0     0.2     -1236.2  -1.7     12347.0  -25.3   16.5
    ...

**Filename Conventions:**
- Filenames should encode lambda values for proper parsing:
- Format: `prefix_λ1_λ2_suffix.ext` (using underscores as separators)
- Examples: `mbar_charge_0.0_1.0.txt`, `fep_vdw_0.5_0.75.dat`
- Lambda values are extracted from specific positions in the filename
- Compression extensions (.gz, .bz2) are automatically removed during parsing

**Supported Units:**
- LAMMPS unit systems: "real", "lj", "metal", "si", "cgs", "electron", "micro", "nano"

**Ensemble Support:**
- NVT: Standard canonical ensemble (no volume correction)
- NPT: Isothermal-isobaric ensemble (requires volume column and pressure specification)

.. versionadded:: 2.4.1

"""

import os
import warnings
from collections.abc import Callable
import numpy as np
import pandas as pd
import glob
from scipy import constants

from . import _init_attrs
from ..postprocessors.units import R_kJmol, kJ2kcal


def _validate_ensemble_parameters(ensemble: str, pressure: float | None) -> None:
    """
    Validate ensemble and pressure parameters for LAMMPS parsing.

    .. versionadded:: 2.4.0

    Parameters
    ----------
    ensemble : str
        Ensemble type ("npt" or "nvt").
    pressure : float or None
        Pressure value for npt ensemble.

    Raises
    ------
    ValueError
        If ensemble/pressure combination is invalid.
    """
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


def beta_from_units(T: float, units: str) -> float:
    """Output value of beta from temperature and units.

    Supported types are: cgs, electron, lj, metal, micro, nano, real, si

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
    beta: float
    match units:
        case "real":  # E in kcal/mol, T in K
            beta = 1 / (R_kJmol * kJ2kcal * T)
        case "lj":  # Nondimensional E and T scaled by epsilon
            beta = 1 / T
        case "metal":  # E in eV, T in K
            beta = 1 / (constants.R * T / constants.eV / constants.Avogadro)
        case "si":  # E in J, T in K
            beta = 1 / (constants.R * T / constants.Avogadro)
        case "cgs":  # E in ergs, T in K
            beta = 1 / (constants.R * T / constants.Avogadro * 1e7)
        case "electron":  # E in Hartrees, T in K
            beta = 1 / (
                constants.R
                * T
                / constants.Avogadro
                / constants.physical_constants["Hartree energy"][0]
            )
        case "micro":  # E in picogram-micrometer^2/microsecond^2, T in K
            beta = 1 / (constants.R * T / constants.Avogadro * 1e15)
        case "nano":  # E in attogram-nanometer^2/nanosecond^2, T in K
            beta = 1 / (constants.R * T / constants.Avogadro * 1e21)
        case _:
            raise ValueError(
                "LAMMPS unit type, {}, is not supported. Supported types are: cgs, electron, "
                "lj, metal, micro, nano, real, si".format(units)
            )

    return beta


def energy_from_units(units: str) -> float:
    """Output conversion factor for pressure * volume to LAMMPS energy units

    Supported types are: cgs, electron, lj, metal, micro, nano, real, si

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
    scaling_factor: float
    match units:
        case "real":  # E in kcal/mol, Vol in Å^3, pressure in atm
            scaling_factor = (
                constants.atm * constants.angstrom**3 / 1e3 * kJ2kcal * constants.N_A
            )
        case "lj":  # Nondimensional E scaled by epsilon, vol in sigma^3, pressure in epsilon / sigma^3
            scaling_factor = 1
        case "metal":  # E in eV, vol in Å^3, pressure in bar
            scaling_factor = constants.bar * constants.angstrom**3 / constants.eV
        case "si":  # E in J, vol in m^3, pressure in Pa
            scaling_factor = 1
        case "cgs":  # E in ergs, vol in cm^3, pressure in dyne/cm^2
            scaling_factor = 1
        case "electron":  # E in Hartrees, vol in Bohr^3, pressure in Pa
            Hartree2J = constants.physical_constants["Hartree energy"][0]
            Bohr2m = constants.physical_constants["Bohr radius"][0]
            scaling_factor = Bohr2m**3 / Hartree2J
        case "micro":
            # E in picogram-micrometer^2/microsecond^2, vol in um^3, pressure in picogram/(micrometer-microsecond^2)
            scaling_factor = 1
        case "nano":
            # E in attogram-nanometer^2/nanosecond^2, vol in nm^3, pressure in attogram/(nanometer-nanosecond^2)
            scaling_factor = 1
        case _:
            raise ValueError(
                "LAMMPS unit type, {}, is not supported. Supported types are: cgs, electron, "
                "lj, metal, micro, nano, real, si".format(units)
            )

    return scaling_factor


def tuple_from_filename(
    filename: str, separator: str = "_", indices: list[int] = [2, 3], prec: int = 4
) -> tuple[float, ...]:
    r"""Pull a tuple representing the lambda values used, as defined by the filenames.

    This function extracts lambda values from structured filenames that contain numerical
    lambda values separated by a specified separator character. The function is designed
    to work with various filename formats as long as they follow a consistent pattern.

    Examples with different indices configurations:

    **Default indices=[2, 3], separator="_":**

    - ``fep_0.0_1.0.txt`` → (0.0, 1.0)
    - ``simulation_data_0.25_0.75_output.dat`` → (0.25, 0.75)
    - ``lammps_run_0.5_1.0.log.gz`` → (0.5, 1.0)
    - ``path/to/file_prefix_0.1_0.9_suffix.txt.bz2`` → (0.1, 0.9)

    **indices=[0, 1], separator="_":**

    - ``0.0_1.0_fep.txt`` → (0.0, 1.0)
    - ``0.25_0.75_simulation.dat`` → (0.25, 0.75)

    **indices=[1, 3], separator="_":**

    - ``run_0.0_data_1.0_output.txt`` → (0.0, 1.0)
    - ``sim_0.25_info_0.75_result.dat`` → (0.25, 0.75)

    **indices=[0, 2], separator="-":**

    - ``0.0-data-1.0.txt`` → (0.0, 1.0)
    - ``0.25-sim-0.75.dat`` → (0.25, 0.75)

    **indices=[-2, -1], separator="_" (negative indexing):**

    - ``prefix_data_0.0_1.0.txt`` → (0.0, 1.0)
    - ``long_filename_with_many_parts_0.25_0.75.dat`` → (0.25, 0.75)

    The function automatically handles compressed files (.gz, .bz2) by removing the
    compression extensions before parsing.

    This module is compatible with the standard outputs of `generate_alchemical_lammps_inputs <https://github.com/usnistgov/generate_alchemical_lammps_inputs>`_.

    Parameters
    ----------
    filename : str
        Filename and path. The filename (excluding path and file extension) should
        contain numerical lambda values separated by the specified separator.
    separator : str, default="_"
        Separator used to breakup the filename. The choice in ``indices`` is dependent on this choice.
    indices : list, default=[2, 3]
        Indices used to pull :math:`\lambda` and :math:`\lambda'` from the filename
        components after splitting by separator. Supports both positive and negative indexing.
    prec : int, default=4
        Number of decimal points in the output.

    Returns
    -------
    tuple[float]
        Tuple of lambda values (:math:`\lambda`, :math:`\lambda'`)

    Raises
    ------
    ValueError
        If the specified indices cannot be converted to float values.

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


def lambda_from_filename(
    filename: str, separator: str = "_", index: int = -1, prec: int = 4
) -> float:
    r"""Pull the :math:`\lambda'` value, as defined by the filenames.

    Here :math:`\lambda'` is the scaling value applied to a configuration that is equilibrated to
    a different value of :math:`\lambda`.

    Parameters
    ----------
    filename : str
        Filename and path
    separator : str, default="_"
        Separator used to breakup the filename. The choice in ``index`` is dependent on this choice.
    index : int, default=-1
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
    except (ValueError, IndexError):
        raise ValueError(
            f"Entry, {index} in filename cannot be converted to float: {name_array[index]}"
        )
    return round(value, prec)


def get_bar_lambdas(
    fep_files: str | list[str],
    indices: list[int] = [2, 3],
    prec: int = 4,
    force: bool = False,
) -> tuple[list[float], list[tuple[float, ...]], float | None]:
    """Retrieves all lambda values from FEP filenames.

    Parameters
    ----------
    fep_files: str or list of str
        Path(s) to fepout files to extract data from.
    indices : list[int], default=[2,3]
        In provided file names, using underscore as a separator, these indices mark the part of the filename
        containing the lambda information. If three values, implies a value of lambda2 is present.
        See :func:`tuple_from_filename` for details on how indices are used to extract lambda values.
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
        tuple_from_filename(y, indices=indices, prec=prec) for y in fep_files
    ]
    if len(indices) == 3:
        lambda2_list = list(
            set(
                [
                    lambda_from_filename(y, index=indices[2], prec=prec)
                    for y in fep_files
                ]
            )
        )
        if len(lambda2_list) > 1:
            raise ValueError(
                "More than one value of lambda2 is present in the provided files."
                f" Restrict filename input to one of: {lambda2_list}"
            )
        else:
            lambda2: float | None = lambda2_list[0]
    else:
        lambda2 = None

    lambda_values = sorted(list(set([x for y in lambda_pairs for x in y])))

    if [x for x in lambda_values if round(float(x), prec) < 0]:
        raise ValueError("Lambda values must be positive: {}".format(lambda_values))

    # check that all needed lamba combinations are present
    lambda_dict = {x: [y[1] for y in lambda_pairs if y[0] == x] for x in lambda_values}

    # Check for MBAR content
    missing_combinations_mbar = []
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

    if missing_combinations_bar and not force:
        raise ValueError(
            "BAR calculation cannot be performed without the following lambda-lambda prime combinations: {}".format(
                missing_combinations_bar
            )
        )

    return lambda_values, lambda_pairs, lambda2


@_init_attrs
def extract_u_nk_from_u_n(
    fep_files: str | list[str],
    T: float,
    column_lambda: int,
    column_U: int,
    column_U_cross: int,
    dependence: Callable[[float], float] = lambda x: (x),
    index: int = -1,
    units: str = "real",
    prec: int = 4,
    ensemble: str = "nvt",
    pressure: float | None = None,
    column_volume: int = 4,
) -> pd.DataFrame:
    """Produce u_nk from files containing u_n given a separable dependence on lambda.

    Parameters
    ----------
    fep_files : str or list
        If not a list, a str representing the path to fepout file(s) to extract data from. Filenames and paths are
        aggregated using `glob <https://docs.python.org/3/library/glob.html>`_. For example, "/path/to/files/something_*.txt".
    T : float
        Temperature in Kelvin at which the simulation was sampled.
    column_lambda : int
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
        containing the lambda information for :func:`alchemlyb.parsing.lambda_from_filename`. If ``column_lambda2 != None``
        this list should be of length three, where the last value represents the invariant lambda.
    units : str, default="real"
        Unit system used in LAMMPS calculation. Currently supported: "cgs", "electron", "lj", "metal", "micro", "nano",
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

    _validate_ensemble_parameters(ensemble, pressure)

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
        set([lambda_from_filename(y, index=index, prec=prec) for y in files])
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
                    # pressure is guaranteed to be float by _validate_ensemble_parameters
                    assert pressure is not None
                    u_nk.loc[u_nk[lambda1_col] == lambda1, lambda12] += (
                        beta * pressure * tmp_df["volume"] * energy_from_units(units)
                    )

    u_nk.set_index(["time", "fep-lambda"], inplace=True)
    u_nk.name = "u_nk"  # type: ignore[attr-defined]

    return u_nk


@_init_attrs
def extract_u_nk(
    fep_files: str | list[str],
    T: float,
    columns_lambda1: list[int] = [1, 2],
    column_dU: int = 4,
    column_U: int = 3,
    column_lambda2: int | None = None,
    indices: list[int] = [1, 2],
    units: str = "real",
    vdw_lambda: int = 1,
    ensemble: str = "nvt",
    pressure: float | None = None,
    column_volume: int = 6,
    prec: int = 4,
    force: bool = False,
    tol: float | None = None,
) -> pd.DataFrame:
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
    column_lambda2 : int, default=None
        Index for column (column number minus one) for the unchanging value of lambda for another potential.
        If ``None`` then we do not expect two lambda values being varied.
    indices : list[int], default=[1,2]
        In provided file names, using underscore as a separator, these indices mark the part of the filename
        containing the lambda information for :func:`alchemlyb.parsing.get_bar_lambdas`. If ``column_lambda2 != None``
        this list should be of length three, where the last value represents the invariant lambda.
    units : str, default="real"
        Unit system used in LAMMPS calculation. Currently supported: "cgs", "electron", "lj", "metal", "micro", "nano",
        "real", "si"
    vdw_lambda : int, default=1
        In the case that ``column_lambda2 is not None``, this integer represents which lambda represents vdw interactions.
    ensemble : str, default="nvt"
        Ensemble from which the given data was generated. Either "nvt" or "npt" is supported where values from NVT are
        unaltered, while those from NPT are corrected
    pressure : float, default=None
        The pressure of the system in the NPT ensemble in units of energy / volume, where the units of energy and volume
        are as recorded in the LAMMPS dump file.
    column_volume : int, default=6
        The column for the volume in a LAMMPS dump file.
    prec : int, default=4
        Number of decimal places defined used in ``round()`` function.
    force : bool, default=False
        If ``True`` the dataframe will be created, even if not all lambda and lambda prime combinations are available.
    tol : float, default=None
        Tolerance in checking that the difference between lambda and lambda' states is zero. If None, this tolerance is set
        to ``np.finfo(float).eps``. Take care in increasing this value! It's more likely that something is wrong with your
        column indexing.

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

    if tol is None:
        tol = np.finfo(float).eps

    # Collect Files
    if isinstance(fep_files, list):
        files = fep_files
    else:
        files = glob.glob(fep_files)

    if not files:
        raise ValueError(f"No files have been found that match: {fep_files}")

    _validate_ensemble_parameters(ensemble, pressure)

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

    lambda_values, _, lambda2 = get_bar_lambdas(
        files, indices=indices, prec=prec, force=force
    )

    if column_lambda2 is not None and lambda2 is None:
        raise ValueError(
            "If column_lambda2 is defined, the length of `indices` should be 3 indicating the value of the "
            "second value of lambda held constant."
        )

    # Set-up u_nk and column names / indices
    if column_lambda2 is None:  # No second lambda state value
        u_nk = pd.DataFrame(
            columns=["time", "fep-lambda"] + [str(lv) for lv in lambda_values]
        )
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
        columns_b_simple: list[str] = [str(lv) for lv in lambda_values]  # u_nk cols > 1
        columns_b_tuple: list[str] = []  # Not used in this branch
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
            # lambda2 is guaranteed to be float when column_lambda2 is not None
            assert lambda2 is not None
            columns_b_simple = []  # Not used in this branch
            columns_b_tuple = [
                str((lambda2, x)) for x in lambda_values
            ]  # u_nk cols > 2
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
            # lambda2 is guaranteed to be float when column_lambda2 is not None
            assert lambda2 is not None
            columns_b_simple = []  # Not used in this branch
            columns_b_tuple = [
                str((x, lambda2)) for x in lambda_values
            ]  # u_nk cols > 2
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
                    column_name_value = lambda_values[column_list[0]]

                tmp_df2 = tmp_df.loc[tmp_df[lambda1_2_col] == lambda12]

                lr = tmp_df2.shape[0]
                if u_nk[u_nk[lambda1_col] == lambda1].shape[0] == 0:
                    # If u_nk doesn't contain rows for this lambda state,
                    # Create rows with values of zero to populate energies
                    # from lambda' values
                    df_columns = (
                        columns_b_simple if column_lambda2 is None else columns_b_tuple
                    )
                    tmp_df3 = pd.concat(
                        [
                            tmp_df2[columns_a],
                            pd.DataFrame(
                                np.zeros((lr, lc)),
                                columns=df_columns,
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
                    # lambda2 is guaranteed to be float by validation above
                    assert lambda2 is not None
                    column_name: str = str(
                        (lambda2, column_name_value)
                        if vdw_lambda == 1
                        else (column_name_value, lambda2)
                    )
                else:
                    column_name = str(column_name_value)

                column_index = list(u_nk.columns).index(column_name)
                row_indices = np.where(u_nk[lambda1_col] == lambda1)[0]

                if u_nk.iloc[row_indices, column_index][0] != 0:
                    raise ValueError(
                        "Energy values already available for lambda, {}, lambda', {}. Check for a duplicate file.".format(
                            lambda1, lambda12
                        )
                    )
                if lambda1 == lambda12 and not np.all(tmp_df2["dU_nk"][0] <= tol):
                    raise ValueError(
                        f"The difference in dU should be zero when lambda = lambda', {lambda1} = {lambda12}, not "
                        f"{np.max(tmp_df2['dU_nk'][0])}. Check that 'column_dU' was defined correctly or increase `tol`"
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
                    # pressure is guaranteed to be float by _validate_ensemble_parameters
                    assert pressure is not None
                    u_nk.iloc[row_indices, column_index] += (
                        beta * pressure * tmp_df2["volume"] * energy_from_units(units)
                    )

    if column_lambda2 is None:
        u_nk.set_index(["time", "fep-lambda"], inplace=True)
    else:
        u_nk.set_index(["time", "coul-lambda", "vdw-lambda"], inplace=True)
    u_nk.name = "u_nk"  # type: ignore[attr-defined]

    u_nk = u_nk.dropna()

    return u_nk


@_init_attrs
def extract_dHdl_from_u_n(
    fep_files: str | list[str],
    T: float,
    column_lambda: int | None = None,
    column_u_cross: int | None = None,
    dependence: Callable[[float], float] = lambda x: (1 / x),
    units: str = "real",
    prec: int = 4,
) -> pd.DataFrame:
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
    column_lambda : int, default=None
        Indices for columns (file column number minus one) representing the lambda at which the system is equilibrated
    column_u_cross : int, default=None
        Index for the column (file column number minus one) representing the cross interaction potential energy of the system
    dependence : func, default=`lambda x : (1/x)`
        Transform of lambda needed to convert the potential energy into the derivative of the potential energy with respect to lambda, which must be separable.
        For example, for the LJ potential U = eps * f(sig, r), dU/deps = f(sig, r), so we need a dependence function of 1/eps to convert the
        potential energy to the derivative with respect to eps.
    units : str, default="real"
        Unit system used in LAMMPS calculation. Currently supported: "cgs", "electron", "lj", "metal", "micro", "nano",
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
        if any(x >= lx for x in col_indices):
            raise ValueError(
                "Number of columns, {}, is less than index: {}".format(lx, col_indices)
            )

        data = data.iloc[:, col_indices]

        data.columns = ["time", "fep-lambda", "U"]
        data["fep-lambda"] = data["fep-lambda"].apply(lambda x: round(x, prec))
        data["fep"] = data["fep-lambda"].apply(dependence) * data.U
        data.drop(columns=["U"], inplace=True)

        dHdl = pd.concat([dHdl, data], axis=0, sort=False) if len(dHdl) != 0 else data

    dHdl.set_index(["time", "fep-lambda"], inplace=True)
    dHdl["fep"] = dHdl["fep"] * beta
    dHdl.name = "dH_dl"  # type: ignore[attr-defined]

    return dHdl


@_init_attrs
def extract_dHdl(
    fep_files: str | list[str],
    T: float,
    column_lambda1: int = 1,
    column_dlambda1: int = 2,
    column_lambda2: int | None = None,
    columns_derivative: list[int] = [8, 7],
    vdw_lambda: int = 1,
    units: str = "real",
    prec: int = 4,
) -> pd.DataFrame:
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
    column_lambda1 : int, default=1
        Index for column (column number minus one) representing the lambda at which the system is equilibrated.
    column_dlambda1 : int, default=2
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
        Unit system used in LAMMPS calculation. Currently supported: "cgs", "electron", "lj", "metal", "micro", "nano",
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
        if any(x >= lx for x in col_indices):
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
            else:
                raise ValueError(
                    f"'vdw_lambda must be either 1 or 2, not: {vdw_lambda}'"
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
        dHdl["fep"] = dHdl["fep"] * beta
    else:
        dHdl.set_index(["time", "coul-lambda", "vdw-lambda"], inplace=True)
        if vdw_lambda == 1:
            dHdl["vdw"] = dHdl["vdw"] * beta
        else:
            dHdl["coul"] = dHdl["coul"] * beta

    dHdl.name = "dH_dl"  # type: ignore[attr-defined]

    return dHdl


@_init_attrs
def extract_H(
    fep_files: str | list[str],
    T: float,
    column_lambda1: int = 1,
    column_pe: int = 5,
    column_lambda2: int | None = None,
    units: str = "real",
    ensemble: str = "nvt",
    pressure: float | None = None,
    column_volume: int = 6,
) -> pd.DataFrame:
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
        Unit system used in LAMMPS calculation. Currently supported: "cgs", "electron", "lj", "metal", "micro", "nano",
        "real", "si"
    ensemble : str, default="nvt"
        Ensemble from which the given data was generated. Either "nvt" or "npt" is supported where values from NVT are
        unaltered, while those from NPT are corrected
    pressure : float, default=None
        The pressure of the system in the NPT ensemble in units of energy / volume, where the units of energy and volume
        are as recorded in the LAMMPS dump file.
    column_volume : int, default=6
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

    _validate_ensemble_parameters(ensemble, pressure)

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
        if any(x >= lx for x in col_indices):
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
            # pressure is guaranteed to be float by _validate_ensemble_parameters
            assert pressure is not None
            data["u_n"] += beta * pressure * data["volume"] * energy_from_units(units)
            del data["volume"]

        df_H = pd.concat([df_H, data], axis=0, sort=False) if len(df_H) != 0 else data

    if column_lambda2 is None:
        df_H.set_index(["time", "fep-lambda"], inplace=True)
    else:
        df_H.set_index(["time", "coul-lambda", "vdw-lambda"], inplace=True)

    return df_H
