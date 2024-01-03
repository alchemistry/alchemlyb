import pandas as pd
from loguru import logger

from . import _init_attrs


def _read_parquet_with_metadata(path: str, T: float) -> pd.DataFrame:
    """
    Check if the metadata is included in the Dataframe and has the correct
    temperature.

    Parameters
    ----------
    path : str
        Path to parquet file to extract dataframe from.
    T : float
        Temperature in Kelvin of the simulations.

    Returns
    -------
    DataFrame
    """
    df = pd.read_parquet(path)
    if "temperature" not in df.attrs:
        logger.warning(
            f"No temperature metadata found in {path}. "
            f"Serialise the Dataframe with pandas>=2.1 to preserve the metadata."
        )
        df.attrs["temperature"] = T
        df.attrs["energy_unit"] = "kT"
    else:
        if df.attrs["temperature"] != T:
            raise ValueError(
                f"Temperature in the input ({T}) doesn't match the temperature "
                f"in the dataframe ({df.attrs['temperature']})."
            )
    return df


def extract_u_nk(path, T):
    r"""Return reduced potentials `u_nk` (unit: kT) from a pandas parquet file.

    The parquet file should be serialised from the dataframe output
    from any parser with command
    (``u_nk_df.to_parquet(path=path, index=True)``).

    Parameters
    ----------
    path : str
        Path to parquet file to extract dataframe from.
    T : float
        Temperature in Kelvin of the simulations.

    Returns
    -------
    u_nk : DataFrame
        Potential energy for each alchemical state (k) for each frame (n).


    Note
    ----
    pyarraw serializers would handle the float or string column name fine but will
    convert multi-lambda column name from `(0.0, 0.0)` to `"('0.0', '0.0')"`.
    This parser will restore the correct column name.
    Also parquet serialisation doesn't preserve the :attr:`pandas.DataFrame.attrs`.
    So the temperature is assigned in this function.


    .. versionadded:: 2.1.0

    """
    u_nk = _read_parquet_with_metadata(path, T)
    columns = list(u_nk.columns)
    if isinstance(columns[0], str) and columns[0][0] == "(":
        new_columns = []
        for column in columns:
            new_columns.append(
                tuple(
                    map(
                        float, column[1:-1].replace('"', "").replace("'", "").split(",")
                    )
                )
            )
        u_nk.columns = new_columns
    return u_nk


@_init_attrs
def extract_dHdl(path, T):
    r"""Return gradients `dH/dl` (unit: kT) from a pandas parquet file.

    The parquet file should be serialised from the dataframe output
    from any parser with command
    (`dHdl_df.to_parquet(path=path, index=True)`).

    Parameters
    ----------
    path : str
        Path to parquet file to extract dataframe from.
    T : float
        Temperature in Kelvin the simulations sampled.

    Returns
    -------
    dH/dl : DataFrame
        dH/dl as a function of time for this lambda window.

    Note
    ----
    Parquet serialisation doesn't preserve the :attr:`pandas.DataFrame.attrs`.
    So the temperature is assigned in this function.


    .. versionadded:: 2.1.0

    """
    return _read_parquet_with_metadata(path, T)
