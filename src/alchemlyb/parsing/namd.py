"""Parsers for extracting alchemical data from `NAMD <http://www.ks.uiuc.edu/Research/namd/>`_ output files.

"""
import pandas as pd
import numpy as np
import logging
from .util import anyopen
from . import _init_attrs
from ..postprocessors.units import R_kJmol, kJ2kcal

logger = logging.getLogger("alchemlyb.parsers.NAMD")

k_b = R_kJmol * kJ2kcal


def _get_lambdas(fep_files):
    """Retrieves all lambda values included in the FEP files provided.
    
    We have to do this in order to tolerate truncated and restarted fepout files.
    The IDWS lambda is not present at the termination of the window, presumably
    for backwards compatibility with ParseFEP and probably other things.

    For a given lambda1, there can be only one lambda2 and at most one lambda_idws.

    Parameters
    ----------
    fep_files: str or list of str
        Path(s) to fepout files to extract data from.

    Returns
    -------
    List of floats, or None if there is more than one lambda_idws for each lambda1.
    """

    lambda_fwd_map, lambda_bwd_map = {}, {}
    is_ascending = set()
    endpoint_windows = []

    for fep_file in sorted(fep_files):
        with anyopen(fep_file, 'r') as f:
            for line in f:
                l = line.strip().split()

                # We might not have a #NEW line so make the best guess
                if l[0] == '#NEW':
                    lambda1, lambda2 = float(l[6]), float(l[8])
                    lambda_idws = float(l[10]) if 'LAMBDA_IDWS' in l else None
                elif l[0] == '#Free':
                    lambda1, lambda2, lambda_idws = float(l[7]), float(l[8]), None
                else:
                    # We only care about lines with lambda values. No need to
                    # do all that other processing below for every line
                    continue

                # Keep track of whether the lambda values are increasing or decreasing, so we can return
                # a sorted list of the lambdas in the correct order.
                # If it changes during parsing of this set of fepout files, then we know something is wrong
                
                # Keep track of endpoints separately since in IDWS runs there must be one of opposite direction
                if 0.0 in (lambda1, lambda2) or 1.0 in (lambda1, lambda2):
                    endpoint_windows.append((lambda1, lambda2))
                else:
                    # If the lambdas are equal then this doesn't represent an ascending window
                    if lambda2 != lambda1:
                        is_ascending.add(lambda2 > lambda1)
                    if lambda_idws is not None and lambda1 != lambda_idws:
                        is_ascending.add(lambda1 > lambda_idws)

                if len(is_ascending) > 1:
                    raise ValueError(f'Lambda values change direction in {fep_file}, relative to the other files: {lambda1} -> {lambda2} (IDWS: {lambda_idws})')

                # Make sure the lambda2 values are consistent
                if lambda1 in lambda_fwd_map and lambda_fwd_map[lambda1] != lambda2:
                    logger.error(f'fwd: lambda1 {lambda1} has lambda2 {lambda_fwd_map[lambda1]} in {fep_file} but it has already been {lambda2}')
                    raise ValueError('More than one lambda2 value for a particular lambda1')

                lambda_fwd_map[lambda1] = lambda2

                # Make sure the lambda_idws values are consistent
                if lambda_idws is not None:
                    if lambda1 in lambda_bwd_map and lambda_bwd_map[lambda1] != lambda_idws:
                        logger.error(f'bwd: lambda1 {lambda1} has lambda_idws {lambda_bwd_map[lambda1]} but it has already been {lambda_idws}')
                        raise ValueError('More than one lambda_idws value for a particular lambda1')
                    lambda_bwd_map[lambda1] = lambda_idws


    is_ascending = next(iter(is_ascending))

    all_lambdas = set()
    all_lambdas.update(lambda_fwd_map.keys())
    all_lambdas.update(lambda_fwd_map.values())
    all_lambdas.update(lambda_bwd_map.keys())
    all_lambdas.update(lambda_bwd_map.values())
    return list(sorted(all_lambdas, reverse=not is_ascending))


@_init_attrs
def extract_u_nk(fep_files, T):
    """Return reduced potentials `u_nk` from NAMD fepout file(s).

    Parameters
    ----------
    fep_file : str or list of str
        Path to fepout file(s) to extract data from.
    T : float
        Temperature in Kelvin at which the simulation was sampled.

    Returns
    -------
    u_nk : DataFrame
        Potential energy for each alchemical state (k) for each frame (n).

    Note
    ----
    If the number of forward and backward samples in a given window are different,
    the extra sample(s) will be discarded. This is typically zero or one sample.

    .. versionchanged:: 0.5.0
        The :mod:`scipy.constants` is used for parsers instead of
        the constants used by the corresponding MD engine.

    .. versionchanged:: 0.6.0
        Support for Interleaved Double-Wide Sampling files added.

        `fep_files` can now be a list of filenames.
    """
    beta = 1/(k_b * T)

    # lists to get times and work values of each window
    win_ts = []
    win_de = []
    win_ts_back = []
    win_de_back = []

    # create dataframe for results
    u_nk = pd.DataFrame(columns=['time','fep-lambda'])

    # boolean flag to parse data after equil time
    parsing = False

    if type(fep_files) is str:
        fep_files = [fep_files]

    # Extract the lambda values only from the fepouts
    all_lambdas = _get_lambdas(fep_files)
    # open and get data from fep file.
    # We sort the list of fep files in case some of them represent restarted windows.
    # The assumption is that they make sense in lexicographic order.
    # We keep track of which lambda window we're in, but since it can span multiple files,
    # only reset these variables here and after the end of each window
    lambda1_at_start, lambda2_at_start, lambda_idws_at_start = None, None, None
    for fep_file in sorted(fep_files):
        # Note we have not set parsing=False because we could be continuing one window across
        # more than one fepout file
        with anyopen(fep_file, 'r') as f:
            has_idws = False
            for line in f:
                l = line.strip().split()
                # We don't know if IDWS was enabled just from the #Free line, and we might not have
                # a #NEW line in this file, so we have to check for the existence of FepE_back lines
                # We rely on short-circuit evaluation to avoid the string comparison most of the time
                if has_idws is False and l[0] == 'FepE_back:':
                    has_idws = True

                # New window, get IDWS lambda if any
                # We keep track of lambdas from the #NEW line and if they disagree with the #Free line
                # within the same file, then complain. This can happen if truncated fepout files
                # are presented in the wrong order.
                if l[0] == '#NEW':
                    lambda1_at_start, lambda2_at_start = float(l[6]), float(l[8])
                    lambda_idws_at_start = float(l[10]) if 'LAMBDA_IDWS' in l else None
                    has_idws = True if lambda_idws_at_start is not None else False

                # this line marks end of window; dump data into dataframe
                if l[0] == '#Free':
                    # extract lambda values for finished window
                    # lambda1 = sampling lambda (row), lambda2 = comparison lambda (col)
                    lambda1 = float(l[7])
                    lambda2 = float(l[8])

                    # If the lambdas are not what we thought they would be, raise an exception to ensure the calculation
                    # fails. This can happen if fepouts where one window spans multiple fepouts are processed out of order
                    # NB: There is no way to tell if lambda_idws changed because it isn't in the '#Free' line that ends a window
                    if lambda1_at_start is not None \
                        and (lambda1, lambda2) != (lambda1_at_start, lambda2_at_start):
                        logger.error(f"Lambdas changed unexpectedly while processing {fep_file}")
                        logger.error(f"l1, l2: {lambda1_at_start}, {lambda2_at_start} changed to {lambda1}, {lambda2}")
                        logger.error(line)
                        raise ValueError("Inconsistent lambda values within the same window")

                    # As we are at the end of a window, convert last window's work and times values to np arrays
                    # (with energy unit kT since they were kcal/mol in the fepouts)
                    win_de_arr = beta * np.asarray(win_de) # dE values
                    win_ts_arr = np.asarray(win_ts) # timesteps

                    # This handles the special case where there are IDWS energies but no lambda_idws value in the
                    # current .fepout file. This can happen when the NAMD firsttimestep is not 0, because NAMD only emits
                    # the '#NEW' line on timestep 0 for some reason. Perhaps the user ran minimize before dynamics,
                    # or this is a restarted run.
                    # We infer lambda_idws_at_start if it wasn't explictly included in this fepout.
                    # If lambdas are in ascending order, choose the one before it all_lambdas, and if descending, choose
                    # the one after. This happens "automatically" because the lambdas were returned already sorted
                    # in the correct direction by _get_lambdas().
                    # The "else" case is handled by the rest of this block, by default.
                    if has_idws and lambda_idws_at_start is None:
                        l1_idx = all_lambdas.index(lambda1)
                        # Test for the highly pathological case where the first window is both incomplete and has IDWS
                        # data but no lambda_idws value.
                        if l1_idx == 0:
                            raise ValueError(f'IDWS data present in first window but lambda_idws not included; no way to infer the correct lambda_idws')
                        lambda_idws_at_start = all_lambdas[l1_idx - 1]
                        logger.warning(f'Warning: {fep_file} has IDWS data but lambda_idws not included.')
                        logger.warning(f'         lambda1 = {lambda1}, lambda2 = {lambda2}; inferring lambda_idws to be {lambda_idws_at_start}')

                    if lambda_idws_at_start is not None:
                        # Mimic classic DWS data
                        # Arbitrarily match up fwd and bwd comparison energies on the same times
                        # truncate extra samples from whichever array is longer
                        win_de_back_arr = beta * np.asarray(win_de_back)
                        n = min(len(win_de_back_arr), len(win_de_arr))

                        tempDF = pd.DataFrame({
                            'time': win_ts_arr[:n],
                            'fep-lambda': np.full(n,lambda1),
                            lambda1: 0,
                            lambda2: win_de_arr[:n],
                            lambda_idws_at_start: win_de_back_arr[:n]})
                        # print(f"{fep_file}: IDWS window {lambda1} {lambda2} {lambda_idws_at_start}")
                    else:
                        # print(f"{fep_file}: Forward-only window {lambda1} {lambda2}")
                        # create dataframe of times and work values
                        # this window's data goes in row LAMBDA1 and column LAMBDA2
                        tempDF = pd.DataFrame({
                            'time': win_ts_arr,
                            'fep-lambda': np.full(len(win_de_arr), lambda1),
                            lambda1: 0,
                            lambda2: win_de_arr})

                    # join the new window's df to existing df
                    u_nk = pd.concat([u_nk, tempDF], sort=True)

                    # reset values for next window of fepout file
                    win_de = []
                    win_ts = []
                    win_de_back = []
                    win_ts_back = []
                    parsing = False
                    has_idws = False
                    lambda1_at_start, lambda2_at_start, lambda_idws_at_start = None, None, None

                # append work value from 'dE' column of fepout file
                if parsing:
                    if l[0] == 'FepEnergy:':
                        win_de.append(float(l[6]))
                        win_ts.append(float(l[1]))
                    elif l[0] == 'FepE_back:':
                        win_de_back.append(float(l[6]))
                        win_ts_back.append(float(l[1]))

                # Turn parsing on after line 'STARTING COLLECTION OF ENSEMBLE AVERAGE'
                if '#STARTING' in l:
                    parsing = True

    if len(win_de) != 0 or len(win_de_back) != 0: # pragma: no cover
        logger.warning('Trailing data without footer line (\"#Free energy...\"). Interrupted run?')

    if lambda2 in (0.0, 1.0):
        # this excludes the IDWS case where a dataframe already exists for both endpoints
        # create last dataframe for fep-lambda at last LAMBDA2
        tempDF = pd.DataFrame({
            'time': win_ts_arr,
            'fep-lambda': lambda2})
        u_nk = pd.concat([u_nk, tempDF], sort=True)

    u_nk.set_index(['time','fep-lambda'], inplace=True)

    return u_nk
