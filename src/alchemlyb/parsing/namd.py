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

    For a given lambda1, there can be only one lambda_idws.

    Parameters
    ----------
    fep_files: str or list of str
        Path(s) to fepout files to extract data from.

    Returns
    -------
    List of floats, or None if there is more than one lambda_idws for each lambda1.
    """

    lambda_fwd_map, lambda_bwd_map = {}, {}

    for fep_file in sorted(fep_files):
        with anyopen(fep_file, 'r') as f:
            for line in f:
                l = line.strip().split()
                if l[0] not in ['#NEW', '#Free']:
                    continue

                # We might not have a #NEW line so make the best guess
                if l[0] == '#NEW':
                    lambda1, lambda2 = float(l[6]), float(l[8])
                    lambda_idws = float(l[10]) if 'LAMBDA_IDWS' in l else None
                elif l[0] == '#Free':
                    lambda1, lambda2, lambda_idws = float(l[7]), float(l[8]), None

                # Make sure the lambda2 values are consistent
                if lambda1 in lambda_fwd_map and lambda_fwd_map[lambda1] != lambda2:
                    logger.error(f'fwd: lambda1 {lambda1} has lambda2 {lambda_fwd_map[lambda1]} but it should be {lambda2}')
                    return None

                lambda_fwd_map[lambda1] = lambda2

                # Make sure the lambda_idws values are consistent
                if lambda_idws is not None:
                    if lambda1 in lambda_bwd_map and lambda_bwd_map[lambda1] != lambda_idws:
                        logger.error(f'namd.py: get_lambdas: Error: bwd: lambda1 {lambda1} has lambda_idws {lambda_bwd_map[lambda1]} but it should be {lambda_idws}')
                        return None
                    lambda_bwd_map[lambda1] = lambda_idws

    all_lambdas = set()
    all_lambdas.update(lambda_fwd_map.keys())
    all_lambdas.update(lambda_fwd_map.values())
    all_lambdas.update(lambda_bwd_map.keys())
    all_lambdas.update(lambda_bwd_map.values())
    return list(sorted(all_lambdas))


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
    lambda_idws = None

    if type(fep_files) is str:
        fep_files = [fep_files]

    time = 0
    # Extract the lambda values only from the fepouts
    all_lambdas = _get_lambdas(fep_files)
    # open and get data from fep file.
    # We sort the list of fep files in case some of them represent restarted windows.
    # The assumption is that they make sense in lexicographic order.
    for fep_file in sorted(fep_files):
        lambda1_at_start, lambda2_at_start, lambda_idws_at_start = None, None, None
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
                    lambda1_idx = all_lambdas.index(lambda1)
                    if has_idws is True and lambda1_idx > 0:
                        lambda_idws = all_lambdas[lambda1_idx - 1]
                    else:
                        lambda_idws = None

                    # If the lambdas are not what we thought they would be, return None, ensuring the calculation
                    # fails. This can happen if fepouts where one window spans multiple fepouts are processed out of order
                    if lambda1_at_start is not None \
                        and (lambda1, lambda2, lambda_idws) != (lambda1_at_start, lambda2_at_start, lambda_idws_at_start):
                        logger.error("namd.py: extract_u_nk: Error: Lambdas changed unexpectedly while processing", fep_file)
                        logger.error(f"namd.py: extract_u_nk: Error: l1, l2, lidws: {lambda1_at_start}, {lambda2_at_start}, {lambda_idws_at_start} changed to {lambda1}, {lambda2}, {lambda_idws}")
                        logger.error(f"namd.py: extract_u_nk: Error: fep_file = {fep_file}; has_idws = {has_idws}")
                        return None

                    # As we are at the end of a window, convert last window's work and times values to np arrays
                    # (with energy unit kT since they were kcal/mol in the fepouts)
                    win_de_arr = beta * np.asarray(win_de) # dE values
                    win_ts_arr = np.asarray(win_ts) # timesteps

                    if lambda_idws is not None:
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
                            lambda_idws: win_de_back_arr[:n]})

                    else:
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

    if len(win_de) != 0 or len(win_de_back) != 0:
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
