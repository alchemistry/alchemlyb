"""Parsers for extracting alchemical data from `NAMD <http://www.ks.uiuc.edu/Research/namd/>`_ output files.

"""
import pandas as pd
import numpy as np
from .util import anyopen
from . import _init_attrs
from ..postprocessors.units import R_kJmol, kJ2kcal

k_b = R_kJmol * kJ2kcal

@_init_attrs
def extract_u_nk(fep_file, T):
    """Return reduced potentials `u_nk` from NAMD fepout file.

    Parameters
    ----------
    fep_file : str
        Path to fepout file to extract data from.
    T : float
        Temperature in Kelvin at which the simulation was sampled.

    Returns
    -------
    u_nk : DataFrame
        Potential energy for each alchemical state (k) for each frame (n).


    .. versionchanged:: 0.5.0
        The :mod:`scipy.constants` is used for parsers instead of
        the constants used by the corresponding MD engine.

    """
    beta = 1/(k_b * T)

    # lists to get times and work values of each window
    win_ts = []
    win_de = []

    # create dataframe for results
    u_nk = pd.DataFrame(columns=['time','fep-lambda'])

    # boolean flag to parse data after equil time
    parsing = False
    lambda_idws = None

    # open and get data from fep file.
    with anyopen(fep_file, 'r') as f:
        time = 0
        for line in f:
            l = line.strip().split()

            # New window, get IDWS lambda if any
            if l[0] == '#NEW':
                if 'LAMBDA_IDWS' in l:
                    lambda_idws = "{0:.2f}".format(float(l[10]))
                else:
                    lambda_idws = None

            # this line marks end of window; dump data into dataframe
            if '#Free' in l:

                # extract lambda values for finished window
                # lambda1 = sampling lambda (row), lambda2 = evaluated lambda (col)
                lambda1 = "{0:.2f}".format(float(l[7]))
                lambda2 = "{0:.2f}".format(float(l[8]))
                print(lambda1, lambda2, lambda_idws)

                # convert last window's work and times values to np arrays
                win_de_arr = beta * np.asarray(win_de)
                win_ts_arr = np.asarray(win_ts)

                if lambda_idws is not None:
                    # Mimic classic DWS data
                    # Arbitrarily match up fwd and bwd comparison energies on the same times
                    # truncate extra samples from whichever array is longer
                    win_de_back_arr = beta * np.asarray(win_de_back)
                    n = min(len(win_de_back_arr), len(win_de_arr))

                    tempDF_idws = pd.DataFrame({
                        'time': range(n),
                        'fep-lambda': np.full(n,lambda1),
                        lambda1: 0,
                        lambda2: win_de_arr[:n],
                        lambda_idws: win_de_back_arr[:n]})

                    # append to current dataframe
                    u_nk = pd.concat([u_nk, tempDF_idws], sort=True)

                else:
                    # create dataframe of times and work values
                    # this window's data goes in row LAMBDA1 and column LAMBDA2
                    tempDF = pd.DataFrame({
                        'time': win_ts_arr,
                        'fep-lambda': np.full(len(win_de_arr),lambda1),
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

            # turn parsing on after line 'STARTING COLLECTION OF ENSEMBLE AVERAGE'
            if '#STARTING' in l:
                parsing = True

    # TODO test if lambda2 is already in dataframe instead of test below
    if (lambda2 == 1):
        # create last dataframe for fep-lambda at last LAMBDA2
        tempDF = pd.DataFrame({
            'time': win_ts_arr,
            'fep-lambda': lambda2})

        u_nk = pd.concat([u_nk, tempDF], sort=True)

    u_nk.set_index(['time','fep-lambda'], inplace=True)

    return u_nk
