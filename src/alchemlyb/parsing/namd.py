"""Parsers for extracting alchemical data from `NAMD <http://www.ks.uiuc.edu/Research/namd/>`_ output files.

"""
import pandas as pd
import numpy as np
from .util import anyopen

# TODO: perhaps move constants elsewhere?
# these are the units we need for dealing with NAMD, which uses
# kcal/mol for energies  http://www.ks.uiuc.edu/Research/namd/2.13/ug/node12.html#SECTION00062200000000000000
# (kB in kcal/molK)
k_b = 1.9872041e-3


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

    """
    beta = 1/(k_b * T)

    # lists to get times and work values of each window
    win_ts = []
    win_de = []

    # create dataframe for results
    u_nk = pd.DataFrame(columns=['time','fep-lambda'])

    # boolean flag to parse data after equil time
    parsing = False

    # open and get data from fep file.
    with anyopen(fep_file, 'r') as f:
        for line in f:
            l = line.strip().split()

            # this line marks end of window; dump data into dataframe
            if '#Free' in l:

                # convert last window's work and times values to np arrays
                win_de_arr = beta * np.asarray(win_de)
                win_ts_arr = np.asarray(win_ts)

                # extract lambda values for finished window
                # lambda1 = sampling lambda (row), lambda2 = evaluated lambda (col)
                lambda1 = "{0:.2f}".format(float(l[7]))
                lambda2 = "{0:.2f}".format(float(l[8]))

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
                parsing = False

            # append work value from 'dE' column of fepout file
            if parsing:
                win_de.append(float(l[6]))
                win_ts.append(float(l[1]))

            # turn parsing on after line 'STARTING COLLECTION OF ENSEMBLE AVERAGE'
            if '#STARTING' in l:
                parsing = True

    # create last dataframe for fep-lambda at last LAMBDA2
    tempDF = pd.DataFrame({
        'time': win_ts_arr,
        'fep-lambda': lambda2})

    u_nk = pd.concat([u_nk, tempDF], sort=True)

    u_nk.set_index(['time','fep-lambda'], inplace=True)

    return u_nk
