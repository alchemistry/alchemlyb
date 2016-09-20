"""Functions for processing datasets obtained from gromacs.

"""
import pandas as pd

import mdsynthesis as mds

import gromacs
from gromacs.formats import XVG


def generate_xvgs(tpr, files, outdir):
    """Obtain XVGs from EDR files giving Hamiltonian differences for each window.

    Parameters
    ----------
    tpr : str
        Path to TPR file.
    files : list
        List of paths to EDR files to use.
    outdir : path
        Path to directory in which to place XVG files.

    """
    outdir = mds.Tree(outdir).makedirs()
    for edr in files:
        edr_file = mds.Leaf(edr)
        outfile = outdir["dhdl.{}.xvg".format(edr_file.name.split('.')[1])]
        gromacs.energy(f=edr_file.abspath, s=tpr, odh=outfile.abspath)


def get_u_kn(xvg, T=310):
    """Return u_kn from a Hamiltonian differences XVG file.
    
    Parameters
    ----------
    xvg : str
        Path to XVG file to extract data from.
    T : float
        Temperature in Kelvin the simulations sampled.
    
    Returns
    -------
    u_kn : array
        Potential energy for each alchemical state (k) for each frame (n).
    
    """
    col_match = r"\xD\f{}H \xl\f{}"
    k_b = 8.3144621E-3
    beta = 1/(k_b * T)
    
    # extract a DataFrame from XVG data
    xvg = XVG(xvg)
    df = xvg.to_df()

    # drop duplicate columns if we (stupidly) have them
    df = df.iloc[:, ~df.columns.duplicated()]
    
    times = df[df.columns[0]]

    # want to grab only dH columns
    DHcols = [col for col in df.columns if (col_match in col)]
    dH = df[DHcols]
    
    # not entirely sure if we need to get potentials relative to
    # the state actually sampled, but perhaps needed to stack
    # samples from all states?
    U = df[df.columns[1]]

    # gromacs also gives us pV directly; need this for reduced potential
    pV = df[df.columns[-1]]

    u_k = dict()
    cols= list()
    for col in dH:
        u_col = 'u' + col.split('to')[1]
        u_k[u_col] = beta * (dH[col].values + U.values + pV.values)
        cols.append(u_col)
    
    u_k = pd.DataFrame(u_k, columns=cols, index=pd.Float64Index(times.values, name='time (ps)'))
    u_k.name = 'reduced potential'
    
    return u_k


def store_u_kn(tree, xvgs):
    """Store u_kn calculated from each dhdl.xvg segment file
    for the given Sim.
    
    """
    xvgs = sim['WORK/dhdl'].glob('dhdl.*.xvg')
    
    dfs = []
    for xvg in xvgs:
        dfs.append(get_u_kn(xvg.abspath))
        
    df = pd.concat(dfs)
    
    sim.data['u_kn'] = df


def get_tau(sim, dt=.1):
    u_kn = sim.data['u_kn']
    state = sim.categories['state']
    return dt * (statisticalInefficiency(u_kn[u_kn.columns[state]]) - 1)/2


def get_block_ukn(sims, lower, upper, step=1):
    """Get a block of u_kn values from a number of simulations.
    
    Parameters
    ----------
    sims : Bundle
        Bundle of sims to grab data from.
    lower : float
        Time (ps) to start block from.
    upper : float
        Time (ps) to end block at.
    step : int
        Step size between samples.
        
    Returns
    -------
    DG : array
        Delta G between each state as calculated by MBAR.
    DDG : array
        Standard deviation of Delta G between each state as calculated by MBAR.
    
    """
    import numpy as np
    from pymbar import MBAR
    import gc
    
    dfs = []
    N_k = []
    for sim in sims:
        print "\r{}".format(sim.name), 
        
        # get data for every `step`
        df = sim.data['u_kn/subsampled_dt1ps'].loc[range(lower, upper, step)]
        dfs.append(df)
        
        N_k.append(len(df))
        
    u_kn = np.vstack(dfs).T
    del dfs
    gc.collect()
    
    mbar = MBAR(u_kn, N_k)
    
    DG, DDG = mbar.getFreeEnergyDifferences()
    
    return DG, DDG
