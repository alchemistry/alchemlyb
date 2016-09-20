

def get_DG(sims, name, lower, upper):
    """Get DG and DDG from set of simulations. Does automatic subsampling
    for each simulation on the basis of automated equilibrium detection on its
    own reduced potential timeseries.
    
    Parameters
    ----------
    sims : Bundle
        Bundle of sims to grab data from.
    name : str
        Name of dataset to use.
    lower : float, dict
        Time (ps) to start block from. Could also be a dict 
        giving state number as keys and float as value.
    upper : float
        Time (ps) to end block at. Could also be a dict 
        giving state number as keys and float as value.
        
    Returns
    -------
    DG : array
        Delta G between each state as calculated by MBAR.
    DDG : array
        Standard deviation of Delta G between each state as calculated by MBAR.
    
    """
    import numpy as np
    import pandas as pd
    from pymbar import MBAR
    from pymbar.timeseries import detectEquilibration
    
    import gc
    
    states = sorted(list(set(sims.categories['state'])))
    
    if isinstance(lower, (float, int)) or lower is None:
        lower = {state: lower for state in states}
            
    if isinstance(upper, (float, int)) or upper is None:
        upper = {state: upper for state in states}
    
    dfs = []
    N_k = []
    
    groups = sims.categories.groupby('state')
    for state in groups:
        dfs_g = []
        for sim in groups[state]:

            # get data for every `step`
            df = sim.data[name].loc[lower[state]:upper[state]]

            # subsample according to statistical inefficiency after equilibration detection
            # we do this after slicing by lower/upper to simulate
            # what we'd get with only this data available
            t, statinef, Neff_max = detectEquilibration(df[df.columns[sim.categories['state']]])

            # we round up
            statinef = int(np.rint(statinef))

            # subsample according to statistical inefficiency and equilibrium detection
            df = df.iloc[t::statinef]

            # extract only columns that have the corresponding sim present        
            df = df[df.columns[states]]

            # drop any NA rows, which can happen from subsampled data
            df = df.dropna()
            
            dfs_g.append(df)
            
        df = np.vstack(dfs_g)
        dfs.append(df)

        N_k.append(len(df))
        del df
        del dfs_g
        gc.collect()
        
    u_kn = np.vstack(dfs).T
    del dfs
    gc.collect()
    
    mbar = MBAR(u_kn, N_k)
    
    DG, DDG = mbar.getFreeEnergyDifferences()
    
    return DG, DDG

def get_neighbor_DG(sims, name, lower, upper, neighbor='right'):
    import pandas as pd
    
    lambdas = sorted(list(set(sims.categories['coul-lambda'])))[:-1]
    
    k_b = 8.3144621E-3
    T = 310
    
    if neighbor == 'right':
        neigh = 1
    elif neighbor == 'left':
        neigh = -1
    else:
        raise ValueError("neighbor must be right or left")
        
    mbar = get_DG(sims=sims, name=name, lower=lower, upper=upper)
    
    df = pd.DataFrame({'DG': k_b * T * mbar[0].diagonal(neigh), 
                      'std': k_b * T * mbar[1].diagonal(neigh)},
                      columns=['DG', 'std'],
                      index=pd.Float64Index(lambdas))

    return df

def get_total_DG(sims, name, lower, upper):
    import pandas as pd
    
    k_b = 8.3144621E-3
    T = 310
        
    mbar = get_DG(sims=sims, name=name, lower=lower, upper=upper)
    
    df = pd.DataFrame({'DG': k_b * T * mbar[0][0,-1:], 
                      'std': k_b * T * mbar[1][0,-1:]},
                      columns=['DG', 'std'])

    return df
