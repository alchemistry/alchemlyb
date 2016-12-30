import numpy as np
import pandas as pd

from pymbar import MBAR

def mbar(dfs):

    for df in dfs:
        N_k.append(len(df))
    
    u_kn = np.vstack(dfs).T
    
    mbar = MBAR(u_kn, N_k)
    
    return mbar
