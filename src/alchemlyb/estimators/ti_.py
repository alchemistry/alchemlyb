import pandas as pd
import scipy.integrate


def thermodynamic_integration(dHdls):
    """

    """


    DG = pd.Series([simps(dHdl['mean'].iloc[:i], x=dHdl.index[:i], even='last') for i in range(2, len(dHdl)+1)], index=dHdl.index[1:])



    return ti 
