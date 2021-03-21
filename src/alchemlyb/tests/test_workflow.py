import pandas as pd
import numpy as np
import pytest
import os

from alchemlyb.workflows import ABFE
from alchemtest.gmx import load_ABFE, load_expanded_ensemble_case_1

# data = load_ABFE()
# dir = os.path.dirname(data['data']['complex'][0])
# workflow = ABFE(dir=dir, T=310)
# workflow.preprocess()
# workflow.estimate()
# workflow.write()
# workflow.plot_overlap_matrix()
# workflow.plot_ti_dhdl()
# workflow.plot_dF_state()
# workflow.check_convergence(10)
#
# workflow = ABFE(dir=dir, T=310)

def test_full_automatic():
    # Obtain the path of the data
    dir = os.path.dirname(load_ABFE()['data']['complex'][0])
    workflow = ABFE(units='kcal/mol', software='Gromacs', dir=dir,
    prefix='dhdl', suffix='xvg', T=310, skiptime=10,
    uncorr='dhdl', threshold=50,
    methods=('mbar', 'bar', 'ti'), out='./',
    resultfilename='result.out', overlap='O_MBAR.pdf',
    breakdown=True, forwrev=10, log='result.log')
    print(workflow.convergence)
test_full_automatic()