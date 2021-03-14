import pandas as pd
import numpy as np
import pytest
import os

from alchemlyb.workflows.abfe import ABFE
from alchemtest.gmx import load_ABFE, load_expanded_ensemble_case_1

data = load_ABFE()
dir = os.path.dirname(data['data']['complex'][0])
workflow = ABFE(dir=dir, T=310)
workflow.preprocess()
workflow.estimate()
workflow.write()
workflow.plot_overlap_matrix()
workflow.plot_ti_dhdl()
workflow.plot_dF_state()
workflow.check_convergence(10)