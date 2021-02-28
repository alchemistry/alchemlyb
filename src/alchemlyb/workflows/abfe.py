import pandas as pd
import logging

from base import WorkflowBase
from ..parsing import gmx

class ABFE(WorkflowBase):
    def load_data(self, software='Gromacs', dir='./', prefix='dhdl',
                  suffix='xvg', T=298):
        xvg_list = super().load_data(dir=dir, prefix=prefix, suffix=suffix)
        if software == 'Gromacs':
            self.logger.info('Using {} parser to read the data.'.format(
                software))
            try:
                u_nk_list = pd.concat(
                    [gmx.extract_u_nk(xvg, T=T) for xvg in xvg_list])
            except:
                self.logger.warning('Could not read u_nk data.')
            try:
                dHdl_list = pd.concat(
                    [gmx.extract_dHdl(xvg, T=T) for xvg in xvg_list])
            except:
                self.logger.warning('Could not read dHdl data.')


