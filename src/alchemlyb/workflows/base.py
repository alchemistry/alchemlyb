import os
import logging

class WorkflowBase():
    '''Base class for workflow creation.
    '''
    def __init__(self, **kwargs):
        self.logger = logging.getLogger('Initialise Alchemlyb Workflow')
        self.load_data(**kwargs)
        self.sub_sampling(**kwargs)
        self.sub_sampling(**kwargs)
        self.compute(**kwargs)
        self.plot(**kwargs)
        self.write(**kwargs)

    def load_data(self, software='Gromacs', dir='./', prefix='dhdl',
                  suffix='xvg', T=298):
        self.logger.info('Finding files with prefix: {}, suffix: {} under '
                         'directory {} produced by {}'.format(prefix, suffix,
                                                              dir, software))
        xvg_list = []
        file_list = os.listdir(dir)
        for file in file_list:
            if file[:len(prefix)] == prefix and file[-len(prefix):] == suffix:
                xvg_list.append(os.path.join(dir, file))

        self.logger.info('Found {} files.'.format(len(xvg_list)))
        self.logger.debug('File list: \n {}'.format('\n'.join(xvg_list)))
        return xvg_list

    def sub_sampling(self):
        pass

    def compute(self):
        pass

    def plot(self):
        pass

    def write(self):
        pass


