import os
import logging

class WorkflowBase():
    '''Base class for workflow creation.
    '''
    def __init__(self, **kwargs):

        self.load_data(**kwargs)
        self.sub_sampling(**kwargs)
        self.sub_sampling(**kwargs)
        self.compute(**kwargs)
        self.plot(**kwargs)
        self.write(**kwargs)

    def load_data(self,
        return xvg_list

    def sub_sampling(self):
        pass

    def compute(self):
        pass

    def plot(self):
        pass

    def write(self):
        pass


