import pandas as pd

class WorkflowBase():
    """The base class for the Workflow.

    Parameters
    ----------

    units : string, optional
        The unit used for printing and plotting results. {'kcal/mol', 'kJ/mol',
        'kT'}. Default: 'kT'.

    software : string, optional
        The software used for generating input. {'Gromacs', }

    T : float, optional,
        Temperature in K. Default: 298.

    out : string, optional
        Directory in which the output files produced by this script will be
        stored. Default: './'.

    Attributes
    ----------

    file_list : list
        A list of files to be read by the parser.

    """
    def __init__(self, units='kT', software='Gromacs', T=298, out='./', *args,
    **kwargs):

        self.T = T
        self.software = software
        self.unit = units
        self.file_list = []
        self.out = out

    def run(self, *args, **kwargs):
        """ Run the flow in an automatic fashion.

        Attributes
        ----------

        u_nk_list : list
            A list of :class:`pandas.DataFrame` of u_nk.
        dHdl_list : list
            A list of :class:`pandas.DataFrame` of dHdl.
        u_nk_sample_list : list
            A list of :class:`pandas.DataFrame` of the subsampled u_nk.
        dHdl_sample_list : list
            A list of :class:`pandas.DataFrame` of the subsampled dHdl.
        result : pandas.Dataframe
            The main result of the workflow.
        convergence : pandas.Dataframe
            The result of the convergence analysis.

        """
        self.read()
        self.preprocess()
        self.estimate()
        self.check_convergence()
        self.plot()

    def read(self, *args, **kwargs):
        """ The function that reads the files in `file_list` and parse them
        into u_nk and dHdl files.

        Attributes
        ----------

        u_nk_list : list
            A list of :class:`pandas.DataFrame` of u_nk.
        dHdl_list : list
            A list of :class:`pandas.DataFrame` of dHdl.

        """
        self.u_nk_list = []
        self.dHdl_list = []

    def preprocess(self, *args, **kwargs):
        """ The function that subsample the u_nk and dHdl in `u_nk_list` and
        `dHdl_list`.

        Attributes
        ----------

        u_nk_sample_list : list
            A list of :class:`pandas.DataFrame` of the subsampled u_nk.
        dHdl_sample_list : list
            A list of :class:`pandas.DataFrame` of the subsampled dHdl.

        """
        self.dHdl_sample_list = []
        self.u_nk_sample_list = []

    def estimate(self, *args, **kwargs):
        """ The function that runs the estimator based on `u_nk_sample_list`
        and `dHdl_sample_list`.

        Attributes
        ----------

        result : pandas.Dataframe
            The main result of the workflow.

        """
        self.result = pd.DataFrame()

    def check_convergence(self, *args, **kwargs):
        """ The function for doing convergence analysis.

        Attributes
        ----------

        convergence : pandas.Dataframe
            The result of the convergence analysis.

        """
        self.convergence = pd.DataFrame()

    def plot(self, *args, **kwargs):
        """ The function for producing any plots.

        """
        pass
