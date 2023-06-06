"""Basic building blocks for free energy workflows."""

from pathlib import Path

import pandas as pd
from loguru import logger

from .. import __version__


class WorkflowBase:
    """The base class for the Workflow.

    This is the base class for the creation of new Workflow. The
    initialisation method takes in the MD engine, unit, temperature and
    output directory. The goal of the initialisation is to check the input
    files and store them in
    :attr:`~alchemlyb.workflows.WorkflowBase.file_list`
    such that they can be read by the
    :func:`~alchemlyb.workflows.WorkflowBase.read` method.

    Parameters
    ----------
    units : string, optional
        The unit used for printing and plotting results. {'kcal/mol', 'kJ/mol',
        'kT'}. Default: 'kT'.

    software : string, optional
        The software used for generating input. {'Gromacs', 'Amber'}

    T : float, optional,
        Temperature in K. Default: 298.

    out : string, optional
        Directory in which the output files produced by this script will be
        stored. Default: './'.

    Attributes
    ----------
    file_list : list
        A list of files to be read by the parser.


    .. versionadded:: 0.7.0
    """

    def __init__(
        self, units="kT", software="Gromacs", T=298, out="./", *args, **kwargs
    ):
        logger.info(f"Alchemlyb Version: f{__version__}")
        logger.info(f"Set Temperature to {T} K.")
        self.T = T
        logger.info(f"Set Software to {software}.")
        self.software = software
        self.unit = units
        self.file_list = []
        self.out = out
        if not Path(out).is_dir():
            logger.info(f"Make output folder {out}.")
            Path(out).mkdir(parents=True)

    def run(self, *args, **kwargs):
        """Run the workflow in an automatic fashion.

        This method would execute the
        :func:`~alchemlyb.workflows.WorkflowBase.read`,
        :func:`~alchemlyb.workflows.WorkflowBase.preprocess`,
        :func:`~alchemlyb.workflows.WorkflowBase.estimate`,
        :func:`~alchemlyb.workflows.WorkflowBase.check_convergence`,
        :func:`~alchemlyb.workflows.WorkflowBase.plot`
        sequentially such that the whole analysis could be done automatically.

        This method takes in an arbitrary number of arguments and pass all
        of them to the underlying methods. The methods will be selecting the
        keywords that they would like to use.

        Running this method would generate the resulting attributes for the
        user to retrieve the results.

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
        self.read(*args, **kwargs)
        self.preprocess(*args, **kwargs)
        self.estimate(*args, **kwargs)
        self.check_convergence(*args, **kwargs)
        self.plot(*args, **kwargs)

    def read(self, *args, **kwargs):
        """The function that reads the files in `file_list` and parse them
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
        """The function that subsample the u_nk and dHdl in `u_nk_list` and
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
        """The function that runs the estimator based on `u_nk_sample_list`
        and `dHdl_sample_list`.

        Attributes
        ----------

        result : pandas.Dataframe
            The main result of the workflow.

        """
        self.result = pd.DataFrame()

    def check_convergence(self, *args, **kwargs):
        """The function for doing convergence analysis.

        Attributes
        ----------

        convergence : pandas.Dataframe
            The result of the convergence analysis.

        """
        self.convergence = pd.DataFrame()

    def plot(self, *args, **kwargs):
        """The function for producing any plots."""
        pass
