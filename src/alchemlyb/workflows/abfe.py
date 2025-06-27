import os
import warnings
from os.path import join
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from loguru import logger

from .base import WorkflowBase
from .. import concat
from ..convergence import forward_backward_convergence
from ..estimators import MBAR, BAR, TI, FEP_ESTIMATORS, TI_ESTIMATORS
from ..parsing import gmx, amber, parquet
from ..postprocessors.units import get_unit_converter
from ..preprocessing.subsampling import decorrelate_dhdl, decorrelate_u_nk
from ..visualisation import (
    plot_mbar_overlap_matrix,
    plot_ti_dhdl,
    plot_dF_state,
    plot_convergence,
)


class ABFE(WorkflowBase):
    """Workflow for absolute and relative binding free energy calculations.

    This workflow provides functionality similar to the ``alchemical-analysis.py`` script.
    It loads multiple input files from alchemical free energy calculations and computes the
    free energies between different alchemical windows using different estimators. It
    produces plots to aid in the assessment of convergence.

    Parameters
    ----------
    T : float
        Temperature in K.
    units : str
        The unit used for printing and plotting results. {'kcal/mol', 'kJ/mol',
        'kT'}. Default: 'kT'.
    software : str
        The software used for generating input (case-insensitive). {'GROMACS', 'AMBER', 'PARQUET'}.
        This option chooses the appropriate parser for the input file.
    dir : str
        Directory in which data files are stored. Default: os.path.curdir.
    prefix : str
        Prefix for datafile sets. This argument accepts regular expressions and
        the input files are searched using
        ``Path(dir).glob("**/" + prefix + "*" + suffix)``. Default: 'dhdl'.
    suffix : str
        Suffix for datafile sets. Default: 'xvg'.
    outdirectory : str
        Directory in which the output files produced by this script will be
        stored. Default: os.path.curdir.

    Attributes
    ----------
    logger : Logger
        The logging object.
    file_list : list
        The list of filenames sorted by the lambda state.


    .. versionadded:: 1.0.0
    .. versionchanged:: 2.0.1
        The `dir` argument expects a real directory without wildcards and wildcards will no longer
        work as expected. Use `prefix` to specify wildcard-based patterns to search under `dir`.
    .. versionchanged:: 2.1.0
        The serialised dataframe could be read via software='PARQUET'.
    """

    def __init__(
        self,
        T,
        units="kT",
        software="GROMACS",
        dir=os.path.curdir,
        prefix="dhdl",
        suffix="xvg",
        outdirectory=os.path.curdir,
    ):
        super().__init__(units, software, T, outdirectory)
        logger.info("Initialise Alchemlyb ABFE Workflow")
        self.update_units(units)
        logger.info(
            f"Finding files with prefix: {prefix}, suffix: "
            f"{suffix} under directory {dir} produced by "
            f"{software}"
        )
        reg_exp = "**/" + prefix + "*" + suffix
        if "*" in dir:
            warnings.warn(
                f"A real directory is expected in `dir`={dir}, wildcard expressions should be supplied to `prefix`."
            )
        if not Path(dir).is_dir():
            raise ValueError(f"The input directory `dir`={dir} is not a directory.")
        self.file_list = list(map(str, Path(dir).glob(reg_exp)))

        if len(self.file_list) == 0:
            raise ValueError(f"No file has been matched to {reg_exp}.")

        logger.info(f"Found {len(self.file_list)} {suffix} files.")
        logger.info("Unsorted file list: \n {}", "\n".join(self.file_list))

        if software == "GROMACS":
            logger.info(f"Using {software} parser to read the data.")
            self._extract_u_nk = gmx.extract_u_nk
            self._extract_dHdl = gmx.extract_dHdl
        elif software == "AMBER":
            self._extract_u_nk = amber.extract_u_nk
            self._extract_dHdl = amber.extract_dHdl
        elif software == "PARQUET":
            self._extract_u_nk = parquet.extract_u_nk
            self._extract_dHdl = parquet.extract_dHdl
        else:
            raise NotImplementedError(f"{software} parser not found.")

    def read(self, read_u_nk=True, read_dHdl=True, n_jobs=1):
        """Read the u_nk and dHdL data from the
        :attr:`~alchemlyb.workflows.ABFE.file_list`

        Parameters
        ----------
        read_u_nk : bool
            Whether to read the u_nk.
        read_dHdl : bool
            Whether to read the dHdl.
        n_jobs : int
            Number of parallel workers to use for reading the data.
            (-1 means using all the threads)

        Attributes
        ----------
        u_nk_list : list
            A list of :class:`pandas.DataFrame` of u_nk.
        dHdl_list : list
            A list of :class:`pandas.DataFrame` of dHdl.
        """
        self.u_nk_sample_list = None
        self.dHdl_sample_list = None

        if read_u_nk:

            def extract_u_nk(_extract_u_nk, file, T):
                try:
                    u_nk = _extract_u_nk(file, T)
                    logger.info(f"Reading {len(u_nk)} lines of u_nk from {file}")
                    return u_nk
                except Exception as exc:
                    msg = f"Error reading u_nk from {file}."
                    logger.error(msg)
                    raise OSError(msg) from exc

            u_nk_list = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(extract_u_nk)(self._extract_u_nk, file, self.T)
                for file in self.file_list
            )
        else:
            u_nk_list = []

        if read_dHdl:

            def extract_dHdl(_extract_dHdl, file, T):
                try:
                    dhdl = _extract_dHdl(file, T)
                    logger.info(f"Reading {len(dhdl)} lines of dhdl from {file}")
                    return dhdl
                except Exception as exc:
                    msg = f"Error reading dHdl from {file}."
                    logger.error(msg)
                    raise OSError(msg) from exc

            dHdl_list = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(extract_dHdl)(self._extract_dHdl, file, self.T)
                for file in self.file_list
            )
        else:
            dHdl_list = []

        # Sort the files according to the state
        if read_u_nk:
            logger.info("Sort files according to the u_nk.")
            column_names = u_nk_list[0].columns.values.tolist()
            index_list = sorted(
                range(len(self.file_list)),
                key=lambda x: column_names.index(
                    u_nk_list[x].reset_index("time").index.values[0]
                ),
            )
        elif read_dHdl:
            logger.info("Sort files according to the dHdl.")
            index_list = sorted(
                range(len(self.file_list)),
                key=lambda x: dHdl_list[x].reset_index("time").index.values[0],
            )
        else:
            self.u_nk_list = []
            self.dHdl_list = []
            return

        self.file_list = [self.file_list[i] for i in index_list]
        logger.info("Sorted file list: \n{}", "\n".join(self.file_list))
        if read_u_nk:
            self.u_nk_list = [u_nk_list[i] for i in index_list]
        else:
            self.u_nk_list = []

        if read_dHdl:
            self.dHdl_list = [dHdl_list[i] for i in index_list]
        else:
            self.dHdl_list = []

    def run(
        self,
        skiptime=0,
        uncorr="dE",
        threshold=50,
        estimators=("MBAR", "BAR", "TI"),
        overlap="O_MBAR.pdf",
        breakdown=True,
        forwrev=None,
        n_jobs=1,
        *args,
        **kwargs,
    ):
        """The method for running the automatic analysis.

        Parameters
        ----------
        skiptime : float
            Discard data prior to this specified time as 'equilibration' data.
            Units are specified by the corresponding MD Engine. Default: 0.
        uncorr : str
            The observable to be used for the autocorrelation analysis; 'dE'.
        threshold : int
            Proceed with correlated samples if the number of uncorrelated samples is
            found to be less than this number. If 0 is given, the time series
            analysis will not be performed at all. Default: 50.
        estimators : str or list of str
            A list of the estimators to estimate the free energy with. Default:
            `('MBAR', 'BAR', 'TI')`.
        overlap : str
            The filename for the plot of overlap matrix. Default: 'O_MBAR.pdf'.
        breakdown : bool
            Plot the free energy differences evaluated for each pair of adjacent
            states for all methods, including the dH/dlambda curve for TI. Default:
            ``True``.
        forwrev : int
            Plot the free energy change as a function of time in both directions,
            with the specified number of points in the time plot. The number of time
            points (an integer) must be provided. Specify as ``None`` will not do
            the convergence analysis. Default: None. By default, 'MBAR'
            estimator will be used for convergence analysis, as it is
            usually the fastest converging method. If the dataset does not
            contain u_nk, please run
            meth:`~alchemlyb.workflows.ABFE.check_convergence` manually
            with estimator='TI'.
        n_jobs : int
            Number of parallel workers to use for reading and decorrelating the data.
            (-1 means using all the threads)

        Attributes
        ----------
        summary : Dataframe
            The summary of the free energy estimate.
        convergence : DataFrame
            The summary of the convergence results. See
            :func:`~alchemlyb.convergence.forward_backward_convergence` for
            further explanation.
        """
        use_FEP = False
        use_TI = False

        if estimators is not None:
            if isinstance(estimators, str):
                estimators = [
                    estimators,
                ]
            for estimator in estimators:
                if estimator in FEP_ESTIMATORS:
                    use_FEP = True
                elif estimator in TI_ESTIMATORS:
                    use_TI = True
                else:
                    msg = (
                        f"Estimator {estimator} is not supported. Choose one from "
                        f"{FEP_ESTIMATORS + TI_ESTIMATORS}."
                    )
                    logger.error(msg)
                    raise ValueError(msg)
            self.read(read_u_nk=use_FEP, read_dHdl=use_TI, n_jobs=n_jobs)

        if uncorr is not None:
            self.preprocess(
                skiptime=skiptime, uncorr=uncorr, threshold=threshold, n_jobs=n_jobs
            )
        if estimators is not None:
            self.estimate(estimators)
            self.generate_result()

        if overlap is not None and use_FEP:
            ax = self.plot_overlap_matrix(overlap)
            plt.close(ax.figure)

        if breakdown:
            if use_TI:
                ax = self.plot_ti_dhdl()
                plt.close(ax.figure)
            fig = self.plot_dF_state()
            plt.close(fig)
            fig = self.plot_dF_state(
                dF_state="dF_state_long.pdf", orientation="landscape"
            )
            plt.close(fig)

        if forwrev:
            ax = self.check_convergence(forwrev, estimator="MBAR", dF_t="dF_t.pdf")
            plt.close(ax.figure)

    def update_units(self, units=None):
        """Update the unit.

        Parameters
        ----------
        units : {'kcal/mol', 'kJ/mol', 'kT'}
            The unit used for printing and plotting results.

        """
        if units is not None:
            logger.info(f"Set unit to {units}.")
            self.units = units or None

    def preprocess(self, skiptime=0, uncorr="dE", threshold=50, n_jobs=1):
        """Preprocess the data by removing the equilibration time and
        decorrelate the date.

        Parameters
        ----------
        skiptime : float
            Discard data prior to this specified time as 'equilibration' data.
            Units are specified by the corresponding MD Engine. Default: 0.
        uncorr : str
            The observable to be used for the autocorrelation analysis; 'dE'.
        threshold : int
            Proceed with correlated samples if the number of uncorrelated
            samples is found to be less than this number. If 0 is given, the
            time series analysis will not be performed at all. Default: 50.
        n_jobs : int
            Number of parallel workers to use for decorrelating the data.
            (-1 means using all the threads)

        Attributes
        ----------
        u_nk_sample_list : list
            The list of u_nk after decorrelation.
        dHdl_sample_list : list
            The list of dHdl after decorrelation.
        """
        logger.info(
            f"Start preprocessing with skiptime of {skiptime} "
            f"uncorrelation method of {uncorr} and threshold of "
            f"{threshold}"
        )
        if len(self.u_nk_list) > 0:
            logger.info(f"Processing the u_nk data set with skiptime of {skiptime}.")

            def _decorrelate_u_nk(u_nk, skiptime, threshold, index):
                u_nk = u_nk[u_nk.index.get_level_values("time") >= skiptime]
                subsample = decorrelate_u_nk(u_nk, uncorr, remove_burnin=True)
                if len(subsample) < threshold:
                    logger.warning(
                        f"Number of u_nk {len(subsample)} "
                        f"for state {index} is less than the "
                        f"threshold {threshold}."
                    )
                    logger.info(f"Take all the u_nk for state {index}.")
                    subsample = u_nk
                else:
                    logger.info(
                        f"Take {len(subsample)} uncorrelated u_nk for state {index}."
                    )
                return subsample

            self.u_nk_sample_list = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(_decorrelate_u_nk)(u_nk, skiptime, threshold, index)
                for index, u_nk in enumerate(self.u_nk_list)
            )
        else:
            logger.info("No u_nk data being subsampled")

        if len(self.dHdl_list) > 0:

            def _decorrelate_dhdl(dHdl, skiptime, threshold, index):
                dHdl = dHdl[dHdl.index.get_level_values("time") >= skiptime]
                subsample = decorrelate_dhdl(dHdl, remove_burnin=True)
                if len(subsample) < threshold:
                    logger.warning(
                        f"Number of dHdl {len(subsample)} for "
                        f"state {index} is less than the "
                        f"threshold {threshold}."
                    )
                    logger.info(f"Take all the dHdl for state {index}.")
                    subsample = dHdl
                else:
                    logger.info(
                        f"Take {len(subsample)} uncorrelated dHdl for state {index}."
                    )
                return subsample

            self.dHdl_sample_list = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(_decorrelate_dhdl)(dHdl, skiptime, threshold, index)
                for index, dHdl in enumerate(self.dHdl_list)
            )
        else:
            logger.info("No dHdl data being subsampled")

    def estimate(self, estimators=("MBAR", "BAR", "TI"), **kwargs):
        """Estimate the free energy using the selected estimator.

        Parameters
        ----------
        estimators : str or list of str
            A list of the estimators to estimate the free energy with. Default:
            ['TI', 'BAR', 'MBAR'].

        kwargs : dict
            Keyword arguments to be passed to the estimator.

        Attributes
        ----------
        estimator : dict
            The dictionary of estimators. The keys are in ['TI', 'BAR',
            'MBAR']. Note that the estimators are in their original form where
            no unit conversion has been attempted.


        .. versionchanged:: 2.1.0
            DeprecationWarning for using analytic error for MBAR estimator.

        """
        # Make estimators into a tuple
        if isinstance(estimators, str):
            estimators = (estimators,)

        for estimator in estimators:
            if estimator not in (FEP_ESTIMATORS + TI_ESTIMATORS):
                msg = f"Estimator {estimator} is not available in {FEP_ESTIMATORS + TI_ESTIMATORS}."
                logger.error(msg)
                raise ValueError(msg)

        logger.info(f"Start running estimator: {','.join(estimators)}.")
        self.estimator = {}
        # Use unprocessed data if preprocess is not performed.
        if "TI" in estimators:
            if self.dHdl_sample_list is not None:
                dHdl = concat(self.dHdl_sample_list)
            else:
                dHdl = concat(self.dHdl_list)
                logger.warning("dHdl has not been preprocessed.")
            logger.info(f"A total {len(dHdl)} lines of dHdl is used.")

        if "BAR" in estimators or "MBAR" in estimators:
            if self.u_nk_sample_list is not None:
                u_nk = concat(self.u_nk_sample_list)
            else:
                u_nk = concat(self.u_nk_list)
                logger.warning("u_nk has not been preprocessed.")
            logger.info(f"A total {len(u_nk)} lines of u_nk is used.")

        for estimator in estimators:
            if estimator == "MBAR":
                logger.info("Run MBAR estimator.")
                warnings.warn(
                    "From 2.2.0, n_bootstraps=50 will be the default for estimating MBAR error.",
                    DeprecationWarning,
                )
                self.estimator[estimator] = MBAR(**kwargs).fit(u_nk)
            elif estimator == "BAR":
                logger.info("Run BAR estimator.")
                self.estimator[estimator] = BAR(**kwargs).fit(u_nk)
            elif estimator == "TI":
                logger.info("Run TI estimator.")
                self.estimator[estimator] = TI(**kwargs).fit(dHdl)

    def generate_result(self):
        """Summarise the result into a dataframe.

        Returns
        -------
        DataFrame
            The DataFrame with convergence data. ::

                                      MBAR  MBAR_Error        BAR  BAR_Error         TI  TI_Error
                States 0 -- 1     0.065967    0.001293   0.066544   0.001661   0.066663  0.001675
                       1 -- 2     0.089774    0.001398   0.089303   0.002101   0.089566  0.002144
                       2 -- 3     0.132036    0.001638   0.132687   0.002990   0.133292  0.003055
                       3 -- 4     0.116494    0.001213   0.116348   0.002691   0.116845  0.002750
                       4 -- 5     0.105251    0.000980   0.106344   0.002337   0.106603  0.002362
                       5 -- 6     0.349320    0.002781   0.343399   0.006839   0.350568  0.007393
                       6 -- 7     0.402346    0.002767   0.391368   0.006641   0.395754  0.006961
                       7 -- 8     0.322284    0.002058   0.319395   0.005333   0.321542  0.005434
                       8 -- 9     0.434999    0.002683   0.425680   0.006823   0.430251  0.007155
                       9 -- 10    0.355672    0.002219   0.350564   0.005472   0.352745  0.005591
                       10 -- 11   3.574227    0.008744   3.513595   0.018711   3.514790  0.018078
                       11 -- 12   2.896685    0.009905   2.821760   0.017844   2.823210  0.018088
                       12 -- 13   2.223769    0.011229   2.188885   0.018438   2.189784  0.018478
                       13 -- 14   1.520978    0.012526   1.493598   0.019155   1.490070  0.019288
                       14 -- 15   0.911279    0.009527   0.894878   0.015023   0.896010  0.015140
                       15 -- 16   0.892365    0.010558   0.886706   0.015260   0.884698  0.015392
                       16 -- 17   1.737971    0.025315   1.720643   0.031416   1.741028  0.030624
                       17 -- 18   1.790706    0.025560   1.788112   0.029435   1.801695  0.029244
                       18 -- 19   1.998635    0.023340   2.007404   0.027447   2.019213  0.027096
                       19 -- 20   2.263475    0.020286   2.265322   0.025023   2.282040  0.024566
                       20 -- 21   2.565680    0.016695   2.561324   0.023611   2.552977  0.023753
                       21 -- 22   1.384094    0.007553   1.385837   0.011672   1.381999  0.011991
                       22 -- 23   1.428567    0.007504   1.422689   0.012524   1.416010  0.013012
                       23 -- 24   1.440581    0.008059   1.412517   0.013125   1.408267  0.013539
                       24 -- 25   1.411329    0.009022   1.419167   0.013356   1.411446  0.013795
                       25 -- 26   1.340320    0.010167   1.360679   0.015213   1.356953  0.015260
                       26 -- 27   1.243745    0.011239   1.245873   0.015711   1.248959  0.015762
                       27 -- 28   1.128429    0.012859   1.124554   0.016999   1.121892  0.016962
                       28 -- 29   1.010313    0.016442   1.005444   0.017692   1.019747  0.017257
                Stages coul      10.215658    0.033903  10.017838   0.041839  10.017854  0.048744
                       vdw       22.547489    0.098699  22.501150   0.060092  22.542936  0.106723
                       bonded     2.374144    0.014995   2.341631   0.005507   2.363828  0.021078
                       TOTAL     35.137291    0.103580  34.860619   0.087022  34.924618  0.119206

        Attributes
        ----------
        summary : Dataframe
            The summary of the free energy estimate.
        """

        # Write estimate
        logger.info("Summarise the estimate into a dataframe.")
        # Make the header name
        logger.info("Generate the row names.")
        estimator_names = list(self.estimator.keys())
        num_states = len(self.estimator[estimator_names[0]].states_)
        data_dict = {"name": [], "state": []}
        for i in range(num_states - 1):
            data_dict["name"].append(str(i) + " -- " + str(i + 1))
            data_dict["state"].append("States")

        try:
            u_nk = self.u_nk_list[0]
            stages = u_nk.reset_index("time").index.names
            logger.info("use the stage name from u_nk")
        except Exception:
            dHdl = self.dHdl_list[0]
            stages = dHdl.reset_index("time").index.names
            logger.info("use the stage name from dHdl")

        for stage in stages:
            data_dict["name"].append(stage.split("-")[0])
            data_dict["state"].append("Stages")
        data_dict["name"].append("TOTAL")
        data_dict["state"].append("Stages")

        col_names = []
        for estimator_name, estimator in self.estimator.items():
            logger.info(f"Read the results from estimator {estimator_name}")

            # Do the unit conversion
            delta_f_ = estimator.delta_f_
            d_delta_f_ = estimator.d_delta_f_
            # Write the estimator header

            col_names.append(estimator_name)
            col_names.append(estimator_name + "_Error")
            data_dict[estimator_name] = []
            data_dict[estimator_name + "_Error"] = []
            for index in range(1, num_states):
                data_dict[estimator_name].append(delta_f_.iloc[index - 1, index])
                data_dict[estimator_name + "_Error"].append(
                    d_delta_f_.iloc[index - 1, index]
                )

            logger.info(f"Generate the staged result from estimator {estimator_name}")
            for index, stage in enumerate(stages):
                if len(stages) == 1:
                    start = 0
                    end = len(estimator.states_) - 1
                else:
                    # Get the start and the end of the state
                    lambda_min = min([state[index] for state in estimator.states_])
                    lambda_max = max([state[index] for state in estimator.states_])
                    if lambda_min == lambda_max:
                        # Deal with the case where a certain lambda is used but
                        # not perturbed
                        start = 0
                        end = 0
                    else:
                        states = [state[index] for state in estimator.states_]
                        start = list(reversed(states)).index(lambda_min)
                        start = num_states - start - 1
                        end = states.index(lambda_max)
                logger.info(f"Stage {stage} is from state {start} to state {end}.")
                # This assumes that the indexes are sorted as the
                # preprocessing should sort the index of the df.
                result = delta_f_.iloc[start, end]
                if estimator_name != "BAR":
                    error = d_delta_f_.iloc[start, end]
                else:
                    error = np.sqrt(
                        sum(
                            [
                                d_delta_f_.iloc[start, start + 1] ** 2
                                for i in range(start, end + 1)
                            ]
                        )
                    )
                data_dict[estimator_name].append(result)
                data_dict[estimator_name + "_Error"].append(error)

            # Total result
            # This assumes that the indexes are sorted as the
            # preprocessing should sort the index of the df.
            result = delta_f_.iloc[0, -1]
            if estimator_name != "BAR":
                error = d_delta_f_.iloc[0, -1]
            else:
                error = np.sqrt(
                    sum([d_delta_f_.iloc[i, i + 1] ** 2 for i in range(num_states - 1)])
                )
            data_dict[estimator_name].append(result)
            data_dict[estimator_name + "_Error"].append(error)
        summary = pd.DataFrame.from_dict(data_dict)

        summary = summary.set_index(["state", "name"])
        # Make sure that the columns are in the right order
        summary = summary[col_names]
        # Remove the name of the index column to make it prettier
        summary.index.names = [None, None]

        summary.attrs = estimator.delta_f_.attrs
        converter = get_unit_converter(self.units)
        summary = converter(summary)
        self.summary = summary
        logger.info(f"Write results:\n{summary.to_string()}")
        return summary

    def plot_overlap_matrix(self, overlap="O_MBAR.pdf", ax=None):
        """Plot the overlap matrix for MBAR estimator using
        :func:`~alchemlyb.visualisation.plot_mbar_overlap_matrix`.

        Parameters
        ----------
        overlap : str
            The filename for the plot of overlap matrix. Default: 'O_MBAR.pdf'
        ax : matplotlib.axes.Axes
            Matplotlib axes object where the plot will be drawn on. If
            ``ax=None``, a new axes will be generated.

        Returns
        -------
        matplotlib.axes.Axes
            An axes with the overlap matrix drawn.
        """
        logger.info("Plot overlap matrix.")
        if "MBAR" in self.estimator:
            ax = plot_mbar_overlap_matrix(self.estimator["MBAR"].overlap_matrix, ax=ax)
            ax.figure.savefig(join(self.out, overlap))
            logger.info(f"Plot overlap matrix to {self.out} under {overlap}.")
            return ax
        else:
            logger.warning("MBAR estimator not found. Overlap matrix not plotted.")

    def plot_ti_dhdl(self, dhdl_TI="dhdl_TI.pdf", labels=None, colors=None, ax=None):
        """Plot the dHdl for TI estimator using
        :func:`~alchemlyb.visualisation.plot_ti_dhdl`.

        Parameters
        ----------
        dhdl_TI : str
            The filename for the plot of TI dHdl. Default: 'dhdl_TI.pdf'
        labels : List
            list of labels for labelling all the alchemical transformations.
        colors : List
            list of colors for plotting all the alchemical transformations.
            Default: ['r', 'g', '#7F38EC', '#9F000F', 'b', 'y']
        ax : matplotlib.axes.Axes
            Matplotlib axes object where the plot will be drawn on. If ``ax=None``,
            a new axes will be generated.

        Returns
        -------
        matplotlib.axes.Axes
            An axes with the TI dhdl drawn.
        """
        logger.info("Plot TI dHdl.")
        if "TI" in self.estimator:
            ax = plot_ti_dhdl(
                self.estimator["TI"],
                units=self.units,
                labels=labels,
                colors=colors,
                ax=ax,
            )
            ax.figure.savefig(join(self.out, dhdl_TI))
            logger.info(f"Plot TI dHdl to {dhdl_TI} under {self.out}.")
            return ax
        else:
            raise ValueError("No TI data available in estimators.")

    def plot_dF_state(
        self,
        dF_state="dF_state.pdf",
        labels=None,
        colors=None,
        orientation="portrait",
        nb=10,
    ):
        """Plot the dF states using
        :func:`~alchemlyb.visualisation.plot_dF_state`.

        Parameters
        ----------
        dF_state : str
            The filename for the plot of dF states. Default: 'dF_state.pdf'
        labels : List
            list of labels for labelling different estimators.
        colors : List
            list of colors for plotting different estimators.
        orientation : string
            The orientation of the figure. Can be `portrait` or `landscape`
        nb : int
            Maximum number of dF states in one row in the `portrait` mode

        Returns
        -------
        matplotlib.figure.Figure
            An Figure with the dF states drawn.
        """
        logger.info("Plot dF states.")
        fig = plot_dF_state(
            self.estimator.values(),
            labels=labels,
            colors=colors,
            units=self.units,
            orientation=orientation,
            nb=nb,
        )
        fig.savefig(join(self.out, dF_state))
        logger.info(f"Plot dF state to {dF_state} under {self.out}.")
        return fig

    def check_convergence(
        self, forwrev, estimator="MBAR", dF_t="dF_t.pdf", ax=None, **kwargs
    ):
        """Compute the forward and backward convergence using
        :func:`~alchemlyb.convergence.forward_backward_convergence`and
        plot with
        :func:`~alchemlyb.visualisation.plot_convergence`.

        Parameters
        ----------
        forwrev : int
            Plot the free energy change as a function of time in both
            directions, with the specified number of points in the time plot.
            The number of time points (an integer) must be provided.
        estimator : {'TI', 'BAR', 'MBAR'}
            The estimator used for convergence analysis. Default: 'MBAR'
        dF_t : str
            The filename for the plot of convergence. Default: 'dF_t.pdf'
        ax : matplotlib.axes.Axes
            Matplotlib axes object where the plot will be drawn on. If ``ax=None``,
            a new axes will be generated.
        kwargs : dict
            Keyword arguments to be passed to the estimator.

        Attributes
        ----------
        convergence : DataFrame

        Returns
        -------
        matplotlib.axes.Axes
            An axes with the convergence drawn.

        """
        logger.info("Start convergence analysis.")
        logger.info("Checking data availability.")

        if estimator in FEP_ESTIMATORS:
            if self.u_nk_sample_list is not None:
                u_nk_list = self.u_nk_sample_list
                logger.info("Subsampled u_nk is available.")
            else:
                if self.u_nk_list is not None:
                    u_nk_list = self.u_nk_list
                    logger.info(
                        "Subsampled u_nk not available, use original data instead."
                    )
                else:
                    msg = (
                        f"u_nk is needed for the f{estimator} estimator. "
                        f"If the dataset only has dHdl, "
                        f"run ABFE.check_convergence(estimator='TI') to "
                        f"use a TI estimator."
                    )
                    logger.error(msg)
                    raise ValueError(msg)
            convergence = forward_backward_convergence(
                u_nk_list, estimator=estimator, num=forwrev, **kwargs
            )
        elif estimator in TI_ESTIMATORS:
            logger.warning("No valid FEP estimator or dataset found. Fallback to TI.")
            if self.dHdl_sample_list is not None:
                dHdl_list = self.dHdl_sample_list
                logger.info("Subsampled dHdl is available.")
            else:
                if self.dHdl_list is not None:
                    dHdl_list = self.dHdl_list
                    logger.info(
                        "Subsampled dHdl not available, use original data instead."
                    )
                else:
                    logger.error(f"dHdl is needed for the f{estimator} estimator.")
                    raise ValueError(f"dHdl is needed for the f{estimator} estimator.")
            convergence = forward_backward_convergence(
                dHdl_list, estimator=estimator, num=forwrev, **kwargs
            )
        else:
            msg = (
                f"Estimator {estimator} is not supported. Choose one from "
                f"{FEP_ESTIMATORS + TI_ESTIMATORS}."
            )
            logger.error(msg)
            raise ValueError(msg)

        unit_converted_convergence = get_unit_converter(self.units)(convergence)
        # Otherwise the data_fraction column is converted as well.
        unit_converted_convergence["data_fraction"] = convergence["data_fraction"]
        self.convergence = unit_converted_convergence

        logger.info(f"Plot convergence analysis to {dF_t} under {self.out}.")

        ax = plot_convergence(self.convergence, units=self.units, ax=ax)
        ax.figure.savefig(join(self.out, dF_t))
        return ax
