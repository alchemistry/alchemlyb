"""Tests for preprocessing functions.

"""

import logging

import numpy as np
import pytest
from numpy.testing import assert_allclose

import alchemlyb
from alchemlyb.preprocessing import (
    slicing,
    statistical_inefficiency,
    equilibrium_detection,
    decorrelate_u_nk,
    decorrelate_dhdl,
    u_nk2series,
    dhdl2series,
)


def _check_data_is_outside_bounds(data, lower, upper):
    """
    Helper function to make sure that `data` has entries that are
    below the `lower` bound, and above the `upper` bound.
    This is used by slicing tests to make sure that the data
    provided is appropriate for the tests.
    """
    assert any(data.reset_index()["time"] < lower)
    assert any(data.reset_index()["time"] > upper)


@pytest.fixture()
def dHdl(gmx_benzene_Coulomb_dHdl):
    return gmx_benzene_Coulomb_dHdl[0]


@pytest.fixture()
def u_nk(gmx_benzene_Coulomb_u_nk):
    return gmx_benzene_Coulomb_u_nk[0]


@pytest.fixture()
def multi_index_u_nk(gmx_ABFE_complex_u_nk):
    return gmx_ABFE_complex_u_nk[0]


@pytest.fixture()
def multi_index_dHdl(gmx_ABFE_complex_dHdl):
    return gmx_ABFE_complex_dHdl[0]


class TestSlicing:
    """Test slicing functionality."""

    def slicer(self, *args, **kwargs):
        return slicing(*args, **kwargs)

    @pytest.mark.parametrize(("data", "size"), [("dHdl", 661), ("u_nk", 661)])
    def test_basic_slicing(self, data, size, request):
        assert (
            len(
                self.slicer(
                    request.getfixturevalue(data), lower=1000, upper=34000, step=5
                )
            )
            == size
        )

    def test_unchanged(self, namd_idws):
        # NAMD energy files only have dE for adjacent lambdas, this ensures
        # that the slicer will not drop these rows as they have NaN values.
        # Do the pre-processing as the u_nk are from all lambdas
        groups = namd_idws.groupby("fep-lambda")
        for key, group in groups:
            group = group[~group.index.duplicated(keep="first")]
            df = self.slicer(group, None, None, None)
            assert len(df) == len(group)

    @pytest.mark.parametrize(
        ("dataloader", "lower", "upper"),
        [
            ("dHdl", 1000, 34000),
            ("u_nk", 1000, 34000),
        ],
    )
    def test_data_is_unchanged(self, dataloader, lower, upper, request):
        """
        Test that slicing does not change the underlying data
        """
        # Load data
        data = request.getfixturevalue(dataloader)
        # Check that the input data is appropriate for the test
        _check_data_is_outside_bounds(data, lower, upper)

        # Slice data, and test that we didn't change the input data
        original_length = len(data)
        sliced = self.slicer(data, lower=lower, upper=upper, step=5)
        assert len(data) == original_length

    @pytest.mark.parametrize(
        ("dataloader", "lower", "upper"),
        [
            ("dHdl", 1000, 34000),
            ("u_nk", 1000, 34000),
        ],
    )
    def test_lower_and_upper_bound(self, dataloader, lower, upper, request):
        """
        Test that the lower and upper time is respected
        """
        # Load data
        data = request.getfixturevalue(dataloader)
        # Check that the input data is appropriate for the test
        _check_data_is_outside_bounds(data, lower, upper)

        # Slice data, and test that we don't observe times outside
        # the prescribed range
        sliced = self.slicer(data, lower=lower, upper=upper, step=5)
        assert all(sliced.reset_index()["time"] >= lower)
        assert all(sliced.reset_index()["time"] <= upper)

    @pytest.mark.parametrize("dataloader", ["dHdl", "u_nk"])
    def test_disordered_exception(self, dataloader, request):
        """Test that a shuffled DataFrame yields a KeyError."""
        data = request.getfixturevalue(dataloader)
        indices = data.index.values
        np.random.shuffle(indices)

        df = data.loc[indices]

        with pytest.raises(KeyError):
            self.slicer(df, lower=200)

    @pytest.mark.parametrize(
        "dataloader", ["gmx_benzene_Coulomb_dHdl", "gmx_benzene_Coulomb_u_nk"]
    )
    def test_duplicated_exception(self, dataloader, request):
        """Test that a DataFrame with duplicate times yields a KeyError."""
        data = alchemlyb.concat(request.getfixturevalue(dataloader))
        with pytest.raises(KeyError):
            self.slicer(data.sort_index(axis=0), lower=200)

    def test_subsample_bounds_and_step(self, multi_index_u_nk):
        """Make sure that slicing the series also works"""
        subsample = statistical_inefficiency(
            multi_index_u_nk, multi_index_u_nk.sum(axis=1), lower=100, upper=400, step=2
        )
        assert len(subsample) == 76

    def test_multiindex_duplicated(self, multi_index_u_nk):
        subsample = statistical_inefficiency(
            multi_index_u_nk, multi_index_u_nk.sum(axis=1)
        )
        assert len(subsample) == 501

    def test_sort_off(self, multi_index_u_nk):
        unsorted = alchemlyb.concat([multi_index_u_nk[-500:], multi_index_u_nk[:500]])
        with pytest.raises(KeyError):
            statistical_inefficiency(unsorted, unsorted.sum(axis=1), sort=False)

    def test_sort_on(self, multi_index_u_nk):
        unsorted = alchemlyb.concat([multi_index_u_nk[-500:], multi_index_u_nk[:500]])
        subsample = statistical_inefficiency(unsorted, unsorted.sum(axis=1), sort=True)
        assert subsample.reset_index(0)["time"].is_monotonic_increasing

    def test_sort_on_noseries(self, multi_index_u_nk):
        unsorted = alchemlyb.concat([multi_index_u_nk[-500:], multi_index_u_nk[:500]])
        subsample = statistical_inefficiency(unsorted, None, sort=True)
        assert subsample.reset_index(0)["time"].is_monotonic_increasing

    def test_duplication_off(self, multi_index_u_nk):
        duplicated = alchemlyb.concat([multi_index_u_nk, multi_index_u_nk])
        with pytest.raises(KeyError):
            statistical_inefficiency(
                duplicated, duplicated.sum(axis=1), drop_duplicates=False
            )

    def test_duplication_on_dataframe(self, multi_index_u_nk):
        duplicated = alchemlyb.concat([multi_index_u_nk, multi_index_u_nk])
        subsample = statistical_inefficiency(
            duplicated, duplicated.sum(axis=1), drop_duplicates=True
        )
        assert len(subsample) < 1000

    def test_duplication_on_dataframe_noseries(self, multi_index_u_nk):
        duplicated = alchemlyb.concat([multi_index_u_nk, multi_index_u_nk])
        subsample = statistical_inefficiency(duplicated, None, drop_duplicates=True)
        assert len(subsample) == 1001

    def test_duplication_on_series(self, multi_index_u_nk):
        duplicated = alchemlyb.concat([multi_index_u_nk, multi_index_u_nk])
        subsample = statistical_inefficiency(
            duplicated.sum(axis=1), duplicated.sum(axis=1), drop_duplicates=True
        )
        assert len(subsample) < 1000

    def test_duplication_on_series_noseries(self, multi_index_u_nk):
        duplicated = alchemlyb.concat([multi_index_u_nk, multi_index_u_nk])
        subsample = statistical_inefficiency(
            duplicated.sum(axis=1), None, drop_duplicates=True
        )
        assert len(subsample) == 1001


class CorrelatedPreprocessors:
    @pytest.mark.parametrize(("dataloader", "size"), [("dHdl", 4001), ("u_nk", 4001)])
    def test_subsampling(self, dataloader, size, request):
        """Basic test for execution; resulting size of dataset sensitive to
        machine and depends on algorithm.
        """
        data = request.getfixturevalue(dataloader)
        assert len(self.slicer(data, series=data.loc[:, data.columns[0]])) <= size

    @pytest.mark.parametrize("dataloader", ["dHdl", "u_nk"])
    def test_no_series(self, dataloader, request):
        """Check that we get the same result as simple slicing with no Series."""
        data = request.getfixturevalue(dataloader)
        df_sub = self.slicer(data, lower=200, upper=5000, step=2)
        df_sliced = slicing(data, lower=200, upper=5000, step=2)

        assert np.all((df_sub == df_sliced))


class TestStatisticalInefficiency(TestSlicing, CorrelatedPreprocessors):
    def slicer(self, *args, **kwargs):
        return statistical_inefficiency(*args, **kwargs)

    @pytest.mark.parametrize(
        ("conservative", "dataloader", "size"),
        [
            (True, "dHdl", 2001),  # 0.00:  g = 1.0559445620585415
            (True, "u_nk", 2001),  # 'fep': g = 1.0560203916559594
            (False, "dHdl", 3789),
            (False, "u_nk", 3571),
        ],
    )
    def test_conservative(self, dataloader, size, conservative, request):
        data = request.getfixturevalue(dataloader)
        sliced = self.slicer(
            data, series=data.loc[:, data.columns[0]], conservative=conservative
        )
        # results can vary slightly with different machines
        # so possibly do
        # delta = 10
        # assert size - delta < len(sliced) < size + delta
        assert len(sliced) == size

    @pytest.mark.parametrize(
        "dataloader,end,step",
        [
            ("dHdl", 20, None),  # wrong length
            ("dHdl", None, -1),  # wrong time stamps (reversed)
        ],
    )
    def test_raise_ValueError_for_mismatched_data(self, dataloader, end, step, request):
        data = request.getfixturevalue(dataloader)
        with pytest.raises(ValueError):
            self.slicer(data, series=data["fep"][:end:step])

    @pytest.mark.parametrize(
        ("dataloader", "lower", "upper"),
        [
            ("dHdl", 1000, 34000),
            ("u_nk", 1000, 34000),
        ],
    )
    @pytest.mark.parametrize("use_series", [True, False])
    @pytest.mark.parametrize("conservative", [True, False])
    def test_data_is_unchanged(
        self, dataloader, use_series, lower, upper, conservative, request
    ):
        """
        Test that using statistical_inefficiency does not change the underlying data

        statistical_inefficiency is equivalent to slicing it its `series` parameter
        is not set. If the `series` parameter is set, additional inefficiency
        calculations are performed. We want to test both behaviors. The behavior
        is toggled using the `use_series` argument.
        """
        # Load data
        data = request.getfixturevalue(dataloader)
        # Check that the input data is appropriate for the test
        _check_data_is_outside_bounds(data, lower, upper)

        # Define subsampling series if required
        series = data.sum(axis=1) if use_series else None

        # Slice data, and test that we didn't change the input data
        original_length = len(data)
        self.slicer(
            data,
            series=series,
            lower=lower,
            upper=upper,
            step=5,
            conservative=conservative,
        )
        assert len(data) == original_length

    @pytest.mark.parametrize(
        ("dataloader", "lower", "upper"),
        [
            ("dHdl", 1000, 34000),
            ("u_nk", 1000, 34000),
        ],
    )
    @pytest.mark.parametrize("use_series", [True, False])
    @pytest.mark.parametrize("conservative", [True, False])
    def test_lower_and_upper_bound_slicer(
        self, dataloader, use_series, lower, upper, conservative, request
    ):
        """
        Test that the lower and upper time is respected when using statistical_inefficiency

        statistical_inefficiency is equivalent to slicing it its `series` parameter
        is not set. If the `series` parameter is set, additional inefficiency
        calculations are performed. We want to test both behaviors. The behavior
        is toggled using the `use_series` argument.
        """
        # Load data
        data = request.getfixturevalue(dataloader)
        # Check that the input data is appropriate for the test
        _check_data_is_outside_bounds(data, lower, upper)

        # Define subsampling series if required
        series = data.sum(axis=1) if use_series else None

        # Slice data, and test that we don't observe times outside
        # the prescribed range
        sliced = self.slicer(
            data,
            series=series,
            lower=lower,
            upper=upper,
            step=5,
            conservative=conservative,
        )
        assert all(sliced.reset_index()["time"] >= lower)
        assert all(sliced.reset_index()["time"] <= upper)

    @pytest.mark.parametrize(
        ("dataloader", "lower", "upper"),
        [
            ("dHdl", 1000, 34000),
            ("u_nk", 1000, 34000),
        ],
    )
    @pytest.mark.parametrize("conservative", [True, False])
    def test_slicing_inefficiency_equivalence(
        self, dataloader, lower, upper, conservative, request
    ):
        """
        Test that first slicing the data frame, then subsampling is equivalent to
        subsampling with lower / upper bounds set
        """
        # Load data
        data = request.getfixturevalue(dataloader)
        # Check that the input data is appropriate for the test
        _check_data_is_outside_bounds(data, lower, upper)

        # Slice dataframe, then subsample it based on the sum of its components
        sliced_data = slicing(data, lower=lower, upper=upper)
        subsampled_sliced_data = self.slicer(
            sliced_data, series=sliced_data.sum(axis=1), conservative=conservative
        )

        # Subsample the dataframe based on the sum of its components while
        # also specifying the slicing range
        subsampled_data = self.slicer(
            data,
            series=data.sum(axis=1),
            lower=lower,
            upper=upper,
            conservative=conservative,
        )

        assert (subsampled_sliced_data == subsampled_data).all(axis=None)


class TestEquilibriumDetection(TestSlicing, CorrelatedPreprocessors):
    def slicer(self, *args, **kwargs):
        return equilibrium_detection(*args, **kwargs)


class Test_Units:
    """Test the preprocessing module."""

    def test_slicing(self, u_nk):
        """Test if extract_u_nk assign the attr correctly"""
        new_u_nk = slicing(u_nk)
        assert new_u_nk.attrs["temperature"] == 300
        assert new_u_nk.attrs["energy_unit"] == "kT"

    def test_statistical_inefficiency(self, dHdl):
        """Test if extract_u_nk assign the attr correctly"""
        new_dhdl = statistical_inefficiency(dHdl)
        assert new_dhdl.attrs["temperature"] == 300
        assert new_dhdl.attrs["energy_unit"] == "kT"

    def test_equilibrium_detection(self, dHdl):
        """Test if extract_u_nk assign the attr correctly"""
        new_dhdl = equilibrium_detection(dHdl)
        assert new_dhdl.attrs["temperature"] == 300
        assert new_dhdl.attrs["energy_unit"] == "kT"


@pytest.mark.parametrize(("method", "size"), [("all", 2001), ("dE", 2001)])
def test_decorrelate_u_nk_single_l(u_nk, method, size):
    assert (
        len(decorrelate_u_nk(u_nk, method=method, drop_duplicates=True, sort=True))
        == size
    )


def test_decorrelate_u_nk_burnin(u_nk):
    assert (
        len(
            decorrelate_u_nk(
                u_nk,
                method="dE",
                drop_duplicates=True,
                sort=True,
                remove_burnin=True,
            )
        )
        == 2848
    )


def test_decorrelate_dhdl_burnin(dHdl):
    assert (
        len(
            decorrelate_dhdl(
                dHdl,
                drop_duplicates=True,
                sort=True,
                remove_burnin=True,
            )
        )
        == 2848
    )


@pytest.mark.parametrize(("method", "size"), [("all", 501), ("dE", 501)])
def test_decorrelate_u_nk_multiple_l(multi_index_u_nk, method, size):
    assert (
        len(
            decorrelate_u_nk(
                multi_index_u_nk,
                method=method,
            )
        )
        == size
    )


def test_decorrelate_dhdl_single_l(u_nk):
    assert len(decorrelate_dhdl(u_nk, drop_duplicates=True, sort=True)) == 2001


def test_decorrelate_dhdl_multiple_l(multi_index_dHdl):
    assert (
        len(
            decorrelate_dhdl(
                multi_index_dHdl,
            )
        )
        == 501
    )


def test_raise_nou_nk(multi_index_dHdl):
    with pytest.raises(ValueError):
        decorrelate_u_nk(
            multi_index_dHdl,
        )


class TestDhdl2series:
    @pytest.mark.parametrize("methodargs", [{}, {"method": "all"}])
    def test_dhdl2series(self, dHdl, methodargs):
        series = dhdl2series(dHdl, **methodargs)
        assert len(series) == len(dHdl)
        assert_allclose(series, dHdl.sum(axis=1))

    def test_other_method_ValueError(self, dHdl):
        with pytest.raises(
            ValueError, match="Only method='all' is supported for dhdl2series()."
        ):
            dhdl2series(dHdl, method="dE")


class TestU_nk2series:
    @pytest.mark.parametrize(
        "methodargs,reference",  # reference = sum
        [
            ({}, 7988.667045),
            ({"method": "all"}, 85982.34668751864),
            ({"method": "dE"}, 7988.667045),
        ],
    )
    def test_u_nk2series(self, u_nk, methodargs, reference):
        series = u_nk2series(u_nk, **methodargs)
        assert len(series) == len(u_nk)
        assert_allclose(series.sum(), reference)

    @pytest.mark.parametrize(
        "methodargs,reference",  # reference = sum
        [
            ({"method": "dhdl_all"}, 85982.34668751864),
            ({"method": "dhdl"}, 7988.667045),
        ],
    )
    def test_u_nk2series_deprecated(self, u_nk, methodargs, reference):
        with pytest.warns(
            DeprecationWarning,
            match=r"Method 'dhdl.*' has been deprecated, using '.*' instead\. "
            r"'dhdl.*' will be removed in alchemlyb 3\.0\.0\.",
        ):
            series = u_nk2series(u_nk, **methodargs)
        assert len(series) == len(u_nk)
        assert_allclose(series.sum(), reference)

    def test_other_method_ValueError(self, u_nk):
        with pytest.raises(ValueError, match="Decorrelation method bogus not found."):
            u_nk2series(u_nk, method="bogus")


class TestLogging:
    def test_detect_equilibration(self, caplog, u_nk):
        with caplog.at_level(logging.DEBUG):
            decorrelate_u_nk(u_nk, remove_burnin=True)

            assert "Running equilibration detection." in caplog.text
            assert "Start index:" in caplog.text
            assert "Statistical inefficiency:" in caplog.text
            assert "Number of uncorrelated samples:" in caplog.text

    def test_statistical_inefficiency(self, caplog, u_nk):
        with caplog.at_level(logging.DEBUG):
            decorrelate_u_nk(u_nk)

            assert "Running statistical inefficiency analysis." in caplog.text
            assert "Statistical inefficiency:" in caplog.text
            assert "Number of uncorrelated samples:" in caplog.text


def test_unequil_input(dHdl):
    with pytest.raises(ValueError, match="should be same as the length of series"):
        statistical_inefficiency(dHdl, series=dHdl[:10])


def test_series_none(dHdl):
    with pytest.warns(
        UserWarning,
        match="The series input is `None`, would not subsample according to statistical inefficiency.",
    ):
        statistical_inefficiency(dHdl, series=None)
