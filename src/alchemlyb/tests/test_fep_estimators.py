"""Tests for all FEP-based estimators in ``alchemlyb``.

"""
import pytest

import alchemlyb
from alchemlyb.estimators import MBAR, BAR


class FEPestimatorMixin:
    """Mixin for all FEP Estimator test classes."""

    def compare_delta_f(self, X_delta_f):
        est = self.cls().fit(X_delta_f[0])
        delta_f, d_delta_f = self.get_delta_f(est)

        assert X_delta_f[1] == pytest.approx(delta_f, rel=1e-3)
        assert X_delta_f[2] == pytest.approx(d_delta_f, rel=1e-3)

    def get_delta_f(self, est):
        # Use .iloc[0, -1] as we want to cater for both
        # delta_f_.loc[0.0, 1.0] and delta_f_.loc[(0.0, 0.0), (0.0, 1.0)]
        return est.delta_f_.iloc[0, -1], est.d_delta_f_.iloc[0, -1]


class TestMBAR(FEPestimatorMixin):
    """Tests for MBAR."""

    cls = MBAR

    @pytest.fixture(
        params=[
            ("gmx_benzene_Coulomb_u_nk", 3.041, 0.02088),
            ("gmx_benzene_VDW_u_nk", -3.007, 0.04519),
            ("gmx_expanded_ensemble_case_1", 75.923, 0.14124),
            ("gmx_expanded_ensemble_case_2", 75.915, 0.14372),
            ("gmx_expanded_ensemble_case_3", 76.173, 0.11345),
            ("gmx_water_particle_with_total_energy", -11.680, 0.083655),
            ("gmx_water_particle_with_potential_energy", -11.675, 0.083589),
            ("gmx_water_particle_without_energy", -11.654, 0.083415),
            ("amber_bace_example_complex_vdw", 2.41149, 0.0620658),
            ("gomc_benzene_u_nk", -0.79994, 0.091579),
        ],
    )
    def X_delta_f(self, request):
        get_unk, E, dE = request.param
        return alchemlyb.concat(request.getfixturevalue(get_unk)), E, dE

    def test_mbar(self, X_delta_f):
        self.compare_delta_f(X_delta_f)


class TestBAR(FEPestimatorMixin):
    """Tests for BAR."""

    cls = BAR

    @pytest.fixture(
        params=[
            ("gmx_benzene_Coulomb_u_nk", 3.044, 0.01640),
            ("gmx_benzene_VDW_u_nk", -3.033, 0.03438),
            ("gmx_expanded_ensemble_case_1", 75.993, 0.11056),
            ("gmx_expanded_ensemble_case_2", 76.009, 0.11220),
            ("gmx_expanded_ensemble_case_3", 76.219, 0.08886),
            ("gmx_water_particle_with_total_energy", -11.675, 0.065055),
            ("gmx_water_particle_with_potential_energy", -11.724, 0.064964),
            ("gmx_water_particle_without_energy", -11.660, 0.064914),
            ("amber_bace_example_complex_vdw", 2.39294, 0.051192),
            ("namd_tyr2ala", 11.0044, 0.10235),
            ("namd_idws", 0.221147, 0.041003),
            ("namd_idws_restarted", 7.081127, 0.0344211),
            ("namd_idws_restarted_reversed", -4.18405, 0.03457),
            ("gomc_benzene_u_nk", -0.87095, 0.071263),
        ],
    )
    def X_delta_f(self, request):
        get_unk, E, dE = request.param
        return alchemlyb.concat(request.getfixturevalue(get_unk)), E, dE

    def test_bar(self, X_delta_f):
        self.compare_delta_f(X_delta_f)

    def get_delta_f(self, est):
        ee = 0.0

        for i in range(len(est.d_delta_f_) - 1):
            ee += est.d_delta_f_.values[i][i + 1] ** 2
        # Use .iloc[0, -1] as we want to cater for both
        # delta_f_.loc[0.0, 1.0] and delta_f_.loc[(0.0, 0.0), (0.0, 1.0)]
        return est.delta_f_.iloc[0, -1], ee**0.5


class Test_Units:
    """Test the units."""

    @staticmethod
    @pytest.fixture()
    def u_nk(gmx_benzene_Coulomb_u_nk):
        return alchemlyb.concat(gmx_benzene_Coulomb_u_nk)

    def test_bar(self, u_nk):
        bar = BAR().fit(u_nk)
        assert bar.delta_f_.attrs["temperature"] == 300
        assert bar.delta_f_.attrs["energy_unit"] == "kT"
        assert bar.d_delta_f_.attrs["temperature"] == 300
        assert bar.d_delta_f_.attrs["energy_unit"] == "kT"

    def test_mbar(self, u_nk):
        mbar = MBAR().fit(u_nk)
        assert mbar.delta_f_.attrs["temperature"] == 300
        assert mbar.delta_f_.attrs["energy_unit"] == "kT"
        assert mbar.d_delta_f_.attrs["temperature"] == 300
        assert mbar.d_delta_f_.attrs["energy_unit"] == "kT"


class TestEstimatorMixOut:
    """Ensure that the attribute d_delta_f_, delta_f_, states_ cannot be
    modified."""

    @pytest.mark.parametrize("estimator", [MBAR, BAR])
    def test_d_delta_f_(self, estimator):
        _estimator = estimator()
        with pytest.raises(AttributeError):
            _estimator.d_delta_f_ = 1

    @pytest.mark.parametrize("estimator", [MBAR, BAR])
    def test_delta_f_(self, estimator):
        _estimator = estimator()
        with pytest.raises(AttributeError):
            _estimator.delta_f_ = 1

    @pytest.mark.parametrize("estimator", [MBAR, BAR])
    def test_states_(self, estimator):
        _estimator = estimator()
        with pytest.raises(AttributeError):
            _estimator.states_ = 1


def test_bootstrap(gmx_benzene_Coulomb_u_nk):
    u_nk = alchemlyb.concat(gmx_benzene_Coulomb_u_nk)
    mbar = MBAR(n_bootstraps=2)
    mbar.fit(u_nk)
    mbar_bootstrap_mean = mbar.delta_f_.loc[0.00, 1.00]
    mbar_bootstrap_err = mbar.d_delta_f_.loc[0.00, 1.00]

    mbar = MBAR()
    mbar.fit(u_nk)
    mbar_mean = mbar.delta_f_.loc[0.00, 1.00]
    mbar_err = mbar.d_delta_f_.loc[0.00, 1.00]

    assert mbar_bootstrap_mean == mbar_mean
    assert mbar_bootstrap_err != mbar_err
