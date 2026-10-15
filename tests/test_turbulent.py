"""Tests for atmoflux.turbulent."""
import numpy as np
import pytest

from atmoflux import turbulent as Tu
from atmoflux.exceptions import OutOfRangeError


def test_air_density_value():
    assert Tu.air_density(15, 101.325) == pytest.approx(1.225, abs=1e-3)


def test_air_density_bad_pressure():
    with pytest.raises(OutOfRangeError):
        Tu.air_density(15, -1.0)


def test_sensible_heat_flux_value():
    assert Tu.sensible_heat_flux(1.2, 3.0, 20.0, 25.0, 0.0013) == pytest.approx(
        23.517, abs=1e-3
    )


def test_sensible_heat_flux_sign():
    # Warmer surface than air -> upward (positive) sensible heat flux.
    assert Tu.sensible_heat_flux(1.2, 3.0, 20.0, 25.0, 0.0013) > 0
    assert Tu.sensible_heat_flux(1.2, 3.0, 25.0, 20.0, 0.0013) < 0


def test_latent_heat_flux_value():
    assert Tu.latent_heat_flux(1.2, 3.0, 0.008, 0.012, 0.0013) == pytest.approx(
        45.864, abs=1e-3
    )


def test_bulk_transfer_coefficient_value():
    assert Tu.bulk_transfer_coefficient(10.0, 0.03) == pytest.approx(
        0.004741, abs=1e-6
    )


def test_bulk_transfer_coefficient_bad_roughness():
    with pytest.raises(OutOfRangeError):
        Tu.bulk_transfer_coefficient(10.0, 0.0)


def test_air_density_array():
    out = Tu.air_density(np.array([15.0, 15.0]), 101.325)
    assert np.allclose(out, 1.225, atol=1e-3)


def test_surface_shear_stress_value():
    assert Tu.surface_shear_stress(1.225, 0.3) == pytest.approx(0.11025, abs=1e-5)


def test_surface_shear_stress_bad_density():
    with pytest.raises(OutOfRangeError):
        Tu.surface_shear_stress(0.0, 0.3)


def test_aerodynamic_resistance_value():
    assert Tu.aerodynamic_resistance(3.0, 10.0, 0.03) == pytest.approx(70.3, abs=0.05)


def test_aerodynamic_resistance_decreases_with_wind():
    # Stronger wind reduces aerodynamic resistance.
    assert Tu.aerodynamic_resistance(5.0, 10.0, 0.03) < Tu.aerodynamic_resistance(
        3.0, 10.0, 0.03
    )


def test_aerodynamic_resistance_bad_wind():
    with pytest.raises(OutOfRangeError):
        Tu.aerodynamic_resistance(0.0, 10.0, 0.03)


def test_aerodynamic_resistance_bad_height():
    with pytest.raises(OutOfRangeError):
        Tu.aerodynamic_resistance(3.0, 0.01, 0.03)


def test_moist_air_density_value():
    assert Tu.moist_air_density(15, 101.325, 0.01) == pytest.approx(1.2176, abs=1e-4)


def test_moist_air_density_dry_limit_matches_air_density():
    # With no water vapor, moist air density reduces to dry air density.
    assert Tu.moist_air_density(15, 101.325, 0.0) == pytest.approx(
        Tu.air_density(15, 101.325)
    )


def test_moist_air_lighter_than_dry_air():
    assert Tu.moist_air_density(25, 101.325, 0.015) < Tu.air_density(25, 101.325)


def test_moist_air_density_kelvin_input():
    assert Tu.moist_air_density(288.15, 101.325, 0.01, unit="K") == pytest.approx(
        Tu.moist_air_density(15, 101.325, 0.01)
    )


def test_moist_air_density_bad_pressure():
    with pytest.raises(OutOfRangeError):
        Tu.moist_air_density(15, 0.0, 0.01)


def test_moist_air_density_negative_mixing_ratio():
    with pytest.raises(OutOfRangeError):
        Tu.moist_air_density(15, 101.325, -0.001)


def test_moist_air_density_array():
    # Density falls as moisture rises
    out = Tu.moist_air_density(15, 101.325, np.array([0.0, 0.01, 0.02]))
    assert out.shape == (3,)
    assert np.all(np.diff(out) < 0) 
