"""Tests for atmoflux.temperature."""
import numpy as np
import pytest

from atmoflux import temperature as T
from atmoflux.constants import DRY_ADIABATIC_LAPSE_RATE
from atmoflux.exceptions import InvalidUnitError, OutOfRangeError, ValidationError


@pytest.mark.parametrize(
    "value, src, dst, expected",
    [
        (100, "C", "F", 212.0),
        (273.15, "K", "C", 0.0),
        (32, "F", "C", 0.0),
        (0, "C", "K", 273.15),
        (25, "C", "C", 25),
    ],
)
def test_convert_temperature(value, src, dst, expected):
    assert T.convert_temperature(value, src, dst) == pytest.approx(expected)


def test_convert_temperature_array():
    out = T.convert_temperature(np.array([0.0, 100.0]), "C", "F")
    assert np.allclose(out, [32.0, 212.0])


def test_convert_temperature_bad_unit():
    with pytest.raises(InvalidUnitError):
        T.convert_temperature(20, "C", "Z")


def test_convert_temperature_non_numeric():
    with pytest.raises(ValidationError):
        T.convert_temperature("hot", "C", "F")


def test_dewpoint_temperature_value():
    assert T.dewpoint_temperature(30, 50) == pytest.approx(18.438, abs=1e-3)


def test_dewpoint_temperature_full_saturation():
    # At 100% RH the dew point equals the air temperature.
    assert T.dewpoint_temperature(20, 100) == pytest.approx(20.0, abs=1e-6)


def test_dewpoint_temperature_bad_rh():
    with pytest.raises(OutOfRangeError):
        T.dewpoint_temperature(20, 0)


def test_dewpoint_from_avp_roundtrip():
    assert T.dewpoint_from_avp(2.338) == pytest.approx(20.0, abs=1e-2)


def test_dewpoint_from_avp_negative():
    with pytest.raises(OutOfRangeError):
        T.dewpoint_from_avp(-1.0)


def test_potential_temperature_at_reference():
    # At the reference pressure, potential temperature equals temperature.
    assert T.potential_temperature(300.0, 101.325) == pytest.approx(300.0)


def test_potential_temperature_value():
    assert T.potential_temperature(273.15, 80.0) == pytest.approx(292.22, abs=1e-2)


def test_potential_temperature_bad_pressure():
    with pytest.raises(OutOfRangeError):
        T.potential_temperature(300.0, -10.0)


def test_virtual_temperature_exceeds_actual():
    assert T.virtual_temperature(300.0, 0.01) > 300.0


def test_virtual_temperature_dry_air():
    assert T.virtual_temperature(300.0, 0.0) == pytest.approx(300.0)


def test_virtual_temperature_negative_mixing():
    with pytest.raises(OutOfRangeError):
        T.virtual_temperature(300.0, -0.01)


def test_lapse_rate_value():
    assert T.lapse_rate(15.0, 8.5, 0.0, 1000.0) == pytest.approx(0.0065, abs=1e-5)


def test_lapse_rate_inversion_negative():
    # Temperature increasing with height -> negative lapse rate.
    assert T.lapse_rate(10.0, 12.0, 0.0, 500.0) < 0


def test_lapse_rate_equal_heights():
    with pytest.raises(OutOfRangeError):
        T.lapse_rate(15.0, 8.0, 100.0, 100.0)


def test_surface_temperature_from_lw_value():
    assert T.surface_temperature_from_lw(390.0) == pytest.approx(287.98, abs=1e-2)


def test_surface_temperature_from_lw_bad_flux():
    with pytest.raises(OutOfRangeError):
        T.surface_temperature_from_lw(-5.0)


def test_surface_temperature_from_lw_bad_emissivity():
    with pytest.raises(OutOfRangeError):
        T.surface_temperature_from_lw(390.0, emissivity=1.5)


def test_wet_bulb_temperature_value():
    assert T.wet_bulb_temperature(25, 50) == pytest.approx(18.0, abs=0.05)


def test_wet_bulb_below_dry_bulb():
    assert T.wet_bulb_temperature(30, 40) < 30


def test_wet_bulb_bad_rh():
    with pytest.raises(OutOfRangeError):
        T.wet_bulb_temperature(25, 0)


def test_equivalent_potential_temperature_value():
    assert T.equivalent_potential_temperature(290.0, 0.01, 90.0) == pytest.approx(
        326.29, abs=1e-2
    )


def test_equivalent_potential_exceeds_potential():
    theta = T.potential_temperature(290.0, 90.0, unit="K")
    theta_e = T.equivalent_potential_temperature(290.0, 0.01, 90.0)
    assert theta_e > theta


def test_equivalent_potential_bad_mixing():
    with pytest.raises(OutOfRangeError):
        T.equivalent_potential_temperature(290.0, -0.01, 90.0)


def test_moist_lapse_rate_value():
    assert T.moist_adiabatic_lapse_rate(20, 0.015) * 1000 == pytest.approx(
        4.302, abs=1e-3
    )


def test_moist_lapse_rate_below_dry():
    # Moist lapse rate is always smaller than the dry adiabatic rate.
    assert T.moist_adiabatic_lapse_rate(20, 0.015) < DRY_ADIABATIC_LAPSE_RATE


def test_dewpoint_temperature_alduchov_value():
    assert T.dewpoint_temperature(30, 50, method="alduchov") == pytest.approx(18.45, abs=0.01)


def test_dewpoint_temperature_default_is_tetens():
    assert T.dewpoint_temperature(30, 50) == T.dewpoint_temperature(30, 50, method="tetens")


def test_dewpoint_temperature_invalid_method():
    with pytest.raises(ValidationError):
        T.dewpoint_temperature(30, 50, method="magnus")


def test_dewpoint_from_avp_alduchov_value():
    assert T.dewpoint_from_avp(2.338, method="alduchov") == pytest.approx(20.03, abs=0.01)


def test_dewpoint_from_avp_invalid_method():
    with pytest.raises(ValidationError):
        T.dewpoint_from_avp(2.338, method=None)


@pytest.mark.parametrize("method", ["tetens", "alduchov"])
def test_dewpoint_from_avp_inverts_actual_vp(method):
    # dewpoint_from_avp must exactly invert actual_vp for the same method.
    from atmoflux import humidity as H

    e = H.actual_vp(12.5, method=method)
    assert T.dewpoint_from_avp(e, method=method) == pytest.approx(12.5, abs=1e-9)


@pytest.mark.parametrize("method", ["tetens", "alduchov"])
def test_dewpoint_temperature_consistent_with_relative_humidity(method):
    # Round trip: RH -> dew point -> RH with the same coefficients.
    from atmoflux import humidity as H

    td = T.dewpoint_temperature(25, 60, method=method)
    assert H.relative_humidity(25, td, method=method) == pytest.approx(60.0, abs=1e-9)
