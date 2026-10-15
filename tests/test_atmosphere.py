"""Tests for atmoflux.atmosphere."""
import numpy as np
import pytest

from atmoflux import atmosphere as Atm
from atmoflux.exceptions import OutOfRangeError


def test_scale_height_value():
    assert Atm.scale_height(15) == pytest.approx(8434.5, abs=0.1)


def test_pressure_at_altitude_value():
    assert Atm.pressure_at_altitude(1000) == pytest.approx(89.997, abs=1e-3)


def test_pressure_decreases_with_altitude():
    assert Atm.pressure_at_altitude(2000) < Atm.pressure_at_altitude(1000)


def test_pressure_at_altitude_bad_reference():
    with pytest.raises(OutOfRangeError):
        Atm.pressure_at_altitude(1000, pressure_ref=-1.0)


def test_hypsometric_thickness_value():
    assert Atm.hypsometric_thickness(101.325, 90.0, 15) == pytest.approx(999.7, abs=0.1)


def test_hypsometric_thickness_bad_order():
    with pytest.raises(OutOfRangeError):
        Atm.hypsometric_thickness(90.0, 101.325, 15)


def test_density_altitude_value():
    assert Atm.density_altitude(101.325, 25) == pytest.approx(353.9, abs=0.1)


def test_density_altitude_bad_pressure():
    with pytest.raises(OutOfRangeError):
        Atm.density_altitude(-1.0, 25)


def test_standard_atmosphere_value():
    temp, pres = Atm.standard_atmosphere(1000)
    assert temp == pytest.approx(8.5, abs=0.01)
    assert pres == pytest.approx(89.875, abs=1e-3)


def test_standard_atmosphere_sea_level():
    temp, pres = Atm.standard_atmosphere(0)
    assert temp == pytest.approx(15.0, abs=0.01)
    assert pres == pytest.approx(101.325, abs=1e-3)


def test_standard_atmosphere_out_of_range():
    with pytest.raises(OutOfRangeError):
        Atm.standard_atmosphere(12000)


def test_pressure_at_altitude_array():
    out = Atm.pressure_at_altitude(np.array([0.0, 1000.0]))
    assert out[0] == pytest.approx(101.325)
    assert out[1] == pytest.approx(89.997, abs=1e-3)
