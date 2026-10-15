"""Tests for atmoflux.hydro."""
import pytest

from atmoflux import hydro as Hy
from atmoflux.exceptions import OutOfRangeError, ValidationError


def test_latent_heat_to_evaporation_value():
    assert Hy.latent_heat_to_evaporation(100.0) == pytest.approx(3.527, abs=1e-3)


def test_latent_heat_to_evaporation_zero():
    assert Hy.latent_heat_to_evaporation(0.0) == pytest.approx(0.0)


def test_latent_heat_to_evaporation_bad_density():
    with pytest.raises(OutOfRangeError):
        Hy.latent_heat_to_evaporation(100.0, density_water=0.0)


def test_penman_evaporation_value():
    result = Hy.penman_evaporation(15.0, 0.0, 25.0, 2.0, 3.169, 1.9, 101.3)
    assert result == pytest.approx(6.326, abs=1e-3)


def test_penman_evaporation_bad_wind():
    with pytest.raises(OutOfRangeError):
        Hy.penman_evaporation(15.0, 0.0, 25.0, -2.0, 3.169, 1.9)


def test_penman_monteith_value():
    result = Hy.penman_monteith(150.0, 20.0, 25.0, 1.5, 1.2, 50.0, 70.0)
    assert result == pytest.approx(6.133, abs=1e-3)


def test_penman_monteith_bad_resistance():
    with pytest.raises(OutOfRangeError):
        Hy.penman_monteith(150.0, 20.0, 25.0, 1.5, 1.2, 0.0, 70.0)


def test_penman_monteith_negative_vpd():
    with pytest.raises(OutOfRangeError):
        Hy.penman_monteith(150.0, 20.0, 25.0, -0.1, 1.2, 50.0, 70.0)


def test_potential_evapotranspiration_value():
    result = Hy.potential_evapotranspiration(15.0, 0.0, 25.0, 2.0, 3.169, 1.9, 101.3)
    assert result == pytest.approx(5.539, abs=1e-3)


def test_open_water_exceeds_reference_et():
    # Open-water Penman evaporation should exceed grass reference ET0.
    pe = Hy.penman_evaporation(15.0, 0.0, 25.0, 2.0, 3.169, 1.9, 101.3)
    et0 = Hy.potential_evapotranspiration(15.0, 0.0, 25.0, 2.0, 3.169, 1.9, 101.3)
    assert pe > et0


def test_equilibrium_evaporation_value():
    assert Hy.equilibrium_evaporation(15.0, 0.0, 25.0) == pytest.approx(
        4.521, abs=1e-3
    )


def test_priestley_taylor_value():
    assert Hy.priestley_taylor(15.0, 0.0, 25.0) == pytest.approx(5.697, abs=1e-3)


def test_priestley_taylor_exceeds_equilibrium():
    # The Priestley-Taylor alpha (~1.26) scales equilibrium evaporation up.
    pt = Hy.priestley_taylor(15.0, 0.0, 25.0)
    eq = Hy.equilibrium_evaporation(15.0, 0.0, 25.0)
    assert pt > eq


def test_priestley_taylor_bad_alpha():
    with pytest.raises(OutOfRangeError):
        Hy.priestley_taylor(15.0, 0.0, 25.0, alpha=0.0)


def test_hargreaves_value():
    assert Hy.hargreaves(25.0, 18.0, 32.0, 36.0) == pytest.approx(5.41, abs=1e-2)


def test_hargreaves_bad_temperature_range():
    with pytest.raises(OutOfRangeError):
        Hy.hargreaves(25.0, 32.0, 18.0, 36.0)


PENMAN_FAMILY = [
    lambda m: Hy.penman_evaporation(15.0, 0.0, 25.0, 2.0, 3.169, 1.9, 101.3, method=m),
    lambda m: Hy.penman_monteith(150.0, 20.0, 25.0, 1.5, 1.2, 50.0, 70.0, method=m),
    lambda m: Hy.equilibrium_evaporation(15.0, 0.0, 25.0, method=m),
    lambda m: Hy.priestley_taylor(15.0, 0.0, 25.0, method=m),
]


@pytest.mark.parametrize("func", PENMAN_FAMILY)
def test_penman_family_alduchov_differs_slightly(func):
    # Alduchov only changes the slope term, so results shift by well under 1 %.
    tetens, alduchov = func("tetens"), func("alduchov")
    assert alduchov != tetens
    assert alduchov == pytest.approx(tetens, rel=0.01)


@pytest.mark.parametrize("func", PENMAN_FAMILY)
def test_penman_family_invalid_method(func):
    with pytest.raises(ValidationError):
        func("bogus")


def test_fao56_has_no_method_parameter():
    # FAO-56 specifies the Tetens slope; the standard form must not be swappable.
    with pytest.raises(TypeError):
        Hy.potential_evapotranspiration(
            15.0, 0.0, 25.0, 2.0, 3.169, 1.9, 101.3, method="alduchov"
        )


def test_dalton_evaporation_value():
    assert Hy.dalton_evaporation(1.2671, 3.0) == pytest.approx(3.801, abs=1e-3)


def test_dalton_evaporation_linear_in_vpd():
    assert Hy.dalton_evaporation(2.0, 3.0) == pytest.approx(2 * Hy.dalton_evaporation(1.0, 3.0))


def test_dalton_evaporation_negative_vpd_is_condensation():
    # Supersaturated air (negative VPD) gives negative evaporation, not an error.
    assert Hy.dalton_evaporation(-0.2, 3.0) < 0


def test_dalton_evaporation_zero_coefficient():
    assert Hy.dalton_evaporation(1.5, 0.0) == 0.0


def test_dalton_evaporation_bad_coefficient():
    with pytest.raises(OutOfRangeError):
        Hy.dalton_evaporation(1.5, -1.0)


def test_dalton_evaporation_array():
    import numpy as np

    out = Hy.dalton_evaporation(np.array([0.5, 1.0, -0.1]), 2.0)
    assert np.allclose(out, [1.0, 2.0, -0.2])
