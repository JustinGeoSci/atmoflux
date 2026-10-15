"""Tests for atmoflux.humidity."""
import numpy as np
import pytest

from atmoflux import humidity as H
from atmoflux.exceptions import OutOfRangeError, ValidationError


def test_saturation_vp_value():
    assert H.saturation_vp(20) == pytest.approx(2.3382, abs=1e-4)


def test_saturation_vp_array():
    out = H.saturation_vp(np.array([0.0, 20.0]))
    assert np.allclose(out, [0.61078, 2.3382], atol=1e-4)


def test_actual_vp_value():
    assert H.actual_vp(10) == pytest.approx(1.2279, abs=1e-4)


def test_saturation_vp_slope_value():
    assert H.saturation_vp_slope(20) == pytest.approx(0.1447, abs=1e-4)


def test_relative_humidity_value():
    assert H.relative_humidity(20, 10) == pytest.approx(52.5, abs=0.1)


def test_relative_humidity_saturated():
    # Dew point equal to temperature implies 100% RH.
    assert H.relative_humidity(15, 15) == pytest.approx(100.0)


def test_specific_humidity_value():
    assert H.specific_humidity(1.2279, 101.325) == pytest.approx(0.00757, abs=1e-5)


def test_specific_humidity_bad_pressure():
    with pytest.raises(OutOfRangeError):
        H.specific_humidity(2.0, 1.0)


def test_mixing_ratio_value():
    assert H.mixing_ratio(1.2279, 101.325) == pytest.approx(0.00763, abs=1e-5)


def test_mixing_ratio_exceeds_specific_humidity():
    e, p = 1.5, 101.325
    assert H.mixing_ratio(e, p) > H.specific_humidity(e, p)


def test_vapor_pressure_deficit_value():
    assert H.vapor_pressure_deficit(25, 60) == pytest.approx(1.2671, abs=1e-4)


def test_vapor_pressure_deficit_bad_rh():
    with pytest.raises(OutOfRangeError):
        H.vapor_pressure_deficit(25, 150)


def test_absolute_humidity_value():
    assert H.absolute_humidity(1.2279, 20) == pytest.approx(0.00908, abs=1e-5)


def test_saturation_vp_ice_value():
    assert H.saturation_vp_ice(-10) == pytest.approx(0.2595, abs=1e-4)


def test_saturation_vp_ice_below_water():
    # Over ice, saturation vapor pressure is below that over supercooled water.
    assert H.saturation_vp_ice(-10) < H.saturation_vp(-10)


def test_specific_humidity_from_dewpoint_value():
    assert H.specific_humidity_from_dewpoint(10, 101.325) == pytest.approx(
        0.00757, abs=1e-5
    )


def test_specific_humidity_from_dewpoint_bad_pressure():
    with pytest.raises(OutOfRangeError):
        H.specific_humidity_from_dewpoint(10, 0.5)


def test_rh_from_specific_humidity_value():
    assert H.relative_humidity_from_specific_humidity(
        0.00757, 20, 101.325
    ) == pytest.approx(52.5, abs=0.1)


def test_rh_from_specific_humidity_roundtrip():
    # q -> RH should invert RH -> q (via dewpoint) to within tolerance.
    q = H.specific_humidity_from_dewpoint(10, 101.325)
    rh = H.relative_humidity_from_specific_humidity(q, 20, 101.325)
    assert rh == pytest.approx(H.relative_humidity(20, 10), abs=0.5)


def test_rh_from_specific_humidity_bad_pressure():
    with pytest.raises(OutOfRangeError):
        H.relative_humidity_from_specific_humidity(0.005, 20, -1.0)


def test_precipitable_water_value():
    assert H.precipitable_water(0.01, 101.325) == pytest.approx(103.32, abs=1e-2)


def test_precipitable_water_bad_humidity():
    with pytest.raises(OutOfRangeError):
        H.precipitable_water(-0.001, 101.325)


SVP_FUNCS_T = [
    lambda m: H.saturation_vp(20, method=m),
    lambda m: H.actual_vp(10, method=m),
    lambda m: H.saturation_vp_slope(20, method=m),
    lambda m: H.saturation_vp_ice(-10, method=m),
    lambda m: H.relative_humidity(20, 10, method=m),
    lambda m: H.vapor_pressure_deficit(25, 60, method=m),
    lambda m: H.specific_humidity_from_dewpoint(10, 101.325, method=m),
    lambda m: H.relative_humidity_from_specific_humidity(0.00757, 20, 101.325, method=m),
    lambda m: H.saturation_specific_humidity(20, 101.325, method=m),
]


@pytest.mark.parametrize(
    "implicit, explicit",
    [
        (lambda: H.saturation_vp(20), lambda: H.saturation_vp(20, method="tetens")),
        (lambda: H.actual_vp(10), lambda: H.actual_vp(10, method="tetens")),
        (lambda: H.saturation_vp_slope(20), lambda: H.saturation_vp_slope(20, method="tetens")),
        (lambda: H.saturation_vp_ice(-10), lambda: H.saturation_vp_ice(-10, method="tetens")),
        (lambda: H.relative_humidity(20, 10), lambda: H.relative_humidity(20, 10, method="tetens")),
        (
            lambda: H.vapor_pressure_deficit(25, 60),
            lambda: H.vapor_pressure_deficit(25, 60, method="tetens"),
        ),
        (
            lambda: H.specific_humidity_from_dewpoint(10, 101.325),
            lambda: H.specific_humidity_from_dewpoint(10, 101.325, method="tetens"),
        ),
        (
            lambda: H.relative_humidity_from_specific_humidity(0.00757, 20, 101.325),
            lambda: H.relative_humidity_from_specific_humidity(
                0.00757, 20, 101.325, method="tetens"
            ),
        ),
        (
            lambda: H.saturation_specific_humidity(20, 101.325),
            lambda: H.saturation_specific_humidity(20, 101.325, method="tetens"),
        ),
    ],
)
def test_default_method_is_tetens(implicit, explicit):
    # Omitting method must reproduce the explicit Tetens result exactly, so
    # existing 1.1.0 calls are unaffected by the new parameter.
    assert implicit() == explicit()


def test_tetens_defaults_unchanged_from_v110():
    # Values published in 1.1.0 docstrings must not move in 1.2.0.
    assert H.saturation_vp(20) == pytest.approx(2.3382, abs=1e-4)
    assert H.actual_vp(10) == pytest.approx(1.2279, abs=1e-4)
    assert H.saturation_vp_slope(20) == pytest.approx(0.1447, abs=1e-4)
    assert H.saturation_vp_ice(-10) == pytest.approx(0.2595, abs=1e-4)


@pytest.mark.parametrize("func", SVP_FUNCS_T)
def test_invalid_method_raises(func):
    with pytest.raises(ValidationError):
        func("ae96")


@pytest.mark.parametrize("method", ["ALDUCHOV", "Tetens"])
def test_method_is_case_insensitive(method):
    assert H.saturation_vp(20, method=method) == H.saturation_vp(
        20, method=method.lower()
    )


def test_saturation_vp_alduchov_value():
    assert H.saturation_vp(20, method="alduchov") == pytest.approx(2.3334, abs=1e-4)


def test_actual_vp_alduchov_value():
    assert H.actual_vp(10, method="alduchov") == pytest.approx(1.2260, abs=1e-4)


def test_saturation_vp_slope_alduchov_value():
    assert H.saturation_vp_slope(20, method="alduchov") == pytest.approx(0.1445, abs=1e-4)


def test_saturation_vp_slope_alduchov_matches_numeric_derivative():
    # The analytic Alduchov slope must agree with a central finite difference.
    h = 1e-4
    numeric = (
        H.saturation_vp(20 + h, method="alduchov") - H.saturation_vp(20 - h, method="alduchov")
    ) / (2 * h)
    assert H.saturation_vp_slope(20, method="alduchov") == pytest.approx(numeric, rel=1e-6)


def test_saturation_vp_ice_alduchov_value():
    assert H.saturation_vp_ice(-10, method="alduchov") == pytest.approx(0.2597, abs=1e-4)


@pytest.mark.parametrize("temp", [-40.0, -20.0, 30.0, 50.0])
def test_alduchov_tracks_goff_gratch_reference(temp):
    # Alduchov & Eskridge (1996) report < 0.384 % error over -40 to 50 C
    # relative to standard reference formulations; Goff-Gratch values (kPa).
    reference = {-40.0: 0.0188944, -20.0: 0.125292, 30.0: 4.2406, 50.0: 12.3334}
    assert H.saturation_vp(temp, method="alduchov") == pytest.approx(
        reference[temp], rel=0.004
    )


def test_alduchov_more_accurate_than_tetens_in_cold():
    # At -40 C Tetens is ~2.5 % low against Goff-Gratch; Alduchov is ~0.4 %.
    reference = 0.0188944  # Goff-Gratch at -40 C (kPa)
    err_tetens = abs(H.saturation_vp(-40) / reference - 1)
    err_alduchov = abs(H.saturation_vp(-40, method="alduchov") / reference - 1)
    assert err_alduchov < err_tetens


def test_methods_agree_closely_at_typical_temperatures():
    # Both formulas are within ~0.2 % of each other between 0 and 40 C.
    t = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    ratio = H.saturation_vp(t, method="alduchov") / H.saturation_vp(t)
    assert np.all(np.abs(ratio - 1) < 0.0025)


def test_alduchov_array_input():
    out = H.saturation_vp(np.array([0.0, 20.0]), method="alduchov")
    assert out.shape == (2,)
    assert out[1] == pytest.approx(2.3334, abs=1e-4)


def test_saturation_specific_humidity_value():
    assert H.saturation_specific_humidity(20, 101.325) == pytest.approx(0.01448, abs=1e-5)


def test_saturation_specific_humidity_matches_specific_humidity():
    # Must equal specific_humidity evaluated at the saturation vapor pressure.
    es = H.saturation_vp(20)
    assert H.saturation_specific_humidity(20, 101.325) == pytest.approx(
        H.specific_humidity(es, 101.325)
    )


def test_saturation_specific_humidity_exceeds_actual():
    # Unsaturated air holds less vapor than saturated air at the same T and P.
    assert H.saturation_specific_humidity(20, 101.325) > H.specific_humidity_from_dewpoint(
        10, 101.325
    )


def test_saturation_specific_humidity_increases_with_temperature():
    assert H.saturation_specific_humidity(30, 101.325) > H.saturation_specific_humidity(
        20, 101.325
    )


def test_saturation_specific_humidity_bad_pressure():
    with pytest.raises(OutOfRangeError):
        H.saturation_specific_humidity(20, 1.0)
