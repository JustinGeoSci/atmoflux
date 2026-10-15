"""Tests for atmoflux.aerosols."""
import numpy as np
import pytest

from atmoflux import aerosols as A
from atmoflux.exceptions import OutOfRangeError


def test_settling_velocity_submicron():
    assert A.settling_velocity(1e-6, 1000.0) * 1e6 == pytest.approx(35.486, abs=1e-2)


def test_settling_velocity_ten_micron():
    assert A.settling_velocity(10e-6, 1000.0) * 1e3 == pytest.approx(3.092, abs=1e-2)


def test_settling_velocity_increases_with_size():
    small = A.settling_velocity(1e-6, 1000.0)
    large = A.settling_velocity(10e-6, 1000.0)
    assert large > small


def test_settling_velocity_bad_diameter():
    with pytest.raises(OutOfRangeError):
        A.settling_velocity(0.0, 1000.0)


def test_settling_velocity_bad_density():
    with pytest.raises(OutOfRangeError):
        A.settling_velocity(1e-6, -10.0)


def test_dry_deposition_velocity_value():
    vs = A.settling_velocity(10e-6, 1000.0)
    assert A.dry_deposition_velocity(vs, 50.0, 20.0) * 1000 == pytest.approx(
        16.774, abs=1e-2
    )


def test_dry_deposition_exceeds_settling():
    vs = A.settling_velocity(10e-6, 1000.0)
    assert A.dry_deposition_velocity(vs, 50.0, 20.0) > vs


def test_dry_deposition_bad_resistance():
    with pytest.raises(OutOfRangeError):
        A.dry_deposition_velocity(0.003, 0.0, 20.0)


def test_emission_flux_value():
    assert A.emission_flux(2e-6, 0.01) == pytest.approx(2e-8)


def test_emission_flux_negative():
    with pytest.raises(OutOfRangeError):
        A.emission_flux(-1.0, 0.01)


def test_particle_reynolds_number_value():
    vs = A.settling_velocity(10e-6, 1000.0)
    assert A.particle_reynolds_number(10e-6, vs) == pytest.approx(0.00212, abs=1e-5)


def test_particle_reynolds_number_formula():
    # Re = rho * U * d / mu with explicit inputs.
    assert A.particle_reynolds_number(1e-4, 0.5, density=1.2, viscosity=1.8e-5) == pytest.approx(
        1.2 * 0.5 * 1e-4 / 1.8e-5
    )


def test_settling_in_stokes_regime_for_fine_particles():
    # Typical fine aerosol settling is deep in the Stokes regime (Re << 1).
    for d in (1e-6, 10e-6):
        vs = A.settling_velocity(d, 1000.0)
        assert A.particle_reynolds_number(d, vs) < 0.1


def test_stokes_regime_breaks_down_for_large_particles():
    # A 200-micron droplet settles fast enough that Stokes' law no longer holds.
    vs = A.settling_velocity(200e-6, 1000.0)
    assert A.particle_reynolds_number(200e-6, vs) > 1


def test_particle_reynolds_number_zero_velocity():
    assert A.particle_reynolds_number(1e-6, 0.0) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"diameter": 0.0, "velocity": 0.01},
        {"diameter": 1e-6, "velocity": -0.01},
        {"diameter": 1e-6, "velocity": 0.01, "density": 0.0},
        {"diameter": 1e-6, "velocity": 0.01, "viscosity": 0.0},
    ],
)
def test_particle_reynolds_number_invalid(kwargs):
    with pytest.raises(OutOfRangeError):
        A.particle_reynolds_number(**kwargs)


def test_particle_reynolds_number_array():
    out = A.particle_reynolds_number(np.array([1e-6, 1e-5]), np.array([0.01, 0.01]))
    assert out[1] == pytest.approx(10 * out[0])
