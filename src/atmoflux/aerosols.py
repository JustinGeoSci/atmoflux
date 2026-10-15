"""
atmoflux.aerosols
=================
Handles fluxes of particles and trace gases in the atmosphere that are not water. 
Includes deposition, emission, and transport calculations for aerosols, and the
particle Reynolds number for checking Stokes-regime validity.

"""

from __future__ import annotations

# Outside imports
import numpy as np

# imports from within atmoflux
from .constants import G, MU_AIR, MFP_AIR, RHO_AIR_STD
from .core import FloatArray
from .exceptions import OutOfRangeError


def settling_velocity(
    diameter: FloatArray,
    particle_density: FloatArray,
    viscosity: FloatArray = MU_AIR,
    mean_free_path: FloatArray = MFP_AIR,
) -> FloatArray:
    """
    Gravitational settling velocity of a spherical aerosol particle.

    Uses Stokes' law with the Cunningham slip correction factor, which is
    important for sub-micron particles whose size approaches the mean free path
    of air.

    Parameters
    ----------
    diameter : Particle diameter (m), must be positive.
    particle_density : Particle density (kg/m³), must be positive.
    viscosity : Dynamic viscosity of air (Pa·s), default from constants.
    mean_free_path : Mean free path of air molecules (m), default from constants.

    Returns
    -------
    Terminal settling velocity in m/s.

    Raises
    ------
    OutOfRangeError
        If diameter or particle_density is not positive.

    Notes
    -----
    Stokes settling velocity with slip correction:
    Cc = 1 + (2 * lambda / d) * (1.257 + 0.4 * exp(-1.1 * d / (2 * lambda)))
    vs = ((rho_p - rho_a) * d ** 2 * g * Cc) / (18 * mu)
    valid in the Stokes regime (small particle Reynolds number).

    Examples
    --------
    >>> print(round(settling_velocity(1e-6, 1000.0) * 1e6, 3))
    35.486
    >>> print(round(settling_velocity(10e-6, 1000.0) * 1e3, 3))
    3.092
    """
    if np.any(diameter <= 0):
        raise OutOfRangeError("Particle diameter must be positive.")
    if np.any(particle_density <= 0):
        raise OutOfRangeError("Particle density must be positive.")

    knudsen_term = 2 * mean_free_path / diameter
    cc = 1 + knudsen_term * (
        1.257 + 0.4 * np.exp(-1.1 * diameter / (2 * mean_free_path))
    )
    num = cc * G * diameter**2 * (particle_density - RHO_AIR_STD)
    den = (18 * viscosity)
    vs = num / den

    return vs


def dry_deposition_velocity(
    settling: FloatArray, resistance_aero: FloatArray, resistance_surface: FloatArray
) -> FloatArray:
    """
    Dry deposition velocity from the resistance-in-series model.

    Combines aerodynamic and surface (quasi-laminar boundary layer) resistances
    with gravitational settling to give the net downward transfer velocity.

    Parameters
    ----------
    settling : Gravitational settling velocity (m/s), must be non-negative.
    resistance_aero : Aerodynamic resistance ra (s/m), must be positive.
    resistance_surface : Surface/boundary-layer resistance rb (s/m), positive.

    Returns
    -------
    Dry deposition velocity in m/s.

    Raises
    ------
    OutOfRangeError
        If settling is negative or either resistance is not positive.

    Notes
    -----
    Resistance model with a settling-velocity contribution:
    vd = vs + 1 / (ra + rb + ra * rb * vs)

    Examples
    --------
    >>> vs = settling_velocity(10e-6, 1000.0)
    >>> print(round(dry_deposition_velocity(vs, 50.0, 20.0) * 1000, 3))
    16.774
    """
    if np.any(settling < 0):
        raise OutOfRangeError("Settling velocity must be non-negative.")
    if np.any(resistance_aero <= 0) or np.any(resistance_surface <= 0):
        raise OutOfRangeError("Resistances must be positive.")
    
    den = resistance_aero + resistance_surface + resistance_aero * resistance_surface * settling
    vd = settling + 1.0 / den

    return vd


def emission_flux(concentration: FloatArray, transfer_velocity: FloatArray) -> FloatArray:
    """
    Surface emission flux from a transfer velocity and concentration.

    Parameters
    ----------
    concentration : Source concentration at the surface (kg/m³), non-negative.
    transfer_velocity : Upward transfer (emission) velocity (m/s), non-negative.

    Returns
    -------
    Emission mass flux in kg/(m²·s).

    Raises
    ------
    OutOfRangeError
        If concentration or transfer_velocity is negative.

    Notes
    -----
    F = transfer_velocity * concentration

    Examples
    --------
    >>> print(emission_flux(2e-6, 0.01))
    2e-08
    """
    if np.any(concentration < 0) or np.any(transfer_velocity < 0):
        raise OutOfRangeError("Concentration and transfer velocity must be non-negative.")
    
    f = transfer_velocity * concentration

    return f


def particle_reynolds_number(
    diameter: FloatArray,
    velocity: FloatArray,
    density: FloatArray = RHO_AIR_STD,
    viscosity: FloatArray = MU_AIR,
) -> FloatArray:
    """
    Particle Reynolds number for a particle moving relative to air.

    The ratio of inertial to viscous forces on the particle. Its main use here
    is checking that settling_velocity is valid: Stokes' law assumes creeping
    flow, which holds for particle Reynolds numbers below roughly 1.

    Parameters
    ----------
    diameter : Particle diameter (m), must be positive.
    velocity : Particle speed relative to the air (m/s), non-negative; e.g. the
        settling velocity.
    density : Air density (kg/m³), default RHO_AIR_STD (standard sea level).
    viscosity : Dynamic viscosity of air (Pa·s), default from constants.

    Returns
    -------
    Particle Reynolds number (dimensionless).

    Raises
    ------
    OutOfRangeError
        If diameter, density, or viscosity is not positive, or velocity is
        negative.

    Notes
    -----
    Re = rho * U * d / mu
    written in the dynamic-viscosity form so that MU_AIR remains the single
    source of truth for air viscosity. Stokes' law (and settling_velocity) is
    accurate for Re < ~1; above that, drag exceeds the Stokes estimate and the
    computed settling velocity is too high.

    Examples
    --------
    >>> vs = settling_velocity(10e-6, 1000.0)
    >>> print(round(particle_reynolds_number(10e-6, vs), 5))
    0.00212
    """
    if np.any(diameter <= 0):
        raise OutOfRangeError("Particle diameter must be positive.")
    if np.any(velocity < 0):
        raise OutOfRangeError("Velocity must be non-negative.")
    if np.any(density <= 0):
        raise OutOfRangeError("Air density must be positive.")
    if np.any(viscosity <= 0):
        raise OutOfRangeError("Viscosity must be positive.")

    re = density * velocity * diameter / viscosity

    return re
