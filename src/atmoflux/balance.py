"""
atmoflux.balance
=================
Calculates net energy budgets and surface energy balance diagnostics. 
Aggregates radiative, turbulent, and storage fluxes to compute Bowen ratios,
closure residuals, and an EnergyBalance container.

"""

from __future__ import annotations

# Outside imports
import numpy as np

# imports from within atmoflux
from .core import EnergyBalance, FloatArray
from .exceptions import OutOfRangeError


def surface_energy_residual(
        net_radiation: FloatArray, 
        sensible_heat: FloatArray, 
        latent_heat: FloatArray, 
        ground_heat: FloatArray = 0.0,
) -> FloatArray:
    """
    Surface energy balance closure residual.

    Parameters
    ----------
    net_radiation : Net all-wave radiation Rn (W/m²).
    sensible_heat : Sensible heat flux H (W/m²).
    latent_heat : Latent heat flux LE (W/m²).
    ground_heat : Ground heat flux G (W/m²), default 0.0.

    Returns
    -------
    Closure residual Rn - G - H - LE (W/m²); zero for a perfectly closed budget.

    Notes
    -----
    residual = Rn - G - H - LE
    A nonzero residual indicates lack of energy balance closure, commonly seen in
    eddy-covariance observations.

    Examples
    --------
    >>> surface_energy_residual(400.0, 150.0, 200.0, 50.0)
    0.0
    """
    residual = net_radiation - ground_heat - sensible_heat - latent_heat

    return residual


def bowen_ratio(sensible_heat: FloatArray, latent_heat: FloatArray) -> FloatArray:
    """
    Bowen ratio of sensible to latent heat flux.

    Parameters
    ----------
    sensible_heat : Sensible heat flux H (W/m²).
    latent_heat : Latent heat flux LE (W/m²), should be nonzero.

    Returns
    -------
    Bowen ratio H / LE (dimensionless); ``inf`` or ``nan`` where LE is 0.

    Notes
    -----
    beta = H / LE
    Values well above 1 indicate dry surfaces dominated by sensible heating;
    values below 1 indicate moist surfaces dominated by evaporation.

    Examples
    --------
    >>> round(bowen_ratio(150.0, 200.0), 3)
    0.75
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.divide(sensible_heat, latent_heat)

    if np.ndim(beta) == 0:
        return float(beta)
    
    return beta


def energy_balance(
    net_radiation: FloatArray,
    sensible_heat: FloatArray,
    latent_heat: FloatArray,
    ground_heat: FloatArray = 0.0,
) -> EnergyBalance:
    """
    Assemble an EnergyBalance container from surface flux components.

    Parameters
    ----------
    net_radiation : Net all-wave radiation Rn (W/m²).
    sensible_heat : Sensible heat flux H (W/m²).
    latent_heat : Latent heat flux LE (W/m²).
    ground_heat : Ground heat flux G (W/m²), default 0.0.

    Returns
    -------
    An :class:`atmoflux.core.EnergyBalance` with the fluxes and derived residual
    and Bowen ratio.

    Examples
    --------
    >>> eb = energy_balance(400.0, 150.0, 200.0, 50.0)
    >>> eb.residual
    0.0
    >>> round(eb.bowen_ratio, 2)
    0.75
    """
    return EnergyBalance(
        net_radiation=net_radiation,
        sensible_heat=sensible_heat,
        latent_heat=latent_heat,
        ground_heat=ground_heat,
    )


def available_energy(net_radiation: FloatArray, ground_heat: FloatArray = 0.0) -> FloatArray:
    """
    Available energy at the surface.

    The energy available to drive the turbulent (sensible and latent) heat
    fluxes after accounting for ground heat storage.

    Parameters
    ----------
    net_radiation : Net all-wave radiation Rn (W/m²).
    ground_heat : Ground heat flux G (W/m²), default 0.0.

    Returns
    -------
    Available energy Rn - G (W/m²).

    Notes
    -----
    A = Rn - G

    Examples
    --------
    >>> available_energy(400.0, 50.0)
    350.0
    """
    a = net_radiation - ground_heat
    
    return a


def energy_balance_ratio(
    sensible_heat: FloatArray,
    latent_heat: FloatArray,
    net_radiation: FloatArray,
    ground_heat: FloatArray = 0.0,
) -> FloatArray:
    """
    Energy balance closure ratio.

    The ratio of the turbulent heat fluxes to the available energy, a common
    diagnostic of energy-balance closure in eddy-covariance measurements. A value
    of 1 indicates perfect closure.

    Parameters
    ----------
    sensible_heat : Sensible heat flux H (W/m²).
    latent_heat : Latent heat flux LE (W/m²).
    net_radiation : Net all-wave radiation Rn (W/m²).
    ground_heat : Ground heat flux G (W/m²), default 0.0.

    Returns
    -------
    Energy balance ratio (H + LE) / (Rn - G); ``inf`` or ``nan`` where the
    available energy is 0.

    Notes
    -----
    EBR = (H + LE) / (Rn - G)

    Examples
    --------
    >>> round(energy_balance_ratio(150.0, 200.0, 400.0, 50.0), 3)
    1.0
    """
    available = net_radiation - ground_heat

    with np.errstate(divide="ignore", invalid="ignore"):
        ebr = np.divide(sensible_heat + latent_heat, available)

    if np.ndim(ebr) == 0:
        return float(ebr)
    
    return ebr


def ground_heat_fraction(net_radiation: FloatArray, fraction: FloatArray = 0.1) -> FloatArray:
    """
    Estimate ground heat flux as a fraction of net radiation.

    A simple parameterization of ground heat flux when direct measurements are
    unavailable, expressing G as a fixed fraction of net radiation.

    Parameters
    ----------
    net_radiation : Net all-wave radiation Rn (W/m²).
    fraction : Fraction of net radiation partitioned to the ground (default 0.1).

    Returns
    -------
    Estimated ground heat flux G (W/m²).

    Raises
    ------
    OutOfRangeError
        If fraction is outside [0, 1].

    Notes
    -----
    G = fraction * Rn
    Typical daytime values of the fraction range from about 0.05 to 0.2 for
    vegetated surfaces.

    Examples
    --------
    >>> ground_heat_fraction(400.0)
    40.0
    """
    if np.any(fraction < 0) or np.any(fraction > 1):
        raise OutOfRangeError("Fraction must be in the interval [0, 1].")
    
    g = fraction * net_radiation
    
    return g