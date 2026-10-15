"""
atmoflux.stability
=================
Computes atmospheric surface-layer stability diagnostics.
Includes the bulk Richardson number, Obukhov length, a Pasquill-style stability
classification, and the integrated Monin-Obukhov stability correction functions
for momentum and heat. All formulations are closed-form (non-iterative).

"""

from __future__ import annotations

# Outside imports
import numpy as np

# imports from within atmoflux
from .constants import G, CP_AIR, KARMAN
from .core import FloatArray
from .exceptions import OutOfRangeError
from .temperature import convert_temperature


def bulk_richardson_number(
    temp_lower: FloatArray,
    temp_upper: FloatArray,
    wind_upper: FloatArray,
    height_lower: FloatArray,
    height_upper: FloatArray,
    unit: str = "C",
) -> FloatArray:
    """
    Bulk Richardson number between two heights.

    The bulk Richardson number is a dimensionless measure of dynamic stability,
    comparing buoyant suppression or production of turbulence to mechanical wind
    shear. Positive values indicate stable stratification, negative unstable, and
    zero neutral.

    Parameters
    ----------
    temp_lower : Temperature at the lower height.
    temp_upper : Temperature at the upper height.
    wind_upper : Wind speed at the upper height (m/s), nonzero.
    height_lower : Lower height (m).
    height_upper : Upper height (m), must differ from height_lower.
    unit : Unit of the input temperatures: "C", "F", or "K" (default "C").

    Returns
    -------
    Bulk Richardson number (dimensionless).

    Raises
    ------
    OutOfRangeError
        If the two heights are equal or wind_upper is zero.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Rb = (g * dTheta * dz) / (T_mean * U ** 2)
    Potential temperature differences are approximated by the temperature
    difference for shallow near-surface layers; temperatures are taken in kelvin.

    Examples
    --------
    >>> print(round(bulk_richardson_number(15, 14, 5.0, 2.0, 10.0), 5))
    -0.01091
    """
    if np.any(height_upper == height_lower):
        raise OutOfRangeError("Upper and lower heights must differ.")
    if np.any(wind_upper == 0):
        raise OutOfRangeError("Upper wind speed must be nonzero.")
    
    t_lower_K = convert_temperature(temp_lower, unit.upper(), "K")
    t_upper_K = convert_temperature(temp_upper, unit.upper(), "K")
    t_mean = (t_lower_K + t_upper_K) / 2.0
    dz = height_upper - height_lower
    dtheta = t_upper_K - t_lower_K
    rb = (G * dtheta * dz) / (t_mean * wind_upper**2)

    return rb


def obukhov_length(
    friction_velocity: FloatArray,
    temp: FloatArray,
    sensible_heat_flux: FloatArray,
    density: FloatArray,
    unit: str = "C",
) -> FloatArray:
    """
    Monin-Obukhov length from friction velocity and sensible heat flux.

    The Obukhov length is the height at which buoyant and shear production of
    turbulence are comparable. It is negative in unstable (convective)
    conditions, positive in stable conditions, and large in magnitude near
    neutral.

    Parameters
    ----------
    friction_velocity : Friction velocity u* (m/s), must be positive.
    temp : Air temperature.
    sensible_heat_flux : Surface sensible heat flux H (W/m²), nonzero.
    density : Air density (kg/m³), must be positive.
    unit : Unit of temp: "C", "F", or "K" (default "C").

    Returns
    -------
    Obukhov length in meters.

    Raises
    ------
    OutOfRangeError
        If friction_velocity or density is not positive, or H is zero.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    L = -(rho * cp * T * u*^3) / (k * g * H)
    with T in kelvin and k the von Kármán constant.

    Examples
    --------
    >>> print(round(obukhov_length(0.3, 20, 100.0, 1.2), 3))
    -24.334
    """
    if np.any(friction_velocity <= 0):
        raise OutOfRangeError("Friction velocity must be positive.")
    if np.any(density <= 0):
        raise OutOfRangeError("Density must be positive.")
    if np.any(sensible_heat_flux == 0):
        raise OutOfRangeError("Sensible heat flux must be nonzero.")
    
    temp_K = convert_temperature(temp, unit.upper(), "K")
    l = -(density * CP_AIR * temp_K * friction_velocity**3) / (
        KARMAN * G * sensible_heat_flux
    )

    return l


def stability_parameter(height: FloatArray, obukhov_length: FloatArray) -> FloatArray:
    """
    Monin-Obukhov stability parameter zeta = z / L.

    Parameters
    ----------
    height : Height above the surface (m).
    obukhov_length : Obukhov length L (m), nonzero.

    Returns
    -------
    Stability parameter zeta (dimensionless); negative unstable, positive stable.

    Raises
    ------
    OutOfRangeError
        If obukhov_length is zero.

    Examples
    --------
    >>> print(round(stability_parameter(10.0, -50.0), 2))
    -0.2
    """
    if np.any(obukhov_length == 0):
        raise OutOfRangeError("Obukhov length must be nonzero.")
    
    zeta = height / obukhov_length

    return zeta


def psi_momentum(zeta: FloatArray) -> FloatArray:
    """
    Integrated stability correction function for momentum.

    Parameters
    ----------
    zeta : Stability parameter z / L (dimensionless).

    Returns
    -------
    Momentum stability correction psi_m (dimensionless).

    Notes
    -----
    Uses Businger-Dyer relations. For unstable conditions (zeta < 0) with
    x = (1 - 16*zeta) ** 0.25:
    psi_m = 2*ln((1+x)/2) + ln((1+x^2)/2) - 2*atan(x) + pi/2
    For stable conditions (zeta >= 0):
    psi_m = -5 * zeta

    Examples
    --------
    >>> psi_momentum(0.1)
    -0.5
    >>> print(round(psi_momentum(-0.1), 4))
    0.2836
    """
    z = np.asarray(zeta, dtype=float)
    unstable = z < 0
    x = (1 - 16 * np.where(unstable, z, 0.0)) ** 0.25
    psi_unstable = (
        2 * np.log((1 + x) / 2)
        + np.log((1 + x**2) / 2)
        - 2 * np.arctan(x)
        + np.pi / 2
    )
    psi_stable = -5 * z
    result = np.where(unstable, psi_unstable, psi_stable)

    if result.ndim == 0:
        return float(result)
    
    return result


def psi_heat(zeta: FloatArray) -> FloatArray:
    """
    Integrated stability correction function for heat.

    Parameters
    ----------
    zeta : Stability parameter z / L (dimensionless).

    Returns
    -------
    Heat stability correction psi_h (dimensionless).

    Notes
    -----
    Uses Businger-Dyer relations. For unstable conditions (zeta < 0) with
    y = (1 - 16*zeta) ** 0.5:
    psi_h = 2 * ln((1 + y) / 2)
    For stable conditions (zeta >= 0):
    psi_h = -5 * zeta

    Examples
    --------
    >>> psi_heat(0.1)
    -0.5
    >>> print(round(psi_heat(-0.1), 4))
    0.5343
    """
    z = np.asarray(zeta, dtype=float)
    unstable = z < 0
    y = (1 - 16 * np.where(unstable, z, 0.0)) ** 0.5
    psi_unstable = 2 * np.log((1 + y) / 2)
    psi_stable = -5 * z
    result = np.where(unstable, psi_unstable, psi_stable)

    if result.ndim == 0:
        return float(result)
    
    return result


def stability_class(richardson: float) -> str:
    """
    Qualitative stability class from a bulk Richardson number.

    Parameters
    ----------
    richardson : Bulk Richardson number (dimensionless).

    Returns
    -------
    One of "unstable", "neutral", or "stable".

    Notes
    -----
    A simple threshold scheme is used:
    Rb < -0.01 -> unstable, |Rb| <= 0.01 -> neutral, Rb > 0.01 -> stable.
    This is a coarse classification; thresholds vary by application.

    Examples
    --------
    >>> stability_class(-0.5)
    'unstable'
    >>> stability_class(0.0)
    'neutral'
    >>> stability_class(0.5)
    'stable'
    """
    if richardson < -0.01:
        return "unstable"
    
    if richardson > 0.01:
        return "stable"
    
    return "neutral"