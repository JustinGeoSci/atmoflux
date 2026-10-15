"""
atmoflux.atmosphere
=================
Provides standard-atmosphere and barometric helpers.
Includes pressure as a function of altitude, the hypsometric relation, scale
height, and a US Standard Atmosphere temperature/pressure profile for the
troposphere.

"""

from __future__ import annotations

# Outside imports
import numpy as np

# imports from within atmoflux
from .constants import G, R_AIR, P0, T0_STANDARD, LAPSE_RATE_STANDARD
from .core import FloatArray
from .exceptions import OutOfRangeError
from .temperature import convert_temperature


def scale_height(temp: FloatArray, unit: str = "C") -> FloatArray:
    """
    Atmospheric pressure scale height for an isothermal layer.

    The scale height is the vertical distance over which pressure decreases by a
    factor of e in an isothermal atmosphere.

    Parameters
    ----------
    temp : Air temperature.
    unit : Unit of temp: "C", "F", or "K" (default "C").

    Returns
    -------
    Scale height in meters.

    Raises
    ------
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    H = R_air * T / g
    with T in kelvin.

    Examples
    --------
    >>> print(round(scale_height(15), 1))
    8434.5
    """
    temp_K = convert_temperature(temp, unit.upper(), "K")

    h = R_AIR * temp_K / G
    
    return h


def pressure_at_altitude(
    altitude: FloatArray,
    pressure_ref: FloatArray = P0,
    temp: FloatArray = 15.0,
    unit: str = "C",
) -> FloatArray:
    """
    Pressure at a given altitude using the isothermal barometric formula.

    Parameters
    ----------
    altitude : Height above the reference level (m).
    pressure_ref : Pressure at the reference level (kPa), default sea-level P0.
    temp : Mean layer air temperature (default 15.0).
    unit : Unit of temp: "C", "F", or "K" (default "C").

    Returns
    -------
    Pressure at the requested altitude (kPa).

    Raises
    ------
    OutOfRangeError
        If pressure_ref is not positive.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Isothermal barometric law:
    P = P_ref * exp(-z / H)
    where H = R_air * T / g is the scale height.

    Examples
    --------
    >>> print(round(pressure_at_altitude(1000), 3))
    89.997
    """
    if np.any(pressure_ref <= 0):
        raise OutOfRangeError("Reference pressure must be positive.")
    
    height = scale_height(temp, unit)

    p = pressure_ref * np.exp(-altitude / height)

    return p


def hypsometric_thickness(
    pressure_lower: FloatArray,
    pressure_upper: FloatArray,
    temp: FloatArray,
    unit: str = "C",
) -> FloatArray:
    """
    Geopotential thickness between two pressure levels (hypsometric equation).

    Parameters
    ----------
    pressure_lower : Pressure at the lower level (kPa), must be positive.
    pressure_upper : Pressure at the upper level (kPa), positive and below
        pressure_lower.
    temp : Mean layer temperature.
    unit : Unit of temp: "C", "F", or "K" (default "C").

    Returns
    -------
    Layer thickness in meters.

    Raises
    ------
    OutOfRangeError
        If pressures are not positive or pressure_upper >= pressure_lower.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Hypsometric equation:
    dz = (R_air * T / g) * ln(P_lower / P_upper)

    Examples
    --------
    >>> print(round(hypsometric_thickness(101.325, 90.0, 15), 1))
    999.7
    """
    if np.any(pressure_lower <= 0) or np.any(pressure_upper <= 0):
        raise OutOfRangeError("Pressures must be positive.")
    if np.any(pressure_upper >= pressure_lower):
        raise OutOfRangeError("Upper pressure must be below lower pressure.")
    
    temp_K = convert_temperature(temp, unit.upper(), "K")
    dz = (R_AIR * temp_K / G) * np.log(pressure_lower / pressure_upper)

    return dz


def density_altitude(pressure: FloatArray, temp: FloatArray, unit: str = "C") -> FloatArray:
    """
    Density altitude from station pressure and temperature.

    Density altitude is the altitude in the standard atmosphere at which the air
    density equals the observed density. It is widely used in aviation and
    performance calculations. It can also support atmospheric characterization and 
    environmental modeling involving air density and surface energy exchange.

    Parameters
    ----------
    pressure : Station (ambient) pressure (kPa), must be positive.
    temp : Air temperature.
    unit : Unit of temp: "C", "F", or "K" (default "C").

    Returns
    -------
    Density altitude in meters.

    Raises
    ------
    OutOfRangeError
        If pressure is not positive.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Derived by inverting the standard-atmosphere density profile:
    rho = P / (R_air * T)
    DA = (T0 / L) * (1 - (rho / rho0) ** ((L * R_air) / (g - L * R_air)))
    with T0 and L the standard sea-level temperature and lapse rate.

    Examples
    --------
    >>> print(round(density_altitude(101.325, 25), 1))
    353.9
    """
    if np.any(pressure <= 0):
        raise OutOfRangeError("Pressure must be positive.")
    
    temp_K = convert_temperature(temp, unit.upper(), "K")
    rho = (pressure * 1000.0) / (R_AIR * temp_K)
    rho0 = (P0 * 1000.0) / (R_AIR * T0_STANDARD)
    exponent = (LAPSE_RATE_STANDARD * R_AIR) / (G - LAPSE_RATE_STANDARD * R_AIR)
    da = (T0_STANDARD / LAPSE_RATE_STANDARD) * (1 - (rho / rho0) ** exponent)

    return da


def standard_atmosphere(altitude: FloatArray, unit: str = "C") -> tuple[FloatArray, FloatArray]:
    """
    Temperature and pressure of the US Standard Atmosphere (troposphere).

    Valid in the troposphere up to roughly 11,000 m, where the standard lapse
    rate applies.

    Parameters
    ----------
    altitude : Geopotential altitude (m), from 0 up to about 11,000 m.
    unit : Unit of the returned temperature: "C", "F", or "K" (default "C").

    Returns
    -------
    Tuple of (temperature, pressure) where temperature is in the requested unit
    and pressure is in kPa.

    Raises
    ------
    OutOfRangeError
        If altitude is negative or above the tropospheric limit.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Standard tropospheric profile:
    T = T0 - L * z
    P = P0 * (T / T0) ** (g / (L * R_air))
    with T0 = 288.15 K, L = 0.0065 K/m.

    Examples
    --------
    >>> temp, pres = standard_atmosphere(1000)
    >>> print(round(temp, 2))
    8.5
    >>> print(round(pres, 3))
    89.875
    """
    if np.any(altitude < 0) or np.any(altitude > 11000):
        raise OutOfRangeError("Altitude must be between 0 and 11000 m.")
    
    temp_K = T0_STANDARD - LAPSE_RATE_STANDARD * altitude
    pressure = P0 * (temp_K / T0_STANDARD) ** (G / (LAPSE_RATE_STANDARD * R_AIR))
    temp_out = convert_temperature(temp_K, "K", unit.upper())

    return temp_out, pressure
