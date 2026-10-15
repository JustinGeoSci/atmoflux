"""
atmoflux.solar
=================
Computes solar position and radiation geometry.
Includes day-of-year helpers, solar declination, hour angle, zenith and
elevation angles, daylight hours, extraterrestrial (top-of-atmosphere)
radiation, and clear-sky shortwave estimates.

"""

from __future__ import annotations

# Standard imports
from datetime import date

# Outside imports
import numpy as np

# imports from within atmoflux
from .constants import SOLAR_CONSTANT, SOLAR_DECLINATION_MAX, ANGSTROM_A, ANGSTROM_B
from .core import FloatArray, IntArray
from .exceptions import OutOfRangeError


def day_of_year(year: int, month: int, day: int) -> int:
    """
    Day-of-year (ordinal day) for a calendar date.

    Parameters
    ----------
    year : Four-digit year.
    month : Month number (1-12).
    day : Day of month.

    Returns
    -------
    Day of year as an integer in [1, 366].

    Raises
    ------
    OutOfRangeError
        If the supplied values do not form a valid calendar date.

    Examples
    --------
    >>> day_of_year(2024, 1, 1)
    1
    >>> day_of_year(2023, 12, 31)
    365
    """
    try:
        return date(year, month, day).timetuple().tm_yday
    except ValueError as exc:
        raise OutOfRangeError(f"Invalid calendar date: {exc}") from exc


def solar_declination(doy: IntArray) -> FloatArray:
    """
    Solar declination angle for a given day of year.

    The declination is the angle between the sun's rays and the equatorial
    plane, varying seasonally with Earth's axial tilt.

    Parameters
    ----------
    doy : Day of year in [1, 366].

    Returns
    -------
    Solar declination in degrees.

    Raises
    ------
    OutOfRangeError
        If doy is outside [1, 366].

    Notes
    -----
    Cooper's approximation:
    delta = 23.45 * sin(360 * (284 + n) / 365)
    with n the day of year and the argument in degrees.

    Examples
    --------
    >>> print(round(solar_declination(172), 2))
    23.45
    >>> print(round(solar_declination(355), 2))
    -23.45
    """
    if np.any(doy < 1) or np.any(doy > 366):
        raise OutOfRangeError("Day of year must be in [1, 366].")
    
    angle = np.radians(360.0 * (284 + doy) / 365.0)
    delta = SOLAR_DECLINATION_MAX * np.sin(angle)

    return delta


def hour_angle(solar_time: FloatArray) -> FloatArray:
    """
    Solar hour angle from local solar time.

    The hour angle is the angular displacement of the sun east or west of the
    local meridian, zero at solar noon and increasing by 15 degrees per hour.

    Parameters
    ----------
    solar_time : Local solar time in hours [0, 24).

    Returns
    -------
    Hour angle in degrees; negative before solar noon, positive after.

    Raises
    ------
    OutOfRangeError
        If solar_time is outside [0, 24).

    Notes
    -----
    H = 15 * (t_solar - 12)

    Examples
    --------
    >>> hour_angle(12.0)
    0.0
    >>> hour_angle(6.0)
    -90.0
    """
    if np.any(solar_time < 0) or np.any(solar_time >= 24):
        raise OutOfRangeError("Solar time must be in [0, 24).")
    
    h = 15.0 * (solar_time - 12.0)

    return h


def solar_zenith_angle(latitude: FloatArray, declination: FloatArray, hour_angle: FloatArray) -> FloatArray:
    """
    Solar zenith angle from latitude, declination, and hour angle.

    The zenith angle is measured from the local vertical; 0 degrees is the sun
    directly overhead and 90 degrees is the horizon.

    Parameters
    ----------
    latitude : Geographic latitude in degrees, positive north.
    declination : Solar declination in degrees.
    hour_angle : Solar hour angle in degrees.

    Returns
    -------
    Solar zenith angle in degrees [0, 180].

    Notes
    -----
    cos(theta_z) = sin(lat) * sin(delta) + cos(lat) * cos(delta) * cos(H)

    Examples
    --------
    >>> print(round(solar_zenith_angle(0.0, 0.0, 0.0), 2))
    0.0
    >>> print(round(solar_zenith_angle(40.0, 20.0, 0.0), 2))
    20.0
    """
    lat = np.radians(latitude)
    dec = np.radians(declination)
    ha = np.radians(hour_angle)
    cos_z = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    cos_z = np.clip(cos_z, -1.0, 1.0)
    theta_z = np.degrees(np.arccos(cos_z))

    return theta_z


def solar_elevation(latitude: FloatArray, declination: FloatArray, hour_angle: FloatArray) -> FloatArray:
    """
    Solar elevation (altitude) angle above the horizon.

    Parameters
    ----------
    latitude : Geographic latitude in degrees, positive north.
    declination : Solar declination in degrees.
    hour_angle : Solar hour angle in degrees.

    Returns
    -------
    Solar elevation angle in degrees; negative when the sun is below the horizon.

    Notes
    -----
    Elevation is the complement of the zenith angle:
    elevation = 90 - theta_z

    Examples
    --------
    >>> print(round(solar_elevation(40.0, 20.0, 0.0), 2))
    70.0
    """
    alpha = 90.0 - solar_zenith_angle(latitude, declination, hour_angle)

    return alpha


def sunset_hour_angle(latitude: FloatArray, declination: FloatArray) -> FloatArray:
    """
    Sunset hour angle for a given latitude and declination.

    Parameters
    ----------
    latitude : Geographic latitude in degrees, positive north.
    declination : Solar declination in degrees.

    Returns
    -------
    Sunset hour angle in degrees [0, 180]. Returns 180 for polar day and 0 for
    polar night.

    Notes
    -----
    cos(Hs) = -tan(lat) * tan(delta)
    The argument is clipped to [-1, 1] so polar day/night return 180 or 0.

    Examples
    --------
    >>> print(round(sunset_hour_angle(0.0, 0.0), 2))
    90.0
    """
    lat = np.radians(latitude)
    dec = np.radians(declination)
    cos_hs = np.clip(-np.tan(lat) * np.tan(dec), -1.0, 1.0)
    hs =  np.degrees(np.arccos(cos_hs))

    return hs


def daylight_hours(latitude: FloatArray, declination: FloatArray) -> FloatArray:
    """
    Length of daylight for a given latitude and declination.

    Parameters
    ----------
    latitude : Geographic latitude in degrees, positive north.
    declination : Solar declination in degrees.

    Returns
    -------
    Daylight duration in hours [0, 24].

    Notes
    -----
    N = (2 / 15) * Hs
    with Hs the sunset hour angle in degrees.

    Examples
    --------
    >>> print(round(daylight_hours(0.0, 0.0), 2))
    12.0
    """
    n = (2.0 / 15.0) * sunset_hour_angle(latitude, declination)

    return n


def extraterrestrial_radiation(latitude: FloatArray, doy: IntArray) -> FloatArray:
    """
    Daily extraterrestrial (top-of-atmosphere) solar radiation.

    The radiation incident on a horizontal surface at the top of the atmosphere,
    integrated over the day, following the FAO-56 formulation.

    Parameters
    ----------
    latitude : Geographic latitude in degrees, positive north.
    doy : Day of year in [1, 366].

    Returns
    -------
    Daily extraterrestrial radiation in MJ/m²/day.

    Raises
    ------
    OutOfRangeError
        If doy is outside [1, 366].

    Notes
    -----
    FAO-56 daily integration:
    dr = 1 + 0.033 * cos(2*pi*n / 365)
    Ra = (24*60/pi) * Gsc_MJ * dr * (Hs*sin(lat)*sin(delta)
         + cos(lat)*cos(delta)*sin(Hs))
    with Gsc the solar constant expressed in MJ/(m²·min) and Hs in radians.

    Examples
    --------
    >>> print(round(extraterrestrial_radiation(0.0, 172), 2))
    33.24
    """
    if np.any(doy < 1) or np.any(doy > 366):
        raise OutOfRangeError("Day of year must be in [1, 366].")
    
    dec = np.radians(solar_declination(doy))
    lat = np.radians(latitude)
    hs = np.radians(sunset_hour_angle(latitude, solar_declination(doy)))
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)

    # Solar constant in MJ/(m^2 min): 1361.6 W/m^2 * 60 s / 1e6
    gsc_mj = SOLAR_CONSTANT * 60.0 / 1.0e6
    ra = (
        (24 * 60 / np.pi)
        * gsc_mj
        * dr
        * (hs * np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.sin(hs))
    )

    return ra


def clear_sky_radiation(
    extraterrestrial: FloatArray,
    sunshine_hours: FloatArray,
    daylight: FloatArray,
    a: FloatArray = ANGSTROM_A,
    b: FloatArray = ANGSTROM_B,
) -> FloatArray:
    """
    Incoming shortwave radiation from the Angstrom-Prescott relation.

    Estimates global shortwave radiation at the surface from extraterrestrial
    radiation and the fraction of bright sunshine.

    Parameters
    ----------
    extraterrestrial : Extraterrestrial radiation Ra (MJ/m²/day).
    sunshine_hours : Actual bright-sunshine duration (hours), non-negative.
    daylight : Maximum possible daylight duration N (hours), must be positive.
    a : Angstrom-Prescott offset coefficient (default from constants).
    b : Angstrom-Prescott slope coefficient (default from constants).

    Returns
    -------
    Incoming shortwave radiation at the surface (MJ/m²/day).

    Raises
    ------
    OutOfRangeError
        If sunshine_hours is negative or daylight is not positive.

    Notes
    -----
    Angstrom-Prescott:
    Rs = (a + b * n / N) * Ra
    with n the actual and N the maximum possible sunshine hours.

    Examples
    --------
    >>> print(round(clear_sky_radiation(36.16, 10.0, 12.0), 3))
    24.107
    """
    if np.any(sunshine_hours < 0):
        raise OutOfRangeError("Sunshine hours must be non-negative.")
    if np.any(daylight <= 0):
        raise OutOfRangeError("Daylight duration must be positive.")
    
    rs = (a + b * sunshine_hours / daylight) * extraterrestrial

    return rs
