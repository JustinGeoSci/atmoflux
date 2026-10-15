"""
atmoflux.humidity
=================
Contains functions and derived variables related to atmospheric moisture. 
Includes saturation and actual vapor pressure (over water and ice), the slope
of the saturation curve, relative, specific, and saturation specific humidity,
mixing ratio, vapor pressure deficit, absolute humidity, and precipitable water.

"""

from __future__ import annotations

# Outside imports
import numpy as np

# imports from within atmoflux
from .constants import RMW, R_VAPOR
from .core import FloatArray
from .exceptions import OutOfRangeError
from .temperature import convert_temperature, _svp_coefficients


def saturation_vp(temp: FloatArray, unit: str = "C", method: str = "tetens") -> FloatArray:
    """
    Saturation vapor pressure of water (kPa) from a Magnus-form formula.

    Parameters
    -----
    temp: Air temperature.
    unit: Unit of temperature. "C", "F", or "K" (default is "C").
    method : Formula: "tetens" (default) or "alduchov".

    Returns
    -----
    Saturation vapor pressure of water in kilopascals (kPa).

    Raises
    ------
    ValidationError
        If temp is not numeric, or method is not "tetens" or "alduchov".
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Magnus form:
    es = A * exp(B * T / (T + C))
    with T in °C. Tetens: A = 0.61078 kPa, B = 17.27, C = 237.3 °C.
    Alduchov & Eskridge: A = 0.61094 kPa, B = 17.625, C = 243.04 °C.

    Examples
    --------
    >>> print(round(saturation_vp(20), 4))
    2.3382
    >>> print(round(saturation_vp(20, method="alduchov"), 4))
    2.3334
    """
    a, b, c = _svp_coefficients(method)
    unit = unit.upper()

    if unit != "C":
        temp_C = convert_temperature(temp, unit, "C")
    else:
        temp_C = temp

    es = a * np.exp((b * temp_C) / (temp_C + c))

    return es


def actual_vp(dewpoint: FloatArray, unit: str = "C", method: str = "tetens") -> FloatArray:
    """
    Actual vapor pressure of water (kPa) from dew point using a Magnus-form formula.

    Parameters
    -----
    dewpoint: Dew point temperature.
    unit : Unit of temperature: "C", "F", or "K" (default is "C").
    method : Formula: "tetens" (default) or "alduchov".

    Returns
    -----
    Actual vapor pressure of water in kilopascals (kPa).

    Raises
    ------
    ValidationError
        If dewpoint is not numeric, or method is not "tetens" or "alduchov".
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    The saturation vapor pressure evaluated at the dew point:
    e = A * exp(B * Td / (Td + C)), with the coefficients of saturation_vp.

    Examples
    --------
    >>> print(round(actual_vp(10), 4))
    1.2279
    >>> print(round(actual_vp(10, method="alduchov"), 4))
    1.226
    """
    a, b, c = _svp_coefficients(method)
    unit = unit.upper()

    if unit != "C":
        Td_C = convert_temperature(dewpoint, unit, "C")
    else:
        Td_C = dewpoint

    e = a * np.exp((b * Td_C) / (Td_C + c))

    return e


def saturation_vp_slope(temp: FloatArray, unit: str = "C", method: str = "tetens") -> FloatArray:
    """
    Slope of the saturation vapor pressure curve (kPa/°C).

    The slope (commonly denoted delta) is the rate of change of saturation vapor
    pressure with temperature and is a key term in the Penman and
    Penman-Monteith evaporation equations.

    Parameters
    ----------
    temp : Air temperature.
    unit : Unit of temperature: "C", "F", or "K" (default is "C").
    method : Formula: "tetens" (default) or "alduchov".

    Returns
    -------
    Slope of the saturation vapor pressure curve in kPa per degree Celsius.

    Raises
    ------
    ValidationError
        If temp is not numeric, or method is not "tetens" or "alduchov".
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Analytic derivative of the Magnus form:
    delta = B * C * es(T) / (T + C) ** 2
    where es(T) is the saturation vapor pressure in kPa and T is in °C.
    For "tetens" the FAO-56 published coefficient B * C = 4098 is used; for
    "alduchov" B * C = 17.625 * 243.04.

    Examples
    --------
    >>> print(round(saturation_vp_slope(20), 4))
    0.1447
    >>> print(round(saturation_vp_slope(20, method="alduchov"), 4))
    0.1445
    """
    _, b, c = _svp_coefficients(method)
    unit = unit.upper()
    if unit != "C":
        temp_C = convert_temperature(temp, unit, "C")
    else:
        temp_C = temp

    # FAO-56 specifies 4098 (= 17.27 * 237.3, rounded) for the Tetens slope.
    if method.lower() == "tetens":
        bc = 4098.0
    else:
        bc = b * c

    es = saturation_vp(temp_C, "C", method)
    delta = bc * es / (temp_C + c) ** 2

    return delta


def relative_humidity(
    temp: FloatArray, dewpoint: FloatArray, unit: str = "C", method: str = "tetens"
) -> FloatArray:
    """
    Relative humidity (%) from temperature and dew point.

    Parameters
    ----------
    temp : Air temperature.
    dewpoint : Dew point temperature (same unit as temp).
    unit : Unit of the input temperatures: "C", "F", or "K" (default "C").
    method : Formula: "tetens" (default) or "alduchov".
    Returns
    -------
    Relative humidity as a percentage.

    Raises
    ------
    InvalidUnitError
        If unit is invalid.
    ValidationError
        If method is not "tetens" or "alduchov".

    Notes
    -----
    RH = 100 * e(Td) / es(T)
    the ratio of actual to saturation vapor pressure.

    Examples
    --------
    >>> print(round(relative_humidity(20, 10), 1))
    52.5
    """
    es = saturation_vp(temp, unit, method)
    ea = actual_vp(dewpoint, unit, method)
    rh = 100.0 * ea / es

    return rh


def specific_humidity(vapor_pressure: FloatArray, pressure: FloatArray) -> FloatArray:
    """
    Specific humidity (kg/kg) from vapor pressure and total pressure.

    Specific humidity is the mass of water vapor per unit mass of moist air.

    Parameters
    ----------
    vapor_pressure : Actual vapor pressure (kPa).
    pressure : Total air pressure (kPa), must exceed vapor_pressure.

    Returns
    -------
    Specific humidity in kg of water vapor per kg of moist air.

    Raises
    ------
    OutOfRangeError
        If pressure is not greater than vapor_pressure.

    Notes
    -----
    q = RMW * e / (P - (1 - RMW) * e)
    where RMW is the ratio of molecular weights of water vapor to dry air.

    Examples
    --------
    >>> print(round(specific_humidity(1.2279, 101.325), 5))
    0.00757
    """
    if np.any(pressure <= vapor_pressure):
        raise OutOfRangeError("Total pressure must exceed vapor pressure.")
    
    q = RMW * vapor_pressure / (pressure - (1 - RMW) * vapor_pressure)

    return q


def mixing_ratio(vapor_pressure: FloatArray, pressure: FloatArray) -> FloatArray:
    """
    Mixing ratio (kg/kg) from vapor pressure and total pressure.

    The mixing ratio is the mass of water vapor per unit mass of dry air.

    Parameters
    ----------
    vapor_pressure : Actual vapor pressure (kPa).
    pressure : Total air pressure (kPa), must exceed vapor_pressure.

    Returns
    -------
    Mixing ratio in kg of water vapor per kg of dry air.

    Raises
    ------
    OutOfRangeError
        If pressure is not greater than vapor_pressure.

    Notes
    -----
    w = RMW * e / (P - e)

    Examples
    --------
    >>> print(round(mixing_ratio(1.2279, 101.325), 5))
    0.00763
    """
    if np.any(pressure <= vapor_pressure):
        raise OutOfRangeError("Total pressure must exceed vapor pressure.")
    
    w = RMW * vapor_pressure / (pressure - vapor_pressure)

    return w


def vapor_pressure_deficit(
    temp: FloatArray, rh: FloatArray, unit: str = "C", method: str = "tetens"
) -> FloatArray:
    """
    Vapor pressure deficit (kPa) from temperature and relative humidity.

    The vapor pressure deficit (VPD) is the difference between the saturation and
    actual vapor pressures and indicates the drying capacity of the air.

    Parameters
    ----------
    temp : Air temperature.
    rh : Relative humidity (%), in the interval (0, 100].
    unit : Unit of temperature: "C", "F", or "K" (default "C").
    method : Formula: "tetens" (default) or "alduchov".

    Returns
    -------
    Vapor pressure deficit in kilopascals (kPa).

    Raises
    ------
    OutOfRangeError
        If rh is not in the interval (0, 100].
    InvalidUnitError
        If unit is invalid.
    ValidationError
        If method is not "tetens" or "alduchov".

    Notes
    -----
    VPD = es(T) * (1 - RH / 100)

    Examples
    --------
    >>> print(round(vapor_pressure_deficit(25, 60), 4))
    1.2671
    """
    if np.any(rh <= 0) or np.any(rh > 100):
        raise OutOfRangeError("Relative humidity must be between 0 and 100%")
    
    es = saturation_vp(temp, unit, method)
    vpd = es * (1 - rh / 100.0)

    return vpd


def absolute_humidity(vapor_pressure: FloatArray, temp: FloatArray, unit: str = "C") -> FloatArray:
    """
    Absolute humidity (kg/m³) from vapor pressure and temperature.

    Absolute humidity is the density of water vapor in the air.

    Parameters
    ----------
    vapor_pressure : Actual vapor pressure (kPa).
    temp : Air temperature.
    unit : Unit of temperature: "C", "F", or "K" (default "C").

    Returns
    -------
    Absolute humidity in kg of water vapor per cubic meter.

    Raises
    ------
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Ideal gas law for water vapor:
    rho_v = e / (R_vapor * T)
    with e in pascals and T in kelvin.

    Examples
    --------
    >>> print(round(absolute_humidity(1.2279, 20), 5))
    0.00908
    """
    temp_K = convert_temperature(temp, unit.upper(), "K")

    # Convert vapor pressure from kPa to Pa for SI consistency.
    e = vapor_pressure * 1000.0
    rho_v = e / (R_VAPOR * temp_K)

    return rho_v


def saturation_vp_ice(temp: FloatArray, unit: str = "C", method: str = "tetens") -> FloatArray:
    """
    Saturation vapor pressure over ice (kPa) from a Magnus-form formula.

    Parameters
    ----------
    temp : Air temperature (typically <= 0 °C).
    unit : Unit of temperature: "C", "F", or "K" (default is "C").
    method : Formula: "tetens" (default) or "alduchov".

    Returns
    -------
    Saturation vapor pressure over ice in kilopascals (kPa).

    Raises
    ------
    InvalidUnitError
        If unit is invalid.
    ValidationError
        If method is not "tetens" or "alduchov".

    Notes
    -----
    Magnus form with ice coefficients:
    es_ice = A * exp(B * T / (T + C))
    with T in °C. Tetens: A = 0.61078 kPa, B = 21.875, C = 265.5 °C.
    Alduchov & Eskridge: A = 0.61121 kPa, B = 22.587, C = 273.86 °C.

    Examples
    --------
    >>> print(round(saturation_vp_ice(-10), 4))
    0.2595
    >>> print(round(saturation_vp_ice(-10, method="alduchov"), 4))
    0.2597
    """
    a, b, c = _svp_coefficients(method, ice=True)
    temp_C = convert_temperature(temp, unit.upper(), "C")
    es_ice = a * np.exp(b * temp_C / (temp_C + c))

    return es_ice


def saturation_specific_humidity(
    temp: FloatArray, pressure: FloatArray, unit: str = "C", method: str = "tetens"
) -> FloatArray:
    """
    Saturation specific humidity (kg/kg) from temperature and total pressure.

    The specific humidity air would have if saturated at the given temperature
    and pressure; the upper bound on specific humidity for those conditions.

    Parameters
    ----------
    temp : Air temperature.
    pressure : Total air pressure (kPa), must exceed the saturation vapor
        pressure.
    unit : Unit of temp: "C", "F", or "K" (default "C").
    method : Formula: "tetens" (default) or "alduchov".

    Returns
    -------
    Saturation specific humidity in kg of water vapor per kg of moist air.

    Raises
    ------
    OutOfRangeError
        If pressure is not greater than the saturation vapor pressure.
    InvalidUnitError
        If unit is invalid.
    ValidationError
        If method is not "tetens" or "alduchov".

    Notes
    -----
    Exact form, consistent with specific_humidity:
    qs = RMW * es / (P - (1 - RMW) * es)
    The common approximation qs ≈ RMW * es / P neglects the (1 - RMW) * es term,
    which is small at typical surface conditions.

    Examples
    --------
    >>> print(round(saturation_specific_humidity(20, 101.325), 5))
    0.01448
    """
    es = saturation_vp(temp, unit, method)
    qs = specific_humidity(es, pressure)

    return qs


def specific_humidity_from_dewpoint(
    dewpoint: FloatArray, pressure: FloatArray, unit: str = "C", method: str = "tetens"
) -> FloatArray:
    """
    Specific humidity (kg/kg) from dew point and total pressure.

    Parameters
    ----------
    dewpoint : Dew point temperature.
    pressure : Total air pressure (kPa), must exceed the vapor pressure.
    unit : Unit of dewpoint: "C", "F", or "K" (default "C").
    method : Formula: "tetens" (default) or "alduchov".

    Returns
    -------
    Specific humidity in kg of water vapor per kg of moist air.

    Raises
    ------
    OutOfRangeError
        If pressure is not greater than the actual vapor pressure.
    InvalidUnitError
        If unit is invalid.
    ValidationError
        If method is not "tetens" or "alduchov".

    Notes
    -----
    Computes actual vapor pressure from the dew point, then applies
    q = RMW * e / (P - (1 - RMW) * e).

    Examples
    --------
    >>> print(round(specific_humidity_from_dewpoint(10, 101.325), 5))
    0.00757
    """
    e = actual_vp(dewpoint, unit, method)

    if np.any(pressure <= e):
        raise OutOfRangeError("Total pressure must exceed vapor pressure.")
    
    q = RMW * e / (pressure - (1 - RMW) * e)

    return q


def relative_humidity_from_specific_humidity(
    specific_humidity: FloatArray,
    temp: FloatArray,
    pressure: FloatArray,
    unit: str = "C",
    method: str = "tetens",
) -> FloatArray:
    """
    Relative humidity (%) from specific humidity, temperature, and pressure.

    Parameters
    ----------
    specific_humidity : Specific humidity (kg/kg), non-negative.
    temp : Air temperature.
    pressure : Total air pressure (kPa), must be positive.
    unit : Unit of temp: "C", "F", or "K" (default "C").
    method : Formula: "tetens" (default) or "alduchov".

    Returns
    -------
    Relative humidity as a percentage.

    Raises
    ------
    OutOfRangeError
        If specific_humidity is negative or pressure is not positive.
    InvalidUnitError
        If unit is invalid.
    ValidationError
        If method is not "tetens" or "alduchov".

    Notes
    -----
    Inverts the specific-humidity definition to recover vapor pressure,
    e = q * P / (RMW + (1 - RMW) * q), then divides by saturation vapor pressure.

    Examples
    --------
    >>> print(round(relative_humidity_from_specific_humidity(0.00757, 20, 101.325), 1))
    52.5
    """
    if np.any(specific_humidity < 0):
        raise OutOfRangeError("Specific humidity must be non-negative.")
    if np.any(pressure <= 0):
        raise OutOfRangeError("Pressure must be positive.")
    
    e = specific_humidity * pressure / (RMW + (1 - RMW) * specific_humidity)
    es = saturation_vp(temp, unit, method)
    rh = 100.0 * e / es

    return rh


def precipitable_water(specific_humidity: FloatArray, pressure: FloatArray) -> FloatArray:
    """
    Column precipitable water from layer-mean specific humidity.

    Estimates the depth of liquid water that would result if all the water vapor
    in an atmospheric column were condensed, using a single layer-mean specific
    humidity and the surface pressure.

    Parameters
    ----------
    specific_humidity : Layer-mean specific humidity (kg/kg), non-negative.
    pressure : Surface pressure (kPa), must be positive.

    Returns
    -------
    Precipitable water in millimeters.

    Raises
    ------
    OutOfRangeError
        If specific_humidity is negative or pressure is not positive.

    Notes
    -----
    Single-layer approximation:
    PW = q * P / (rho_water * g)
    with P converted to pascals and the result expressed in millimeters.

    Examples
    --------
    >>> print(round(precipitable_water(0.01, 101.325), 2))
    103.32
    """
    if np.any(specific_humidity < 0):
        raise OutOfRangeError("Specific humidity must be non-negative.")
    if np.any(pressure <= 0):
        raise OutOfRangeError("Pressure must be positive.")
    
    g = 9.80665
    rho_water = 1000.0
    p_pa = pressure * 1000.0

    # Depth of water in meters, converted to millimeters.
    pw = specific_humidity * p_pa / (rho_water * g) * 1000.0

    return pw