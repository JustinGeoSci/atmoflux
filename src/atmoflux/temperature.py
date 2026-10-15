"""
atmoflux.temperature
=================
Provides functions and derived variables related to atmospheric and surface temperature. 
Includes unit conversion, dew point, potential and virtual temperature, lapse
rate, and surface temperature retrieved from upwelling longwave radiation.

"""

from __future__ import annotations

# Outside imports
import numpy as np

# imports from within atmoflux
from .constants import (
    CP_AIR, R_AIR, P0, STEFAN_BOLTZMANN, G, LV,
    SVP_A, SVP_B, SVP_C,
    SVP_ICE_A, SVP_ICE_B, SVP_ICE_C,
    SVP_A_ALDUCHOV, SVP_B_ALDUCHOV, SVP_C_ALDUCHOV,
    SVP_ICE_A_ALDUCHOV, SVP_ICE_B_ALDUCHOV, SVP_ICE_C_ALDUCHOV,
)
from .core import FloatArray
from .exceptions import InvalidUnitError, OutOfRangeError, ValidationError

_VALID_UNITS = {"C", "F", "K"}

# Magnus-form saturation vapor pressure coefficients (A [kPa], B, C [°C]) by
# method, for liquid water and for ice.
_SVP_COEFFICIENTS = {
    "tetens": (SVP_A, SVP_B, SVP_C),
    "alduchov": (SVP_A_ALDUCHOV, SVP_B_ALDUCHOV, SVP_C_ALDUCHOV),
}
_SVP_ICE_COEFFICIENTS = {
    "tetens": (SVP_ICE_A, SVP_ICE_B, SVP_ICE_C),
    "alduchov": (SVP_ICE_A_ALDUCHOV, SVP_ICE_B_ALDUCHOV, SVP_ICE_C_ALDUCHOV),
}


def _svp_coefficients(method: str, ice: bool = False) -> tuple[float, float, float]:
    """Return Magnus-form (A, B, C) coefficients for a saturation vapor pressure method."""
    table = _SVP_ICE_COEFFICIENTS if ice else _SVP_COEFFICIENTS

    if not isinstance(method, str) or method.lower() not in table:
        raise ValidationError(
            f"Saturation vapor pressure method must be one of {sorted(table)}; "
            f"got {method!r}."
        )

    coefficients = table[method.lower()]

    return coefficients


def convert_temperature(temp: FloatArray, input_unit: str, output_unit: str) -> FloatArray:
    """
    Convert temperature between Celsius, Fahrenheit, and Kelvin.
    
    Parameters
    ----------
    temp : Temperature value (scalar or array-like).
    input_unit : Unit of input temperature: "C", "F", or "K".
    output_unit : Unit of output temperature: "C", "F", or "K".
    
    Returns
    -------
    Temperature in the specified output unit.
    
    Raises
    ------
    ValidationError
        If temp is not numeric.
    InvalidUnitError
        If input_unit or output_unit is not "C", "F", or "K".

    Examples
    --------
    >>> convert_temperature(100, "C", "F")
    212.0
    >>> convert_temperature(273.15, "K", "C")
    0.0
    >>> a, b, c = 80, "F", "C"
    >>> temp = convert_temperature(a, b, c)
    >>> print(f"{a}°{b} is equal to {round(temp, 2)}°{c}.")
    80°F is equal to 26.67°C.
    """
    # Check temp is numeric (scalars or numeric arrays)
    if not isinstance(temp, (int, float, np.number, np.ndarray, list, tuple)):
        raise ValidationError("Temperature must be numeric.")
    if isinstance(temp, (list, tuple)):
        temp = np.asarray(temp, dtype=float)
    if isinstance(temp, np.ndarray) and not np.issubdtype(temp.dtype, np.number):
        raise ValidationError("Temperature must be numeric.")
    
    # Normalize units to uppercase
    input_unit = input_unit.upper()
    output_unit = output_unit.upper()
    
    # Validate units
    if input_unit not in _VALID_UNITS:
        raise InvalidUnitError(f"Input unit must be one of {_VALID_UNITS}")
    if output_unit not in _VALID_UNITS:
        raise InvalidUnitError(f"Output unit must be one of {_VALID_UNITS}")
    
    # If units are the same, return the temperature as-is
    if input_unit == output_unit:
        return temp
    
    if input_unit == "C":
        if output_unit == "F":
            return temp * 9 / 5 + 32
        else:  # output_unit == "K"
            return temp + 273.15
    
    elif input_unit == "F":
        if output_unit == "C":
            return (temp - 32) * 5 / 9
        else:  # output_unit == "K"
            return (temp - 32) * 5 / 9 + 273.15
    
    else:  # input_unit == "K"
        if output_unit == "C":
            return temp - 273.15
        else:  # output_unit == "F"
            return (temp - 273.15) * 9 / 5 + 32
        

def dewpoint_temperature(
    temp: FloatArray, rh: FloatArray, unit: str = "C", method: str = "tetens"
) -> FloatArray:
    """
    Calculate dew point temperature from temperature and relative humidity.

    Defaults to the Magnus formula, which is accurate for normal atmospheric conditions.
    User switch to the Alduchov & Eskridge formula, which is more accurate at extreme values.

    Parameters
    ----------
    temp : Air temperature.
    rh : Relative humidity (%).
    unit : Unit of input/output temperature: "C", "F", or "K" (default is "C").
    method : Magnus coefficient set: "tetens" (default) or "alduchov".

    Returns
    -------
    Dew point temperature in the same unit as input.

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
    Uses the Magnus approximation:
    Td = (b * alpha) / (a - alpha)
    where alpha = ln(RH/100) + (a*T)/(b+T)
    Constants: a = 17.27, b = 237.3°C (Tetens) or
    a = 17.625, b = 243.04°C (Alduchov & Eskridge).

    Valid for:
    - Temperature range: -40°C to 50°C
    - Relative humidity: 1% to 100%

    Examples
    --------
    >>> print(round(dewpoint_temperature(30, 50), 2))
    18.44
    >>> print(round(dewpoint_temperature(86, 50, unit="F"), 2))
    65.19
    >>> print(round(dewpoint_temperature(30, 50, method="alduchov"), 2))
    18.45
    """
    # Validate relative humidity
    if np.any(rh <= 0) or np.any(rh > 100):
        raise OutOfRangeError("Relative humidity must be between 0 and 100%")

    # Magnus formula constants
    _, a, b = _svp_coefficients(method)

    # Convert temperature to Celsius for calculation
    unit = unit.upper()
    if unit != "C":
        temp_C = convert_temperature(temp, unit, "C")
    else:
        temp_C = temp

    # Calculate alpha
    alpha = (a * temp_C) / (b + temp_C) + np.log(rh / 100.0)
    
    # Calculate dew point in Celsius
    Td_C = (b * alpha) / (a - alpha)
    
    # Convert back to original unit if necessary
    if unit != "C":
        return convert_temperature(Td_C, "C", unit)
    else:
        return Td_C


def dewpoint_from_avp(avp: FloatArray, unit: str = "C", method: str = "tetens") -> FloatArray:
    """
    Calculate dew point temperature from actual vapor pressure.

    Parameters
    ----------
    avp : Actual vapor pressure (kPa).
    unit: Unit of output temperature: "C", "F", or "K" (default is "C").
    method : Magnus coefficient set: "tetens" (default) or "alduchov".

    Returns
    -------
    Dew point temperature in specified unit

    Raises
    ------
    OutOfRangeError
        If avp is not strictly positive.
    InvalidUnitError
        If unit is invalid.
    ValidationError
        If method is not "tetens" or "alduchov".

    Notes
    -----
    Inverts the Magnus saturation vapor pressure formula:
    Td = (C * ln(e / A)) / (B - ln(e / A))
    with e in kPa and the (A, B, C) coefficients of the chosen method.

    Examples
    --------
    >>> print(round(dewpoint_from_avp(2.338), 2))
    20.0
    >>> print(round(dewpoint_from_avp(2.338, method="alduchov"), 2))
    20.03
    """
    if np.any(avp <= 0):
        raise OutOfRangeError("Actual vapor pressure must be positive.")

    # Invert the Magnus saturation vapor pressure formula
    a, b, c = _svp_coefficients(method)
    ln_ratio = np.log(avp / a)
    Td_C = (c * ln_ratio) / (b - ln_ratio)
    
    # Convert to desired unit if necessary
    unit = unit.upper()
    if unit != "C":
        return convert_temperature(Td_C, "C", unit)
    else:
        return Td_C


def potential_temperature(
    temp: FloatArray,
    pressure: FloatArray,
    reference_pressure: FloatArray = P0,
    unit: str = "K",
) -> FloatArray:
    """
    Calculate potential temperature of dry air.

    Potential temperature is the temperature a parcel of dry air would have if
    brought adiabatically to a reference pressure (sea level by default). It is
    conserved under dry adiabatic processes and is widely used to compare air at
    different altitudes.

    Parameters
    ----------
    temp : Air temperature.
    pressure : Ambient pressure (kPa).
    reference_pressure : Reference pressure (kPa), default sea-level P0.
    unit : Unit of input/output temperature: "C", "F", or "K" (default "K").

    Returns
    -------
    Potential temperature in the same unit as the input temperature.

    Raises
    ------
    OutOfRangeError
        If pressure or reference_pressure is not strictly positive.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Poisson's equation for dry air:
    theta = T * (P0 / P) ** (R_air / cp)

    The exponent R_air / cp ≈ 0.286 for dry air. The calculation is performed in
    Kelvin and converted back to the requested unit.

    Examples
    --------
    >>> print(round(potential_temperature(273.15, 80.0), 2))
    292.22
    >>> print(round(potential_temperature(0.0, 80.0, unit="C"), 2))
    19.07
    """
    if np.any(pressure <= 0) or np.any(reference_pressure <= 0):
        raise OutOfRangeError("Pressure values must be positive.")

    temp_K = convert_temperature(temp, unit.upper(), "K")
    theta_K = temp_K * (reference_pressure / pressure) ** (R_AIR / CP_AIR)
    theta = convert_temperature(theta_K, "K", unit.upper())

    return theta


def virtual_temperature(temp: FloatArray, mixing_ratio: FloatArray, unit: str = "K") -> FloatArray:
    """
    Calculate virtual temperature of moist air.

    Virtual temperature is the temperature dry air would need to have the same
    density as a given sample of moist air at the same pressure. It accounts for
    the lower density of water vapor relative to dry air.

    Parameters
    ----------
    temp : Air temperature.
    mixing_ratio : Water vapor mixing ratio (kg/kg).
    unit : Unit of input/output temperature: "C", "F", or "K" (default "K").

    Returns
    -------
    Virtual temperature in the same unit as the input temperature.

    Raises
    ------
    OutOfRangeError
        If mixing_ratio is negative.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Approximation:
    Tv = T * (1 + 0.61 * w)
    where w is the mixing ratio in kg/kg and T is in Kelvin. The coefficient
    0.61 ≈ (1 - RMW) / RMW.

    Examples
    --------
    >>> round(virtual_temperature(300.0, 0.01), 3)
    301.83
    """
    if np.any(mixing_ratio < 0):
        raise OutOfRangeError("Mixing ratio must be non-negative.")

    temp_K = convert_temperature(temp, unit.upper(), "K")
    tv_K = temp_K * (1 + 0.61 * mixing_ratio)
    tv = convert_temperature(tv_K, "K", unit.upper())

    return tv


def lapse_rate(
    temp_lower: FloatArray,
    temp_upper: FloatArray,
    height_lower: FloatArray,
    height_upper: FloatArray,
    unit: str = "C",
) -> FloatArray:
    """
    Calculate the environmental lapse rate between two heights.

    The environmental lapse rate is the rate of temperature decrease with
    height. It is positive when temperature decreases with altitude (the usual
    tropospheric case) and negative under a temperature inversion.

    Parameters
    ----------
    temp_lower : Temperature at the lower height.
    temp_upper : Temperature at the upper height.
    height_lower : Lower height (m).
    height_upper : Upper height (m), must differ from height_lower.
    unit : Unit of the input temperatures: "C", "F", or "K" (default "C").

    Returns
    -------
    Lapse rate in degrees (of the input unit, or K for "K") per meter. Multiply
    by 1000 to express the result per kilometer.

    Raises
    ------
    OutOfRangeError
        If the two heights are equal.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Gamma = -(T_upper - T_lower) / (z_upper - z_lower)

    Temperatures are converted to Kelvin for the difference so that the
    magnitude is consistent for "C" and "K" inputs; the returned value carries
    the same per-degree size as Celsius/Kelvin. For Fahrenheit inputs the result
    is reported in K/m.

    Examples
    --------
    >>> round(lapse_rate(15.0, 8.5, 0.0, 1000.0) * 1000, 1)
    6.5
    """
    if np.any(height_upper == height_lower):
        raise OutOfRangeError("Upper and lower heights must differ.")

    t_lower_K = convert_temperature(temp_lower, unit.upper(), "K")
    t_upper_K = convert_temperature(temp_upper, unit.upper(), "K")
    gamma = -(t_upper_K - t_lower_K) / (height_upper - height_lower)

    return gamma


def surface_temperature_from_lw(
    lw_up: FloatArray, emissivity: FloatArray = 1.0, unit: str = "K"
) -> FloatArray:
    """
    Retrieve surface temperature from upwelling longwave radiation.

    Inverts the Stefan-Boltzmann law to estimate the radiating (skin)
    temperature of a surface given its upwelling longwave flux and emissivity.

    Parameters
    ----------
    lw_up : Upwelling longwave radiative flux (W/m²), must be positive.
    emissivity : Surface emissivity in (0, 1] (default 1.0 for a blackbody).
    unit : Unit of output temperature: "C", "F", or "K" (default "K").

    Returns
    -------
    Surface temperature in the requested unit.

    Raises
    ------
    OutOfRangeError
        If lw_up is not positive or emissivity is outside (0, 1].
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Inverse Stefan-Boltzmann law:
    Ts = (LW_up / (emissivity * sigma)) ** 0.25
    with sigma the Stefan-Boltzmann constant. Ts is computed in Kelvin and
    converted to the requested unit.

    Examples
    --------
    >>> print(round(surface_temperature_from_lw(390.0), 2))
    287.98
    >>> print(round(surface_temperature_from_lw(390.0, unit="C"), 2))
    14.83
    """
    if np.any(lw_up <= 0):
        raise OutOfRangeError("Upwelling longwave flux must be positive.")
    if np.any(emissivity <= 0) or np.any(emissivity > 1):
        raise OutOfRangeError("Emissivity must be in the interval (0, 1].")

    ts_K = (lw_up / (emissivity * STEFAN_BOLTZMANN)) ** 0.25
    ts = convert_temperature(ts_K, "K", unit.upper())

    return ts


def wet_bulb_temperature(temp: FloatArray, rh: FloatArray, unit: str = "C") -> FloatArray:
    """
    Wet-bulb temperature using Stull's empirical approximation.

    The wet-bulb temperature is the lowest temperature air can reach by
    evaporative cooling at constant pressure. Stull's relation is accurate to
    within a few tenths of a degree for typical surface conditions.

    Parameters
    ----------
    temp : Air temperature.
    rh : Relative humidity (%), in the interval (0, 100].
    unit : Unit of input/output temperature: "C", "F", or "K" (default "C").

    Returns
    -------
    Wet-bulb temperature in the same unit as the input.

    Raises
    ------
    OutOfRangeError
        If rh is not in the interval (0, 100].
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Stull (2011), valid near standard sea-level pressure:
    Tw = T*atan(0.151977*(RH+8.313659)**0.5) + atan(T+RH)
         - atan(RH-1.676331) + 0.00391838*RH**1.5*atan(0.023101*RH)
         - 4.686035
    with T in °C and RH in percent.

    Examples
    --------
    >>> print(round(wet_bulb_temperature(25, 50), 2))
    18.0
    """
    if np.any(rh <= 0) or np.any(rh > 100):
        raise OutOfRangeError("Relative humidity must be between 0 and 100%")
    
    temp_C = convert_temperature(temp, unit.upper(), "C")
    tw = (
        temp_C * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
        + np.arctan(temp_C + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )

    if unit.upper() != "C":
        return convert_temperature(tw, "C", unit.upper())
    
    return tw


def equivalent_potential_temperature(
    temp: FloatArray, mixing_ratio: FloatArray, pressure: FloatArray, unit: str = "K"
) -> FloatArray:
    """
    Approximate equivalent potential temperature of moist air.

    Equivalent potential temperature is the potential temperature a parcel would
    attain if all its water vapor condensed and the released latent heat warmed
    it. It is conserved under both dry and moist adiabatic processes.

    Parameters
    ----------
    temp : Air temperature.
    mixing_ratio : Water vapor mixing ratio (kg/kg), non-negative.
    pressure : Ambient pressure (kPa), must be positive.
    unit : Unit of input/output temperature: "C", "F", or "K" (default "K").

    Returns
    -------
    Equivalent potential temperature in the same unit as the input.

    Raises
    ------
    OutOfRangeError
        If mixing_ratio is negative or pressure is not positive.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Bolton-style approximation using the dry potential temperature theta:
    theta_e = theta * exp(Lv * w / (cp * T))
    with T in kelvin and w the mixing ratio in kg/kg.

    Examples
    --------
    >>> print(round(equivalent_potential_temperature(290.0, 0.01, 90.0), 2))
    326.29
    """
    if np.any(mixing_ratio < 0):
        raise OutOfRangeError("Mixing ratio must be non-negative.")
    if np.any(pressure <= 0):
        raise OutOfRangeError("Pressure must be positive.")
    
    temp_K = convert_temperature(temp, unit.upper(), "K")
    theta = potential_temperature(temp_K, pressure, unit="K")
    theta_e = theta * np.exp(LV * mixing_ratio / (CP_AIR * temp_K))
    theta_e_T = convert_temperature(theta_e, "K", unit.upper())

    return theta_e_T


def moist_adiabatic_lapse_rate(temp: FloatArray, mixing_ratio: FloatArray, unit: str = "C") -> FloatArray:
    """
    Saturated (moist) adiabatic lapse rate.

    The rate of temperature decrease with height for a saturated parcel rising
    adiabatically. It is smaller than the dry rate because latent heat release
    partially offsets adiabatic cooling.

    Parameters
    ----------
    temp : Air temperature.
    mixing_ratio : Saturation mixing ratio (kg/kg), non-negative.
    unit : Unit of temp: "C", "F", or "K" (default "C").

    Returns
    -------
    Moist adiabatic lapse rate in K/m.

    Raises
    ------
    OutOfRangeError
        If mixing_ratio is negative.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Standard expression:
    Gamma_m = g * (1 + Lv*w / (R_air*T)) / (cp + Lv^2*w / (R_vapor*T^2))
    Here R_vapor is approximated through the molecular-weight ratio embedded in
    the latent-heat term; valid for typical tropospheric conditions.

    Examples
    --------
    >>> print(round(moist_adiabatic_lapse_rate(20, 0.015) * 1000, 3))
    4.302
    """
    if np.any(mixing_ratio < 0):
        raise OutOfRangeError("Mixing ratio must be non-negative.")
    
    temp_K = convert_temperature(temp, unit.upper(), "K")
    r_vapor = 461.5
    numerator = G * (1 + LV * mixing_ratio / (R_AIR * temp_K))
    denominator = CP_AIR + (LV**2 * mixing_ratio) / (r_vapor * temp_K**2)
    gamma_m = numerator / denominator
    
    return gamma_m