"""
atmoflux.hydro
=================
Calculates water fluxes within the hydrological cycle. 
Includes conversion of latent heat flux to evaporation depth, open-water Penman
evaporation, the general Penman-Monteith equation, FAO-56 reference
evapotranspiration, equilibrium and Priestley-Taylor evaporation, Hargreaves
reference evapotranspiration, and Dalton mass-transfer evaporation.

"""

from __future__ import annotations

# Outside imports
import numpy as np

# imports from within atmoflux
from .constants import CP_AIR, LV, RMW, RHO_WATER, P0, PRIESTLEY_TAYLOR_ALPHA
from .core import FloatArray
from .exceptions import OutOfRangeError
from .humidity import saturation_vp_slope
from .temperature import convert_temperature

# Seconds per day, used to express fluxes as daily evaporation depths.
_SECONDS_PER_DAY = 86400.0


def latent_heat_to_evaporation(
    latent_heat: FloatArray, density_water: FloatArray = RHO_WATER
) -> FloatArray:
    """
    Convert latent heat flux to an equivalent evaporation depth.

    Parameters
    ----------
    latent_heat : Latent heat flux LE (W/m²).
    density_water : Density of liquid water (kg/m³), default from constants.

    Returns
    -------
    Evaporation rate in mm/day.

    Raises
    ------
    OutOfRangeError
        If density_water is not positive.

    Notes
    -----
    E = LE / (Lv * rho_water) converted from m/s to mm/day:
    E[mm/day] = LE / (Lv * rho_water) * 1000 * 86400

    Examples
    --------
    >>> print(round(latent_heat_to_evaporation(100.0), 3))
    3.527
    """
    if np.any(density_water <= 0):
        raise OutOfRangeError("Water density must be positive.")
    
    e_m_per_s = latent_heat / (LV * density_water)
    e = e_m_per_s * 1000.0 * _SECONDS_PER_DAY

    return e


def _psychrometric_constant(pressure: FloatArray) -> FloatArray:
    """
    Psychrometric constant (kPa/°C) at a given pressure (kPa).
    For more precise calculation when pressure is known.
    """
    gamma = CP_AIR * pressure / (RMW * LV)

    return gamma


def penman_evaporation(
    net_radiation: FloatArray,
    ground_heat: FloatArray,
    temp: FloatArray,
    wind_2m: FloatArray,
    es: FloatArray,
    ea: FloatArray,
    pressure: FloatArray = P0,
    unit: str = "C",
    method: str = "tetens",
) -> FloatArray:
    """
    Open-water evaporation using the Penman combination equation.

    Combines a radiation (energy) term and an aerodynamic (mass-transfer) term to
    estimate evaporation from an open water surface.

    Parameters
    ----------
    net_radiation : Net radiation Rn (MJ/m²/day).
    ground_heat : Ground/water heat flux G (MJ/m²/day).
    temp : Mean air temperature.
    wind_2m : Wind speed at 2 m (m/s), must be non-negative.
    es : Saturation vapor pressure at air temperature (kPa).
    ea : Actual vapor pressure (kPa).
    pressure : Atmospheric pressure (kPa), default sea-level P0.
    unit : Unit of temp: "C", "F", or "K" (default "C").
    method : Saturation vapor pressure formula for the slope term: "tetens"
        (default) or "alduchov".

    Returns
    -------
    Open-water evaporation in mm/day.

    Raises
    ------
    OutOfRangeError
        If wind_2m is negative.
    InvalidUnitError
        If unit is invalid.
    ValidationError
        If method is not "tetens" or "alduchov".

    Notes
    -----
    ET = [delta * (Rn - G) + gamma * Ea] / [lambda * (delta + gamma)]
    with the aerodynamic term Ea = 6.43 * (1 + 0.536 * u2) * (es - ea) in
    MJ/m²/day and lambda the latent heat of vaporization (MJ/kg). delta is the
    slope of the saturation vapor pressure curve and gamma the psychrometric
    constant, both in kPa/°C.

    Examples
    --------
    >>> print(round(penman_evaporation(15.0, 0.0, 25.0, 2.0, 3.169, 1.9, 101.3), 3))
    6.326
    """
    if np.any(wind_2m < 0):
        raise OutOfRangeError("Wind speed must be non-negative.")

    temp_C = convert_temperature(temp, unit.upper(), "C")
    delta = saturation_vp_slope(temp_C, "C", method)
    gamma = _psychrometric_constant(pressure)
    lam = LV / 1.0e6  # latent heat in MJ/kg

    aerodynamic = 6.43 * (1 + 0.536 * wind_2m) * (es - ea)
    numerator = delta * (net_radiation - ground_heat) + gamma * aerodynamic
    et = numerator / (lam * (delta + gamma))

    return et


def penman_monteith(
    net_radiation: FloatArray,
    ground_heat: FloatArray,
    temp: FloatArray,
    vpd: FloatArray,
    density: FloatArray,
    resistance_aero: FloatArray,
    resistance_surface: FloatArray,
    pressure: FloatArray = P0,
    unit: str = "C",
    method: str = "tetens",
) -> FloatArray:
    """
    Evaporation from the general Penman-Monteith equation.

    The Penman-Monteith equation combines available energy and an aerodynamic
    vapor term with explicit aerodynamic and surface (canopy) resistances.

    Parameters
    ----------
    net_radiation : Net radiation Rn (W/m²).
    ground_heat : Ground heat flux G (W/m²).
    temp : Mean air temperature.
    vpd : Vapor pressure deficit (kPa), must be non-negative.
    density : Air density (kg/m³).
    resistance_aero : Aerodynamic resistance ra (s/m), must be positive.
    resistance_surface : Surface/canopy resistance rs (s/m), must be non-negative.
    pressure : Atmospheric pressure (kPa), default sea-level P0.
    unit : Unit of temp: "C", "F", or "K" (default "C").
    method : Saturation vapor pressure formula for the slope term: "tetens"
        (default) or "alduchov".

    Returns
    -------
    Evaporation in mm/day.

    Raises
    ------
    OutOfRangeError
        If vpd is negative, resistance_aero is not positive, or
        resistance_surface is negative.
    InvalidUnitError
        If unit is invalid.
    ValidationError
        If method is not "tetens" or "alduchov".

    Notes
    -----
    Latent heat flux (W/m²):
    lambda_E = [delta * A + rho * cp * D / ra] / [delta + gamma * (1 + rs / ra)]
    with available energy A = Rn - G (W/m²), D the vapor pressure deficit (Pa),
    and delta and gamma expressed in Pa/K. The flux is converted to mm/day.

    Examples
    --------
    >>> print(round(penman_monteith(150.0, 20.0, 25.0, 1.5, 1.2, 50.0, 70.0), 3))
    6.133
    """
    if np.any(vpd < 0):
        raise OutOfRangeError("Vapor pressure deficit must be non-negative.")
    if np.any(resistance_aero <= 0):
        raise OutOfRangeError("Aerodynamic resistance must be positive.")
    if np.any(resistance_surface < 0):
        raise OutOfRangeError("Surface resistance must be non-negative.")

    temp_C = convert_temperature(temp, unit.upper(), "C")
    delta = saturation_vp_slope(temp_C, "C", method) * 1000.0  # kPa/K -> Pa/K
    gamma = _psychrometric_constant(pressure) * 1000.0  # kPa/K -> Pa/K
    vpd_pa = vpd * 1000.0  # kPa -> Pa
    available = net_radiation - ground_heat

    numerator = delta * available + density * CP_AIR * vpd_pa / resistance_aero
    denominator = delta + gamma * (1 + resistance_surface / resistance_aero)
    latent_heat = numerator / denominator
    lambda_E = latent_heat_to_evaporation(latent_heat)

    return lambda_E


def potential_evapotranspiration(
    net_radiation: FloatArray,
    ground_heat: FloatArray,
    temp: FloatArray,
    wind_2m: FloatArray,
    es: FloatArray,
    ea: FloatArray,
    pressure: FloatArray = P0,
    unit: str = "C",
) -> FloatArray:
    """
    Reference evapotranspiration from the FAO-56 Penman-Monteith equation.

    Computes daily reference evapotranspiration (ET0) for a hypothetical
    well-watered grass surface using the standardized FAO-56 form.

    Parameters
    ----------
    net_radiation : Net radiation Rn (MJ/m²/day).
    ground_heat : Soil heat flux G (MJ/m²/day).
    temp : Mean daily air temperature.
    wind_2m : Wind speed at 2 m (m/s), must be non-negative.
    es : Saturation vapor pressure at mean temperature (kPa).
    ea : Actual vapor pressure (kPa).
    pressure : Atmospheric pressure (kPa), default sea-level P0.
    unit : Unit of temp: "C", "F", or "K" (default "C").

    Returns
    -------
    Reference evapotranspiration ET0 in mm/day.

    Raises
    ------
    OutOfRangeError
        If wind_2m is negative.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    FAO-56 reference equation:
    ET0 = [0.408 * delta * (Rn - G) + gamma * (900 / (T + 273)) * u2 * (es - ea)]
          / [delta + gamma * (1 + 0.34 * u2)]
    with T in °C, delta and gamma in kPa/°C, and vapor pressures in kPa.

    Examples
    --------
    >>> print(round(potential_evapotranspiration(15.0, 0.0, 25.0, 2.0, 3.169, 1.9, 101.3), 3))
    5.539
    """
    if np.any(wind_2m < 0):
        raise OutOfRangeError("Wind speed must be non-negative.")

    temp_C = convert_temperature(temp, unit.upper(), "C")
    delta = saturation_vp_slope(temp_C, "C")
    gamma = _psychrometric_constant(pressure)

    radiation_term = 0.408 * delta * (net_radiation - ground_heat)
    aero_term = gamma * (900.0 / (temp_C + 273.0)) * wind_2m * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * wind_2m)
    et0 = (radiation_term + aero_term) / denominator

    return et0


def equilibrium_evaporation(
    net_radiation: FloatArray,
    ground_heat: FloatArray,
    temp: FloatArray,
    pressure: FloatArray = P0,
    unit: str = "C",
    method: str = "tetens",
) -> FloatArray:
    """
    Equilibrium evaporation from available energy.

    Equilibrium evaporation is the evaporation a saturated surface would sustain
    under conditions of minimal advection, depending only on available energy and
    the temperature-dependent partitioning between sensible and latent heat.

    Parameters
    ----------
    net_radiation : Net radiation Rn (MJ/m²/day).
    ground_heat : Ground heat flux G (MJ/m²/day).
    temp : Mean air temperature.
    pressure : Atmospheric pressure (kPa), default sea-level P0.
    unit : Unit of temp: "C", "F", or "K" (default "C").
    method : Saturation vapor pressure formula for the slope term: "tetens"
        (default) or "alduchov".

    Returns
    -------
    Equilibrium evaporation in mm/day.

    Raises
    ------
    InvalidUnitError
        If unit is invalid.
    ValidationError
        If method is not "tetens" or "alduchov".

    Notes
    -----
    E_eq = [delta / (delta + gamma)] * (Rn - G) / lambda
    with delta the slope of the saturation vapor pressure curve, gamma the
    psychrometric constant, and lambda the latent heat of vaporization (MJ/kg).

    Examples
    --------
    >>> print(round(equilibrium_evaporation(15.0, 0.0, 25.0), 3))
    4.521
    """
    temp_C = convert_temperature(temp, unit.upper(), "C")
    delta = saturation_vp_slope(temp_C, "C", method)
    gamma = _psychrometric_constant(pressure)
    lam = LV / 1.0e6  # latent heat in MJ/kg
    E_eq = (delta / (delta + gamma)) * (net_radiation - ground_heat) / lam

    return E_eq


def priestley_taylor(
    net_radiation: FloatArray,
    ground_heat: FloatArray,
    temp: FloatArray,
    alpha: FloatArray = PRIESTLEY_TAYLOR_ALPHA,
    pressure: FloatArray = P0,
    unit: str = "C",
    method: str = "tetens",
) -> FloatArray:
    """
    Priestley-Taylor evaporation.

    Scales equilibrium evaporation by an empirical coefficient to estimate
    actual evaporation from well-watered surfaces under minimal advection.

    Parameters
    ----------
    net_radiation : Net radiation Rn (MJ/m²/day).
    ground_heat : Ground heat flux G (MJ/m²/day).
    temp : Mean air temperature.
    alpha : Priestley-Taylor coefficient (default from constants, ~1.26).
    pressure : Atmospheric pressure (kPa), default sea-level P0.
    unit : Unit of temp: "C", "F", or "K" (default "C").
    method : Saturation vapor pressure formula for the slope term: "tetens"
        (default) or "alduchov".

    Returns
    -------
    Priestley-Taylor evaporation in mm/day.

    Raises
    ------
    OutOfRangeError
        If alpha is not positive.
    InvalidUnitError
        If unit is invalid.
    ValidationError
        If method is not "tetens" or "alduchov".

    Notes
    -----
    E_pt = alpha * E_eq
    where E_eq is the equilibrium evaporation.

    Examples
    --------
    >>> print(round(priestley_taylor(15.0, 0.0, 25.0), 3))
    5.697
    """
    if np.any(alpha <= 0):
        raise OutOfRangeError("Priestley-Taylor coefficient must be positive.")
    
    E_eq = equilibrium_evaporation(
        net_radiation, ground_heat, temp, pressure, unit, method
    )
    E_pt = alpha * E_eq

    return E_pt


def hargreaves(
    temp_mean: FloatArray,
    temp_min: FloatArray,
    temp_max: FloatArray,
    extraterrestrial: FloatArray,
    unit: str = "C",
) -> FloatArray:
    """
    Hargreaves reference evapotranspiration.

    A temperature-based estimate of reference evapotranspiration, useful when
    humidity, wind, and radiation measurements are unavailable. Only air
    temperature and extraterrestrial radiation are required.

    Parameters
    ----------
    temp_mean : Mean daily air temperature.
    temp_min : Minimum daily air temperature.
    temp_max : Maximum daily air temperature (>= temp_min).
    extraterrestrial : Extraterrestrial radiation Ra (MJ/m²/day).
    unit : Unit of the input temperatures: "C", "F", or "K" (default "C").

    Returns
    -------
    Reference evapotranspiration in mm/day.

    Raises
    ------
    OutOfRangeError
        If temp_max is less than temp_min.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Hargreaves-Samani:
    ET0 = 0.0023 * (T_mean + 17.8) * (T_max - T_min) ** 0.5 * Ra / lambda
    with temperatures in °C, Ra in MJ/m²/day, and lambda = 2.45 MJ/kg so the
    0.408 radiation-to-depth factor is applied internally.

    Examples
    --------
    >>> print(round(hargreaves(25.0, 18.0, 32.0, 36.0), 3))
    5.41
    """
    t_mean_C = convert_temperature(temp_mean, unit.upper(), "C")
    t_min_C = convert_temperature(temp_min, unit.upper(), "C")
    t_max_C = convert_temperature(temp_max, unit.upper(), "C")

    if np.any(t_max_C < t_min_C):
        raise OutOfRangeError("Maximum temperature must be at least the minimum.")
    
    et0 = (
        0.0023
        * (t_mean_C + 17.8)
        * (t_max_C - t_min_C) ** 0.5
        * 0.408
        * extraterrestrial
    )

    return et0


def dalton_evaporation(vpd: FloatArray, transfer_coeff: FloatArray) -> FloatArray:
    """
    Mass-transfer (Dalton-type) evaporation from the vapor pressure deficit.

    The simplest aerodynamic evaporation model: evaporation is proportional to
    the drying power of the air, expressed as the vapor pressure deficit, with
    an empirical transfer coefficient that lumps together wind and surface
    effects.

    Parameters
    ----------
    vpd : Vapor pressure deficit (kPa). Negative values (supersaturation) give
        negative evaporation, i.e. condensation.
    transfer_coeff : Empirical transfer coefficient K (mm/day/kPa), non-negative.

    Returns
    -------
    Evaporation rate in mm/day; negative values indicate condensation.

    Raises
    ------
    OutOfRangeError
        If transfer_coeff is negative.

    Notes
    -----
    Dalton's law:
    E = K * VPD
    K is site- and wind-dependent and must be calibrated or taken from the
    literature; it often takes the form K = a + b * u for wind speed u.

    Examples
    --------
    >>> print(round(dalton_evaporation(1.2671, 3.0), 3))
    3.801
    """
    if np.any(transfer_coeff < 0):
        raise OutOfRangeError("Transfer coefficient must be non-negative.")

    e = transfer_coeff * vpd

    return e
