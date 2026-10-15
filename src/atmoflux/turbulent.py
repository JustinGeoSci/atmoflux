"""
atmoflux.turbulent
=================
Calculates turbulent fluxes driven by wind and surface-air gradients.
Includes dry and moist air density, bulk-aerodynamic sensible and latent heat
fluxes, neutral bulk transfer coefficients, surface shear stress, and
aerodynamic resistance.

"""

from __future__ import annotations

# Outside imports
import numpy as np

# imports from within atmoflux
from .constants import CP_AIR, LV, R_AIR, KARMAN
from .core import FloatArray
from .exceptions import OutOfRangeError
from .temperature import convert_temperature, virtual_temperature


def air_density(temp: FloatArray, pressure: FloatArray, unit: str = "C") -> FloatArray:
    """
    Density of dry air from temperature and pressure (ideal gas law).

    Parameters
    ----------
    temp : Air temperature.
    pressure : Air pressure (kPa), must be positive.
    unit : Unit of temp: "C", "F", or "K" (default "C").

    Returns
    -------
    Air density in kg/m³.

    Raises
    ------
    OutOfRangeError
        If pressure is not positive.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Ideal gas law for dry air:
    rho = P / (R_air * T)
    with P in pascals and T in kelvin.

    Examples
    --------
    >>> print(round(air_density(15, 101.325), 4))
    1.225
    """
    if np.any(pressure <= 0):
        raise OutOfRangeError("Pressure must be positive.")
    
    temp_K = convert_temperature(temp, unit.upper(), "K")
    p_pa = pressure * 1000.0  # kPa -> Pa
    rho = p_pa / (R_AIR * temp_K)

    return rho


def sensible_heat_flux(
    density: FloatArray,
    wind_speed: FloatArray,
    temp_air: FloatArray,
    temp_surface: FloatArray,
    transfer_coeff: FloatArray,
    unit: str = "C",
) -> FloatArray:
    """
    Bulk-aerodynamic sensible heat flux.

    Parameters
    ----------
    density : Air density (kg/m³).
    wind_speed : Wind speed at the reference height (m/s).
    temp_air : Air temperature at the reference height.
    temp_surface : Surface (skin) temperature.
    transfer_coeff : Bulk transfer coefficient for heat (dimensionless).
    unit : Unit of the input temperatures: "C", "F", or "K" (default "C").

    Returns
    -------
    Sensible heat flux H (W/m²), positive when directed from surface to air.

    Raises
    ------
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Bulk-aerodynamic form:
    H = rho * cp * Ch * U * (Ts - Ta)
    Temperatures are converted to kelvin so the difference is independent of the
    input scale.

    Examples
    --------
    >>> print(round(sensible_heat_flux(1.2, 3.0, 20.0, 25.0, 0.0013), 4))
    23.517
    """
    ts_K = convert_temperature(temp_surface, unit.upper(), "K")
    ta_K = convert_temperature(temp_air, unit.upper(), "K")
    h = density * CP_AIR * transfer_coeff * wind_speed * (ts_K - ta_K)

    return h


def latent_heat_flux(
    density: FloatArray,
    wind_speed: FloatArray,
    q_air: FloatArray,
    q_surface: FloatArray,
    transfer_coeff: FloatArray,
) -> FloatArray:
    """
    Bulk-aerodynamic latent heat flux.

    Parameters
    ----------
    density : Air density (kg/m³).
    wind_speed : Wind speed at the reference height (m/s).
    q_air : Specific humidity of the air (kg/kg).
    q_surface : Specific humidity at the surface (kg/kg).
    transfer_coeff : Bulk transfer coefficient for moisture (dimensionless).

    Returns
    -------
    Latent heat flux LE (W/m²), positive for evaporation (surface to air).

    Notes
    -----
    Bulk-aerodynamic form:
    LE = rho * Lv * Ce * U * (q_s - q_a)

    Examples
    --------
    >>> print(round(latent_heat_flux(1.2, 3.0, 0.008, 0.012, 0.0013), 4))
    45.864
    """
    le = density * LV * transfer_coeff * wind_speed * (q_surface - q_air)

    return le


def bulk_transfer_coefficient(
    height: FloatArray, roughness: FloatArray, displacement: FloatArray = 0.0, karman: FloatArray = KARMAN
) -> FloatArray:
    """
    Neutral bulk transfer coefficient from surface-layer geometry.

    Parameters
    ----------
    height : Reference height (m), above displacement + roughness.
    roughness : Aerodynamic roughness length z0 (m), must be positive.
    displacement : Zero-plane displacement height d (m), default 0.0.
    karman : von Kármán constant (default from constants).

    Returns
    -------
    Neutral bulk transfer coefficient (dimensionless).

    Raises
    ------
    OutOfRangeError
        If roughness is not positive or height is not above d + z0.

    Notes
    -----
    Assumes neutral stability and equal momentum and scalar roughness lengths:
    C = k ** 2 / ln((z - d) / z0) ** 2

    Examples
    --------
    >>> print(round(bulk_transfer_coefficient(10.0, 0.03), 6))
    0.004741
    """
    if np.any(roughness <= 0):
        raise OutOfRangeError("Roughness length must be positive.")
    if np.any((height - displacement) <= roughness):
        raise OutOfRangeError("Height must exceed displacement plus roughness length.")
    
    c = karman**2 / np.log((height - displacement) / roughness) ** 2

    return c


def surface_shear_stress(density: FloatArray, friction_velocity: FloatArray) -> FloatArray:
    """
    Surface shear stress (momentum flux) from friction velocity.

    Parameters
    ----------
    density : Air density (kg/m³), must be positive.
    friction_velocity : Friction velocity u* (m/s), non-negative.

    Returns
    -------
    Surface shear stress in N/m² (Pa).

    Raises
    ------
    OutOfRangeError
        If density is not positive or friction_velocity is negative.

    Notes
    -----
    tau = rho * u*^2

    Examples
    --------
    >>> print(round(surface_shear_stress(1.225, 0.3), 5))
    0.11025
    """
    if np.any(density <= 0):
        raise OutOfRangeError("Density must be positive.")
    if np.any(friction_velocity < 0):
        raise OutOfRangeError("Friction velocity must be non-negative.")
    
    tau = density * friction_velocity**2

    return tau


def aerodynamic_resistance(
    wind_speed: FloatArray,
    height: FloatArray,
    roughness: FloatArray,
    displacement: FloatArray = 0.0,
    karman: FloatArray = KARMAN,
) -> FloatArray:
    """
    Neutral aerodynamic resistance to turbulent transfer.

    The bulk aerodynamic resistance to momentum or scalar transfer under
    neutral stability, used in resistance-based flux and Penman-Monteith
    formulations.

    Parameters
    ----------
    wind_speed : Wind speed at the reference height (m/s), must be positive.
    height : Reference height (m), above displacement + roughness.
    roughness : Aerodynamic roughness length z0 (m), must be positive.
    displacement : Zero-plane displacement height d (m), default 0.0.
    karman : von Kármán constant (default from constants).

    Returns
    -------
    Aerodynamic resistance in s/m.

    Raises
    ------
    OutOfRangeError
        If wind_speed or roughness is not positive, or height is not above
        d + z0.

    Notes
    -----
    Neutral form, assuming equal momentum and scalar roughness lengths:
    ra = ln((z - d) / z0) ** 2 / (k ** 2 * U)

    Examples
    --------
    >>> print(round(aerodynamic_resistance(3.0, 10.0, 0.03), 2))
    70.3
    """
    if np.any(wind_speed <= 0):
        raise OutOfRangeError("Wind speed must be positive.")
    if np.any(roughness <= 0):
        raise OutOfRangeError("Roughness length must be positive.")
    if np.any((height - displacement) <= roughness):
        raise OutOfRangeError("Height must exceed displacement plus roughness length.")
    
    ra = np.log((height - displacement) / roughness) ** 2 / (karman**2 * wind_speed)

    return ra


def moist_air_density(
    temp: FloatArray, pressure: FloatArray, mixing_ratio: FloatArray, unit: str = "C"
) -> FloatArray:
    """
    Density of moist air from temperature, pressure, and mixing ratio.

    Moist air is less dense than dry air at the same temperature and pressure
    because water vapor is lighter than dry air. The effect is captured through
    the virtual temperature, so the dry-air gas constant applies.

    Parameters
    ----------
    temp : Air temperature.
    pressure : Total air pressure (kPa), must be positive.
    mixing_ratio : Water vapor mixing ratio (kg/kg), non-negative.
    unit : Unit of temp: "C", "F", or "K" (default "C").

    Returns
    -------
    Moist air density in kg/m³.

    Raises
    ------
    OutOfRangeError
        If pressure is not positive or mixing_ratio is negative.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Ideal gas law with virtual temperature:
    rho = P / (R_air * Tv)
    with P in pascals, R_air the dry-air gas constant, and Tv = T * (1 + 0.61 * w)
    the virtual temperature in kelvin. With mixing_ratio = 0 this reduces to
    air_density.

    Examples
    --------
    >>> print(round(moist_air_density(15, 101.325, 0.01), 4))
    1.2176
    """
    if np.any(pressure <= 0):
        raise OutOfRangeError("Pressure must be positive.")

    temp_K = convert_temperature(temp, unit.upper(), "K")
    tv_K = virtual_temperature(temp_K, mixing_ratio, unit="K")
    p_pa = pressure * 1000.0  # kPa -> Pa
    rho = p_pa / (R_AIR * tv_K)

    return rho
