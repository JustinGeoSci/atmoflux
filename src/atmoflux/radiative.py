"""
atmoflux.radiative
=================
Computes radiative fluxes at the surface or in the atmosphere. 
Includes blackbody emission, net shortwave and longwave components, net all-wave
radiation, and clear-sky atmospheric emissivity.

"""

from __future__ import annotations

# Outside imports
import numpy as np

# imports from within atmoflux
from .constants import STEFAN_BOLTZMANN
from .core import FloatArray
from .exceptions import OutOfRangeError
from .temperature import convert_temperature


def blackbody_radiation(
    temp: FloatArray, emissivity: FloatArray = 1.0, unit: str = "K"
) -> FloatArray:
    """
    Radiant emittance of a surface from the Stefan-Boltzmann law.

    Parameters
    ----------
    temp : Surface temperature.
    emissivity : Surface emissivity in (0, 1] (default 1.0 for a blackbody).
    unit : Unit of input temperature: "C", "F", or "K" (default "K").

    Returns
    -------
    Emitted longwave radiative flux in W/m².

    Raises
    ------
    OutOfRangeError
        If emissivity is outside (0, 1].
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Stefan-Boltzmann law:
    L = emissivity * sigma * T ** 4
    with T in kelvin and sigma the Stefan-Boltzmann constant.

    Examples
    --------
    >>> print(round(blackbody_radiation(288.0), 2))
    390.11
    >>> print(round(blackbody_radiation(15.0, 0.97, unit="C"), 2))
    379.19
    """
    if np.any(emissivity <= 0) or np.any(emissivity > 1):
        raise OutOfRangeError("Emissivity must be in the interval (0, 1].")
    
    temp_K = convert_temperature(temp, unit.upper(), "K")
    l = emissivity * STEFAN_BOLTZMANN * temp_K**4

    return l


def net_shortwave(sw_down: FloatArray, albedo: FloatArray) -> FloatArray:
    """
    Net shortwave radiation absorbed at a surface.

    Parameters
    ----------
    sw_down : Incoming (downwelling) shortwave irradiance (W/m²).
    albedo : Surface shortwave albedo (reflectance) in [0, 1].

    Returns
    -------
    Net shortwave radiation absorbed by the surface (W/m²).

    Raises
    ------
    OutOfRangeError
        If albedo is outside [0, 1].

    Notes
    -----
    SW_net = sw_down * (1 - albedo)

    Examples
    --------
    >>> print(net_shortwave(800.0, 0.2))
    640.0
    """
    if np.any(albedo < 0) or np.any(albedo > 1):
        raise OutOfRangeError("Albedo must be in the interval [0, 1].")
    
    sw_net = sw_down * (1 - albedo)

    return sw_net


def net_longwave(
    lw_down: FloatArray, temp_surface: FloatArray, emissivity: FloatArray = 1.0, unit: str = "K"
) -> FloatArray:
    """
    Net longwave radiation at a surface (positive into the surface).

    Parameters
    ----------
    lw_down : Incoming (downwelling) longwave irradiance (W/m²).
    temp_surface : Surface (skin) temperature.
    emissivity : Surface emissivity in (0, 1] (default 1.0).
    unit : Unit of temp_surface: "C", "F", or "K" (default "K").

    Returns
    -------
    Net longwave radiation (W/m²); positive values indicate a net gain by the
    surface, negative values a net loss.

    Raises
    ------
    OutOfRangeError
        If emissivity is outside (0, 1].
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Upwelling longwave includes emitted plus reflected components:
    LW_up = emissivity * sigma * Ts ** 4 + (1 - emissivity) * lw_down
    LW_net = lw_down - LW_up

    Examples
    --------
    >>> print(round(net_longwave(350.0, 288.0), 2))
    -40.11
    """
    if np.any(emissivity <= 0) or np.any(emissivity > 1):
        raise OutOfRangeError("Emissivity must be in the interval (0, 1].")
    
    ts_K = convert_temperature(temp_surface, unit.upper(), "K")
    lw_up = emissivity * STEFAN_BOLTZMANN * ts_K**4 + (1 - emissivity) * lw_down
    lw_net = lw_down - lw_up

    return lw_net


def net_radiation(
    sw_down: FloatArray,
    lw_down: FloatArray,
    albedo: FloatArray,
    temp_surface: FloatArray,
    emissivity: FloatArray = 1.0,
    unit: str = "K",
) -> FloatArray:
    """
    Net all-wave radiation at a surface.

    Combines the net shortwave and net longwave components into the total net
    radiation, positive downward (into the surface).

    Parameters
    ----------
    sw_down : Incoming shortwave irradiance (W/m²).
    lw_down : Incoming longwave irradiance (W/m²).
    albedo : Surface shortwave albedo in [0, 1].
    temp_surface : Surface (skin) temperature.
    emissivity : Surface emissivity in (0, 1] (default 1.0).
    unit : Unit of temp_surface: "C", "F", or "K" (default "K").

    Returns
    -------
    Net all-wave radiation Rn (W/m²).

    Raises
    ------
    OutOfRangeError
        If albedo or emissivity is outside its valid range.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Rn = SW_net + LW_net = sw_down * (1 - albedo) + (lw_down - LW_up)

    Examples
    --------
    >>> print(round(net_radiation(800.0, 350.0, 0.2, 288.0), 2))
    599.89
    """
    sw_net = net_shortwave(sw_down, albedo)
    lw_net = net_longwave(lw_down, temp_surface, emissivity, unit)
    rn = sw_net + lw_net

    return rn


def clear_sky_emissivity(temp_air: FloatArray, vapor_pressure: FloatArray, unit: str = "C") -> FloatArray:
    """
    Clear-sky atmospheric emissivity using Brutsaert's relation.

    Estimates the effective emissivity of a cloudless atmosphere from screen-level
    air temperature and vapor pressure, for use in downwelling longwave
    estimates.

    Parameters
    ----------
    temp_air : Air temperature.
    vapor_pressure : Near-surface actual vapor pressure (kPa), must be positive.
    unit : Unit of temp_air: "C", "F", or "K" (default "C").

    Returns
    -------
    Clear-sky atmospheric emissivity (dimensionless).

    Raises
    ------
    OutOfRangeError
        If vapor_pressure is not positive.
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    Brutsaert (1975):
    emissivity = 1.24 * (e / T) ** (1 / 7)
    with e in hectopascals (hPa) and T in kelvin. Vapor pressure supplied in kPa
    is converted to hPa internally.

    Examples
    --------
    >>> print(round(clear_sky_emissivity(20, 1.5), 4))
    0.8109
    """
    if np.any(vapor_pressure <= 0):
        raise OutOfRangeError("Vapor pressure must be positive.")
    
    temp_K = convert_temperature(temp_air, unit.upper(), "K")
    e_hpa = vapor_pressure * 10.0  # kPa -> hPa
    ecs = 1.24 * (e_hpa / temp_K) ** (1.0 / 7.0)

    return ecs


def net_longwave_cloud(
    temp_air: FloatArray,
    vapor_pressure: FloatArray,
    cloud_fraction: FloatArray,
    unit: str = "C",
) -> FloatArray:
    """
    Net longwave radiation at the surface with a cloud adjustment (FAO-56 style).

    Estimates net outgoing longwave radiation from air temperature, humidity, and
    a relative cloudiness term, as used in reference evapotranspiration when
    surface temperature is unavailable.

    Parameters
    ----------
    temp_air : Air temperature.
    vapor_pressure : Actual vapor pressure (kPa), must be positive.
    cloud_fraction : Relative shortwave cloudiness term Rs/Rso in [0, 1], where 1
        is clear sky and 0 is fully overcast.
    unit : Unit of temp_air: "C", "F", or "K" (default "C").

    Returns
    -------
    Net longwave radiation (W/m²); positive indicates a net loss from the
    surface (outgoing), following the FAO-56 sign convention.

    Raises
    ------
    OutOfRangeError
        If vapor_pressure is not positive or cloud_fraction is outside [0, 1].
    InvalidUnitError
        If unit is invalid.

    Notes
    -----
    FAO-56 net longwave (converted to W/m² and using the package sign of a
    positive outgoing loss):
    Rnl = sigma * T^4 * (0.34 - 0.14*sqrt(e)) * (1.35 * (Rs/Rso) - 0.35)
    with e in kPa and T in kelvin.

    Examples
    --------
    >>> print(round(net_longwave_cloud(20, 1.5, 1.0), 2))
    70.58
    """
    if np.any(vapor_pressure <= 0):
        raise OutOfRangeError("Vapor pressure must be positive.")
    if np.any(cloud_fraction < 0) or np.any(cloud_fraction > 1):
        raise OutOfRangeError("Cloud fraction must be in the interval [0, 1].")
    
    temp_K = convert_temperature(temp_air, unit.upper(), "K")
    emissivity_term = 0.34 - 0.14 * np.sqrt(vapor_pressure)
    cloud_term = 1.35 * cloud_fraction - 0.35
    rnl = STEFAN_BOLTZMANN * temp_K**4 * emissivity_term * cloud_term

    return rnl


def diffuse_fraction(clearness_index: FloatArray) -> FloatArray:
    """
    Diffuse fraction of global shortwave radiation from the clearness index.

    Partitions incoming global shortwave into its diffuse component using the
    Erbs relation, widely applied in solar resource and canopy radiation models.

    Parameters
    ----------
    clearness_index : Clearness index kt = Rs / Ra in [0, 1].

    Returns
    -------
    Diffuse fraction (dimensionless) in [0, 1].

    Raises
    ------
    OutOfRangeError
        If clearness_index is outside [0, 1].

    Notes
    -----
    Erbs et al. (1982) piecewise relation:
    kt <= 0.22: kd = 1 - 0.09*kt
    0.22 < kt <= 0.80: kd = 0.9511 - 0.1604*kt + 4.388*kt^2
                            - 16.638*kt^3 + 12.336*kt^4
    kt > 0.80: kd = 0.165

    Examples
    --------
    >>> print(round(diffuse_fraction(0.5), 4))
    0.6591
    """
    if np.any(clearness_index < 0) or np.any(clearness_index > 1):
        raise OutOfRangeError("Clearness index must be in the interval [0, 1].")
    
    kt = np.asarray(clearness_index, dtype=float)
    kd = np.where(
        kt <= 0.22,
        1 - 0.09 * kt,
        np.where(
            kt <= 0.80,
            0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4,
            0.165,
        ),
    )

    if kd.ndim == 0:
        return float(kd)
    
    return kd