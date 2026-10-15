"""
atmoflux.core
=================
Defines shared data structures and base classes used across atmoflux. 
Includes common containers and abstractions for surface, atmospheric, 
or flux state that are not tied to a single physical process. Compatible 
with both scalar and array-valued inputs.

"""

from __future__ import annotations

# Standard imports
from dataclasses import dataclass
from typing import TypeAlias

# Outside imports
import numpy as np

# Numeric inputs and outputs accept Python scalars or NumPy arrays.
FloatArray: TypeAlias = float | np.ndarray
IntArray: TypeAlias = int | np.ndarray


@dataclass
class EnergyBalance:
    """
    Container for surface energy-balance components and derived diagnostics.

    Stores the four primary energy-budget terms and computes the budget
    residual and Bowen ratio on construction. All terms use the sign convention
    that net radiation is positive downward (into the surface) and the turbulent
    and storage fluxes are positive when directed away from the surface, so a
    perfectly closed budget gives ``residual == 0``.

    Each field accepts a scalar or a NumPy array; derived attributes broadcast to
    the shape of the inputs.

    Parameters
    ----------
    net_radiation : Net all-wave radiation, Rn (W/m²).
    sensible_heat : Sensible heat flux, H (W/m²).
    latent_heat : Latent heat flux, LE (W/m²).
    ground_heat : Ground (storage) heat flux, G (W/m²), default 0.0.

    Attributes
    ----------
    net_radiation : FloatArray or numpy.ndarray
        Net all-wave radiation, Rn (W/m²).
    sensible_heat : FloatArray or numpy.ndarray
        Sensible heat flux, H (W/m²).
    latent_heat : FloatArray or numpy.ndarray
        Latent heat flux, LE (W/m²).
    ground_heat : FloatArray or numpy.ndarray
        Ground heat flux, G (W/m²).
    residual : FloatArray or numpy.ndarray
        Budget closure residual, Rn - G - H - LE (W/m²).
    bowen_ratio : FloatArray or numpy.ndarray
        Bowen ratio, H / LE (dimensionless); ``inf`` or ``nan`` where LE is 0.

    Examples
    --------
    >>> eb = EnergyBalance(net_radiation=400.0, sensible_heat=150.0,
    ...                    latent_heat=200.0, ground_heat=50.0)
    >>> eb.residual
    0.0
    >>> round(eb.bowen_ratio, 2)
    0.75
    >>> eb
    EnergyBalance(Rn=400.0, H=150.0, LE=200.0, G=50.0, residual=0.0)
    """

    net_radiation: FloatArray | np.ndarray
    sensible_heat: FloatArray | np.ndarray
    latent_heat: FloatArray | np.ndarray
    ground_heat: FloatArray | np.ndarray = 0.0

    def __post_init__(self) -> None:
        """Compute the budget residual and Bowen ratio from the stored fluxes."""
        self.residual = (
            self.net_radiation
            - self.ground_heat
            - self.sensible_heat
            - self.latent_heat
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            self.bowen_ratio = np.divide(self.sensible_heat, self.latent_heat)
        # Preserve Python scalars for scalar inputs rather than 0-d arrays.
        if np.ndim(self.bowen_ratio) == 0:
            self.bowen_ratio = float(self.bowen_ratio)

    def __repr__(self) -> str:
        """Compact representation that summarizes arrays by shape and mean."""

        def fmt(value: FloatArray | np.ndarray) -> str:
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                return f"{float(arr):.1f}"
            return f"<array shape={arr.shape}, mean={np.nanmean(arr):.1f}>"

        return (
            f"EnergyBalance(Rn={fmt(self.net_radiation)}, "
            f"H={fmt(self.sensible_heat)}, LE={fmt(self.latent_heat)}, "
            f"G={fmt(self.ground_heat)}, residual={fmt(self.residual)})"
        )

    def to_dict(self) -> dict:
        """
        Convert the energy balance to a dictionary.

        Returns
        -------
        Dictionary containing all primary fluxes and derived diagnostics.

        Examples
        --------
        >>> eb = EnergyBalance(400.0, 150.0, 200.0, 50.0)
        >>> eb.to_dict()["residual"]
        0.0
        """
        return {
            "net_radiation": self.net_radiation,
            "sensible_heat": self.sensible_heat,
            "latent_heat": self.latent_heat,
            "ground_heat": self.ground_heat,
            "residual": self.residual,
            "bowen_ratio": self.bowen_ratio,
        }


@dataclass
class AtmosphericState:
    """
    Container bundling the primary atmospheric state variables.

    Groups the near-surface meteorological variables that are commonly passed
    together through flux calculations, so callers can carry a single object
    rather than many loose arguments. No unit conversion or derivation is
    performed; the container simply records the state in the package's standard
    units. Each field accepts a scalar or a NumPy array.

    Parameters
    ----------
    temperature : Air temperature (°C).
    pressure : Air pressure (kPa).
    wind_speed : Wind speed (m/s).
    relative_humidity : Relative humidity (%), optional.

    Attributes
    ----------
    temperature : FloatArray or numpy.ndarray
        Air temperature (°C).
    pressure : FloatArray or numpy.ndarray
        Air pressure (kPa).
    wind_speed : FloatArray or numpy.ndarray
        Wind speed (m/s).
    relative_humidity : FloatArray, numpy.ndarray, or None
        Relative humidity (%), or None if not supplied.

    Examples
    --------
    >>> state = AtmosphericState(temperature=20.0, pressure=101.325,
    ...                          wind_speed=3.0, relative_humidity=55.0)
    >>> state.temperature
    20.0
    >>> state
    AtmosphericState(T=20.0°C, P=101.3kPa, U=3.0m/s, RH=55.0%)
    """

    temperature: FloatArray | np.ndarray
    pressure: FloatArray | np.ndarray
    wind_speed: FloatArray | np.ndarray
    relative_humidity: FloatArray | np.ndarray | None = None

    def __repr__(self) -> str:
        """Compact representation summarizing arrays by shape and mean."""

        def fmt(value: FloatArray | np.ndarray) -> str:
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                return f"{float(arr):.1f}"
            return f"<array shape={arr.shape}, mean={np.nanmean(arr):.1f}>"

        rh = "None" if self.relative_humidity is None else f"{fmt(self.relative_humidity)}%"
        return (
            f"AtmosphericState(T={fmt(self.temperature)}°C, "
            f"P={fmt(self.pressure)}kPa, U={fmt(self.wind_speed)}m/s, RH={rh})"
        )

    def to_dict(self) -> dict:
        """
        Convert the atmospheric state to a dictionary.

        Returns
        -------
        Dictionary containing all stored state variables.

        Examples
        --------
        >>> state = AtmosphericState(20.0, 101.325, 3.0)
        >>> state.to_dict()["pressure"]
        101.325
        """
        return {
            "temperature": self.temperature,
            "pressure": self.pressure,
            "wind_speed": self.wind_speed,
            "relative_humidity": self.relative_humidity,
        }