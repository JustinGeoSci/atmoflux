"""
atmoflux: Tools for computing atmospheric and surface fluxes.

This package provides state-variable helpers (temperature, humidity, wind),
physical constants, and process-based flux calculations for radiative,
turbulent, hydrological, aerosol, and energy balance applications.

"""

# Package metadata
__version__ = "1.2.0"
__author__ = "JustinGeoSci"
__email__ = "contact@justingeosci.dev"
__license__ = "MIT"
__description__ = "Tools for climate data processing and analysis"

# Atmospheric state
from . import temperature
from . import humidity
from . import wind
from . import atmosphere

# Radiation
from . import solar
from . import radiative

# Surface fluxes

from . import turbulent
from . import stability
from . import hydro
from . import aerosols
from . import balance

# Core & utilities
from . import core
from . import constants
from . import exceptions

__all__ = [
    "temperature",
    "humidity",
    "wind",
    "atmosphere",
    "solar",
    "radiative",
    "turbulent",
    "stability",
    "hydro",
    "aerosols",
    "balance",
    "core",
    "constants",
    "exceptions",
]