# atmoflux
A Python package for atmospheric and surface flux calculation and analysis.

## Overview

**atmoflux** is a Python library for atmospheric and surface flux calculations relevant to climate, micrometeorology, and urban environmental research. The package provides modular tools for radiative, turbulent, hydrological, aerosol, and energy balance calculations, alongside supporting utilities for atmospheric state variables and physical constants.

The **atmoflux** library emphasizes physically explicit formulations and transparent diagnostics, supporting process-based analysis and integration with statistical or machine learning models. All functions accept scalars or NumPy arrays, so calculations vectorize naturally over gridded or time-series data.

**atmoflux** is being developed alongside dissertation research focused on urban microclimates, heat stress, and land–atmosphere interactions, and is intended to evolve as a flexible research framework rather than a fixed operational model.

## Citation

If you use **atmoflux** in academic work, please cite the software using the DOI provided below.

Each release is archived via Zenodo and assigned a permanent DOI to ensure reproducibility.

The recommended citation format is available in CITATION.cff and via the “Cite this repository” button on GitHub.

DOI: https://doi.org/10.5281/zenodo.20470203 

## Requirements

- Python ≥ 3.10 (developed and tested in 3.12)
- [NumPy](https://numpy.org/) ≥ 1.26 — the only runtime dependency

Keeping the dependency footprint minimal is a deliberate design goal; NumPy is the sole external requirement.

## Installation

### From PyPI (recommended)

```bash
pip install atmoflux
```

### From source

Clone the repository and install from the project root. Use an editable
install if you intend to modify the code:

```bash
git clone https://github.com/JustinGeoSci/atmoflux.git
cd atmoflux

pip install .       # regular installation
pip install -e .    # editable (development) installation
```

## Quick Start

```python
import atmoflux as af

# Net all-wave radiation at a surface (W/m²)
rn = af.radiative.net_radiation(
    sw_down=800, lw_down=350, albedo=0.2, temp_surface=295
)

# Turbulent sensible heat flux via the bulk-aerodynamic method
rho = af.turbulent.air_density(temp=22, pressure=101.325)
h = af.turbulent.sensible_heat_flux(
    rho, wind_speed=3.0, temp_air=22, temp_surface=25, transfer_coeff=0.0013
)

# Assemble a surface energy balance with derived diagnostics
eb = af.balance.energy_balance(
    rn, sensible_heat=h, latent_heat=250, ground_heat=60
)
print(eb.residual)       # closure residual (W/m²)
print(eb.bowen_ratio)    # H / LE
```

## Modules

**Atmospheric state**

| Module | Description |
| --- | --- |
| `temperature` | Unit conversion, dew point, potential, virtual, wet-bulb and equivalent potential temperature, lapse rates, surface temperature from longwave |
| `humidity` | Saturation and actual vapor pressure, relative, specific, and saturation specific humidity, mixing ratio, vapor pressure deficit, absolute humidity, precipitable water |
| `wind` | Speed unit conversion, vector components, log and power-law profiles, friction velocity, shear, wind power density, canopy-derived roughness and displacement |
| `atmosphere` | Scale height, barometric pressure with altitude, hypsometric thickness, density altitude, US Standard Atmosphere profile |

**Radiation**
| Module | Description |
| --- | --- |
| `solar` | Solar declination, hour angle, zenith and elevation, daylight hours, extraterrestrial and clear-sky shortwave radiation |
| `radiative` | Blackbody emission, net shortwave and longwave (clear-sky and cloud-adjusted), net radiation, clear-sky emissivity, diffuse fraction |

**Surface fluxes**

| Module | Description |
| --- | --- |
| `turbulent` | Dry and moist air density, bulk-aerodynamic sensible and latent heat fluxes, transfer coefficients, surface shear stress, aerodynamic resistance |
| `stability` | Bulk Richardson number, Obukhov length, stability parameter and correction functions, stability classification |
| `hydro` | Latent-heat-to-evaporation conversion, Penman, Penman-Monteith, FAO-56 reference ET, Priestley-Taylor, Hargreaves, equilibrium and Dalton evaporation |
| `aerosols` | Gravitational settling, particle Reynolds number, dry deposition, surface emission flux |
| `balance` | Energy budget residual, Bowen ratio, available energy, energy balance ratio, ground heat fraction, `EnergyBalance` assembly |

**Core & utilities**

| Module | Description |
| --- | --- |
| `core` | Shared data structures, including the `EnergyBalance` and `AtmosphericState` containers |
| `constants` | Physical and derived constants with documented units |
| `exceptions` | Package exception hierarchy rooted at `AtmofluxError` |

## Testing

The package ships with a `pytest` suite mirroring the module structure, plus doctests in every public function.

```bash
# Unit tests
pytest atmoflux/tests/

# Docstring examples
pytest --doctest-modules atmoflux/
```

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for the full version history. The latest release is **1.2.0**.

## License

Released under the MIT License. See the [`LICENSE`](LICENSE) file for details.