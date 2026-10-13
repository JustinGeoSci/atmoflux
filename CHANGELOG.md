# Changelog

All notable changes to **atmoflux** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-06-06

Additive feature release. No existing 1.0.0 functions changed signature or
behavior, so upgrading from 1.0.0 requires no code changes.

### Added

- **New module `solar`** — solar position and radiation geometry:
  `day_of_year`, `solar_declination`, `hour_angle`, `solar_zenith_angle`,
  `solar_elevation`, `sunset_hour_angle`, `daylight_hours`,
  `extraterrestrial_radiation`, and `clear_sky_radiation` (Angstrom-Prescott).
- **New module `stability`** — surface-layer atmospheric stability diagnostics
  (all closed-form, non-iterative): `bulk_richardson_number`, `obukhov_length`,
  `stability_parameter`, `psi_momentum`, `psi_heat`, and `stability_class`.
- **New module `atmosphere`** — standard-atmosphere and barometric helpers:
  `scale_height`, `pressure_at_altitude`, `hypsometric_thickness`,
  `density_altitude`, and `standard_atmosphere`.
- **`temperature`**: `wet_bulb_temperature`, `equivalent_potential_temperature`,
  and `moist_adiabatic_lapse_rate`.
- **`humidity`**: `saturation_vp_ice`, `specific_humidity_from_dewpoint`,
  `relative_humidity_from_specific_humidity`, and `precipitable_water`.
- **`wind`**: `wind_components`, `wind_power_density`, `roughness_from_canopy`,
  and `displacement_from_canopy`.
- **`radiative`**: `net_longwave_cloud` (cloud-adjusted) and `diffuse_fraction`
  (Erbs partitioning).
- **`turbulent`**: `surface_shear_stress` and `aerodynamic_resistance`.
- **`hydro`**: `equilibrium_evaporation`, `priestley_taylor`, and `hargreaves`.
- **`balance`**: `available_energy`, `energy_balance_ratio`, and
  `ground_heat_fraction`.
- **`core`**: `AtmosphericState` dataclass for bundling near-surface state
  variables, alongside the existing `EnergyBalance`.
- **`constants`**: solar/Angstrom-Prescott coefficients, the Priestley-Taylor
  coefficient, the dry adiabatic lapse rate (`DRY_ADIABATIC_LAPSE_RATE`), and
  standard-atmosphere reference values.
- Expanded the `pytest` suite and doctests to cover all new functionality.

### Changed

- Raised the minimum supported NumPy version to `>=1.26` to align with the
  current scientific-Python support window. (Python `>=3.9` is unchanged.)

## [1.0.0] - 2026-05-30

First stable release. Establishes the public API across all core modules under
semantic versioning.

### Added

- **State modules**: `temperature` (unit conversion, dew point, potential and
  virtual temperature, lapse rate, surface temperature from longwave),
  `humidity` (saturation and actual vapor pressure and its slope, relative and
  specific humidity, mixing ratio, vapor pressure deficit, absolute humidity),
  and `wind` (speed unit conversion, vector components, log and power-law
  profiles, friction velocity, shear).
- **Flux / process modules**: `radiative` (blackbody emission, net shortwave and
  longwave, net radiation, clear-sky emissivity), `turbulent` (air density,
  bulk-aerodynamic sensible and latent heat fluxes, transfer coefficients),
  `hydro` (latent-heat-to-evaporation conversion, Penman, Penman-Monteith,
  FAO-56 reference ET), `aerosols` (gravitational settling, dry deposition,
  surface emission flux), and `balance` (energy budget residual, Bowen ratio,
  `EnergyBalance` assembly).
- **Support modules**: `constants` (physical and derived constants, all
  pressures in kPa), `core` (the `EnergyBalance` container), and `exceptions`
  (the `AtmofluxError` hierarchy).
- Vectorized API: all functions accept scalars or NumPy arrays.
- Full `pytest` suite mirroring the package structure, plus doctests in every
  public function.

[1.1.0]: https://github.com/JustinGeoSci/atmoflux/releases/tag/v1.1.0
[1.0.0]: https://github.com/JustinGeoSci/atmoflux/releases/tag/v1.0.0
