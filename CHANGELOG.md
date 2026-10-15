# Changelog

All notable changes to **atmoflux** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-10-15

Additive feature release. No existing function changed its default result:
every 1.1.0 call returns the same value in 1.2.0, so upgrading requires no code
changes.

### Added

- **Selectable saturation vapor pressure formula.** A new keyword argument
  `method` accepts `"tetens"` (default, unchanged behavior) or `"alduchov"`
  (Alduchov & Eskridge, 1996). The Alduchov form keeps relative error within
  about 0.4 % from -40 to 50 °C over water and -80 to 0 °C over ice, whereas
  Tetens error grows in the cold (about -2.5 % at -40 °C over water and -10 %
  at -80 °C over ice). Between 0 and 40 °C the two agree to within about 0.2 %.
  Available on:
  - **`humidity`**: `saturation_vp`, `actual_vp`, `saturation_vp_slope`,
    `saturation_vp_ice`, `relative_humidity`, `vapor_pressure_deficit`,
    `specific_humidity_from_dewpoint`, `relative_humidity_from_specific_humidity`,
    and the new `saturation_specific_humidity`.
  - **`temperature`**: `dewpoint_temperature` and `dewpoint_from_avp`.
  - **`hydro`**: `penman_evaporation`, `penman_monteith`,
    `equilibrium_evaporation`, and `priestley_taylor` (slope term).
    `potential_evapotranspiration` is unchanged and remains Tetens-only, as the
    FAO-56 standard specifies.
- **`humidity`**: `saturation_specific_humidity`.
- **`turbulent`**: `moist_air_density`, using the virtual temperature with the
  dry-air gas constant.
- **`hydro`**: `dalton_evaporation`, Dalton mass-transfer evaporation
  `E = K * VPD`.
- **`aerosols`**: `particle_reynolds_number`, for checking that
  `settling_velocity` is within the Stokes regime (Re < ~1).
- **`constants`**: Alduchov & Eskridge water (`SVP_A_ALDUCHOV`,
  `SVP_B_ALDUCHOV`, `SVP_C_ALDUCHOV`) and ice (`SVP_ICE_A_ALDUCHOV`,
  `SVP_ICE_B_ALDUCHOV`, `SVP_ICE_C_ALDUCHOV`) coefficients, and the Tetens ice
  coefficients (`SVP_ICE_A`, `SVP_ICE_B`, `SVP_ICE_C`) previously inlined in
  `saturation_vp_ice`.
- `py.typed` marker (PEP 561), so type checkers use atmoflux's annotations.

### Changed

- Minimum supported Python is now 3.10. Users on 3.9 can continue to use atmoflux 1.1.0.
- Corrected type annotations to reflect array support via shared `FloatArray`
  and `IntArray` aliases in `core`. No runtime or API changes.
- `dewpoint_temperature` and `dewpoint_from_avp` now read their Magnus
   coefficients from `constants` instead of inline literals. Results are unchanged.

### Fixed

- `wind.log_wind_profile` raised `ZeroDivisionError` instead of the documented
  `OutOfRangeError` when `roughness` was zero, because a redundant check ran
  before the roughness guard. The redundant check has been removed.
- Test expectations for `settling_velocity`, `dry_deposition_velocity`, and
  `scale_height` updated to match the 1.1.0 constant and buoyancy changes
  (the package code was already correct).

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

[1.2.0]: https://github.com/JustinGeoSci/atmoflux/releases/tag/v1.2.0
[1.1.0]: https://github.com/JustinGeoSci/atmoflux/releases/tag/v1.1.0
[1.0.0]: https://github.com/JustinGeoSci/atmoflux/releases/tag/v1.0.0