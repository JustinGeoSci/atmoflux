"""
atmoflux.constants
=================
Provides physical and derived constants used across the package.
Includes psychrometric constants, molecular ratios, latent heats, radiative and
turbulent-transfer constants, and aerosol microphysical parameters. All
pressures and vapor pressures in atmoflux are expressed in kilopascals (kPa).

"""

'''
Psychrometric Constants
Thermodynamic properties defining the relationship between moisture and air temperature.
'''
# Specific heat of dry air at constant pressure
# Measures the thermal energy needed to raise the temperature of 1 kg of dry air by 1 K
# Units: J/(kg·K) (Standard in atmospheric science and meteorology calculations)
CP_AIR = 1005.0  

# Lv: Latent heat of vaporization of water
# Measures the thermal energy required to convert 1 kg of water from liquid to vapor
# Units: J/kg
LV = 2.45e6  

# Psychrometric Constant
# Relates the change in air temperature to the change in vapor pressure during evaporation
# Sea-level approximation; the true value scales with pressure as
# gamma = CP_AIR * P / (RMW * LV). Modules needing precision compute it from pressure.
# Units: kPa/K
PC = 0.066

# Specific gas constant for dry air
# Relates pressure, density, and temperature for dry air
# Units: J/(kg·K)
R_AIR = 287.05287

# Specific gas constant for water vapor
# Relates pressure, density, and temperature for water vapor
# Units: J/(kg·K)
R_VAPOR = 461.52332

# Ratio of the molecular weight of water vapor to dry air
# Units: Dimensionless factor used in humidity calculations
RMW = 0.622  

# Saturation Vapor Pressure constants (Tetens formula)
# Units: A = kPa, B= dimensionless, C = degrees Celsius
SVP_A = 0.61078
SVP_B = 17.27
SVP_C = 237.3

# Saturation Vapor Pressure constants over ice (Tetens formula)
# Units: A = kPa, B = dimensionless, C = degrees Celsius
SVP_ICE_A = 0.61078
SVP_ICE_B = 21.875
SVP_ICE_C = 265.5

# Saturation Vapor Pressure constants (Alduchov & Eskridge, 1996)
# Improved Magnus form over liquid water, recommended for -40 °C to 50 °C
# Units: A = kPa, B = dimensionless, C = degrees Celsius
SVP_A_ALDUCHOV = 0.61094
SVP_B_ALDUCHOV = 17.625
SVP_C_ALDUCHOV = 243.04

# Saturation Vapor Pressure constants over ice (Alduchov & Eskridge, 1996)
# Improved Magnus form over a plane ice surface, recommended for -80 °C to 0 °C
# Units: A = kPa, B = dimensionless, C = degrees Celsius
SVP_ICE_A_ALDUCHOV = 0.61121
SVP_ICE_B_ALDUCHOV = 22.587
SVP_ICE_C_ALDUCHOV = 273.86

'''
Radiative Constants
Parameters governing shortwave and longwave radiative transfer.
'''
# Stefan-Boltzmann constant
# Relates blackbody radiant emittance to the fourth power of absolute temperature
# Units: W/(m²·K⁴)
STEFAN_BOLTZMANN = 5.670374419e-8

# Solar constant
# Mean top-of-atmosphere solar irradiance at 1 astronomical unit
# Units: W/m²
SOLAR_CONSTANT = 1361.6

'''
Turbulent-Transfer and Reference Constants
Parameters used in surface-layer flux and profile calculations.
'''
# von Kármán constant
# Empirical constant in the logarithmic wind profile and flux-gradient relations
# Units: dimensionless
KARMAN = 0.40

# Standard gravitational acceleration
# Units: m/s²
G = 9.80665

# Standard sea-level atmospheric pressure
# Units: kPa
P0 = 101.325

# Density of liquid water
# Used to convert latent heat flux to an equivalent depth of water
# Units: kg/m³
RHO_WATER = 1000.0

# Standard density of dry air (ISA, 15 °C and sea-level pressure P0)
# Reference value only; compute actual density with air_density(temp, pressure)
# Units: kg/m³
RHO_AIR_STD = 1.225

'''
Aerosol Microphysical Constants
Properties of air controlling particle settling and deposition.
'''
# Dynamic viscosity of air at ~15 °C
# Units: Pa·s
MU_AIR = 1.789e-5

# Mean free path of air molecules at sea level and ~15 °C
# Units: m
MFP_AIR = 6.63e-8

'''
Solar and Astronomical Constants
Parameters governing solar geometry and extraterrestrial radiation.
'''
# Solar declination amplitude (Cooper. 1969)
# Maximum tilt of Earth's axis toward the sun (axial obliquity)
# Units: degrees
SOLAR_DECLINATION_MAX = 23.45

# Angstrom-Prescott regression coefficients
# Default coefficients relating sunshine duration to clear-sky shortwave
# FAO-56 standard values
# Units: dimensionless
ANGSTROM_A = 0.25
ANGSTROM_B = 0.50

'''
Evaporation Constants
Empirical coefficients used in evapotranspiration parameterizations.
'''
# Priestley-Taylor coefficient
# Empirical multiplier on equilibrium evaporation for well-watered surfaces
# Units: dimensionless
PRIESTLEY_TAYLOR_ALPHA = 1.26

'''
Lapse-Rate and Standard-Atmosphere Constants
Reference values for the dry atmosphere and barometric calculations.
'''
# Dry adiabatic lapse rate
# Rate of temperature decrease for an unsaturated parcel rising adiabatically
# Equivalent to G / CP_AIR; provided explicitly for convenience
# Units: K/m
DRY_ADIABATIC_LAPSE_RATE = G / CP_AIR

# Reference temperature for the standard atmosphere at sea level
# Units: K
T0_STANDARD = 288.15

# Mean tropospheric (environmental) lapse rate of the US Standard Atmosphere
# Units: K/m
LAPSE_RATE_STANDARD = 0.0065