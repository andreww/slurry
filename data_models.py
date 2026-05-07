from dataclasses import dataclass

import numpy as np

@dataclass(kw_only=True, frozen=True)
class solution_profiles:
    """
    Representation of solutions from the flayer code
    """
    radius: np.ndarray # Radius of solution points, in m
    pressure: np.ndarray # GPa
    temperature: np.ndarray # K
    liquid_composition: np.ndarray
    mass_fraction_solid_production_rate: np.ndarray # kg/s -- units?
    heat_production_rate: np.ndarray # W/m^3
    o_prod_rate: np.ndarray # 
    liquid_density: np.ndarray
    solid_density: np.ndarray
    solid_volume_fraction: np.ndarray

    def __post_init__(self):
        assert self.radius.shape == self.pressure.shape, "radius-pressure miss match"
        assert self.radius.shape == self.temperature.shape, "radius-temperature miss match"
        assert self.radius.shape == self.liquid_composition.shape, "radius-liqX miss match"
        assert self.radius.shape == self.mass_fraction_solid_production_rate.shape, "radius-gamma miss match"
        assert self.radius.shape == self.heat_production_rate.shape, "radius-heat miss match"
        assert self.radius.shape == self.o_prod_rate.shape, "radius-oprod miss match"
        assert self.radius.shape == self.liquid_density.shape, "radius-liquidrho miss match"
        assert self.radius.shape == self.solid_density.shape, "radius-solidrho miss match"
        assert self.radius.shape == self.solid_volume_fraction.shape, "radius-solidvf miss match"

@dataclass(kw_only=True, frozen=True)
class particle_histograms:
    """
    The population of particles is represented by 2D arrays
    indexed first by the location where we look at the particle
    property (radius here) and second by the location where the
    particle nucleated. So, the population of particle velocities
    at radius[i] is given by particle_velocities[i,:], for example.
    When interpreting these arrays remember that particles will only
    be found below (smaller r) than their own nuc_radius.
    """
    radius: np.ndarray # Radius (location) of solution points, in m
    nuc_radius: np.ndarray # Radius (location) of particle nucleation, in m
    particle_size: np.ndarray # Radius (size) of the particle, m
    particle_density: np.ndarray # Number density of particles, particles/m^3
    particle_velocity: np.ndarray # Velocity of particle, m/s
    particle_volume_growth_rate: np.ndarray # How fast is the particle volume growing, m^3/s
    particle_age: np.ndarray # How old is the particle, s


    def __post_init__(self):
        assert len(self.radius.shape) == 1, "Radius (position) must be 1D"
        assert len(self.nuc_radius.shape) == 1, "Nucleation radius (position) must be 1D"
        expected_shape = (self.radius.shape[0], self.nuc_radius.shape[0])
        assert self.particle_size.shape == expected_shape, "particle_size array mismatch"
        assert self.particle_density.shape == expected_shape, "particle_density array mismatch"
        assert self.particle_velocity.shape == expected_shape, "particle_velocity array mismatch"
        assert self.particle_volume_growth_rate.shape == expected_shape, "particle_growth_rate array mismatch"
        assert self.particle_age.shape == expected_shape, "particle_age array mismatch"