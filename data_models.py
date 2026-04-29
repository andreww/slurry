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