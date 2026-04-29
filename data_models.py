from dataclasses import dataclass

import numpy as np

@dataclass(kw_only=True)
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



