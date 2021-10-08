#!/usr/bin/env python

import numpy as np
import scipy.optimize

@np.vectorize
def zhang_particle_dynamics(radius, kinematic_viscosity, gravity, 
                            delta_density, fluid_density, thermal_diffusivity,
                            chemical_diffusivity, brunt_vaisala=None):
    """
    Calculate falling velocity and dynamical properties of a falling sphere
    
    This function estimates the termial falling velocity of a spherical particle through
    a viscous fluid due to the action of gravity. It goes beyond Stokes flow (low Reynolds
    number) by including an empical drag coefficent following Zhang and Xu (2003) which
    involves self-consistent solution of the falling velocity and Reynolds number. The
    falling velocity is then used to evaluate the thickness of the momentum, thermal and
    chemical boundary layers follwing the scaling relations developed in Inman et al (2020),
    which involves calculation of other dimensionless quantities. 
    
    Input arguments:
    * radius: particle radius (m)
    * kinematic_viscosity: viscosity of fluid (m^2/s)
    * gravity: accelration due to gravity (m/s^2)
    * delta_density: difference in density between particle and fluid (kg/m^3)
    * fluid_density: density of fluid (kg/m^3)
    * thermal_diffusivity: thermal diffusivity of fluid (m^2/s)
    * chemical_diffusivity: chemical diffusivity of fluid (m^2/s)
    * brunt_vaisala: Brunt–Väisälä frequency. Optional argument. If None (the default)
          an estimate for the outer core is used. (Hz)
    
    Returns:
    * falling_velocity: velocity of particle, positive downwards (m/s)
    * drag_coefficient: empricial drag coefficent based on Reynolds number scaling (-)
    * re: Reynolds number (-)
    * pe_t: Thermal Péclet number (-) 
    * pe_c: Chemical Péclet number (-)
    * fr: Froude number (-)
    * delta_u: Mechanical boundary layer thickness (m)
    * delta_t: Thermal boundary layer thickness (m)
    * delta_c: Chemical boundary layer thickness (m)
    
    References:
        Zhang, Y., & Xu, Z. (2003). Kinetics of convective crystal dissolution and melting,
    with applications to methane hydrate dissolution and dissociation in seawater. Earth 
    and Planetary Science Letters, 213(1–2), 133–148.
    https://doi.org/10.1016/S0012-821X(03)00297-8
        Inman, B. G., Davies, C. J., Torres, C. R., & Franks, P. J. S. (2020). Deformation
    of ambient chemical gradients by sinking spheres. Journal of Fluid Mechanics, 892, A33.
    https://doi.org/10.1017/jfm.2020.191
    """
    # First calculate Re and the falling velocity - pg 139 of Zhang, point "1"
    # uses _opt_zhang function to do optimisation. Assume solution changes sign
    # between -1 and 100 m/s
    falling_velocity = scipy.optimize.brentq(_fzhang_opt, -1.0, 100.0,
                            args=(radius, kinematic_viscosity, gravity, delta_density,
                                  fluid_density))
    
    # Recalculate drag and Re from solution velocity
    re, drag_coefficient = _fzhang_re_cd(falling_velocity, radius, kinematic_viscosity)
    
    # Dimensionless numbers
    sc = kinematic_viscosity / chemical_diffusivity
    pr = kinematic_viscosity / thermal_diffusivity
    pe_t = re * kinematic_viscosity / thermal_diffusivity
    pe_c = re * kinematic_viscosity / chemical_diffusivity
    
    if brunt_vaisala is None:
        # Default - use PREM estimates... using prem4derg
        # implementation and denstiy at and 100m above ICB
        rho_0 = 12166.33 # density above ICB (kg/m^3
        drho_dz = -0.0005 # density grad above ICB (kg/m^2)
        icb_g = 4.4 # gravity above ICB
        brunt_vaisala = np.sqrt(-(icb_g / rho_0) * drho_dz)
    fr = falling_velocity / (brunt_vaisala * radius)
    
    # Boundary layer analysisi
    if re < 1e-2:
        delta_u = 2.0 * radius
        delta_c = 2.0 * radius
        delta_t = 2.0 * radius
    elif re < 1e2 and re > 1e-2:               # Intermediate Re case
        delta_u = 2.0 * radius
        delta_t = 2.0 * radius                 # For T, apply low Re limit as Pe is low till Re~100
        delta_c = 2.0 * pe_c**(-1/3) * 2.0 * radius
    elif re > 1e2:                             # High Re case
        delta_u = re**(-0.5) * 2.0 * radius 
        delta_c = 4.5 * re**(-0.5) * (sc)**(-1/3) * 2.0 * radius # FIXME: where does the 4.5 come from?
        delta_t = 3.0 * re**(-0.5) * (pr)**(-0.5) * 2.0 * radius # FIXME: where does the 3 come from?
    # FIXME - limits on Re.

    return falling_velocity, drag_coefficient, re, pe_t, pe_c, fr, delta_u, delta_t, delta_c


def _fzhang_re_cd(u, r, mu):
    Re  = np.abs(2*r * u / mu)                                              # Eqn 1  of ZX02
    Cd  = (24.0/Re) * (1.0 + 0.15*Re**0.687) + 0.42/(1.0 + 42500*Re**-1.16) # Eqn 19 of ZX02
    return Re, Cd


def _fzhang_opt(u, rad, mu, g, drho, rhol):
    """
    Finds the velocity error based on an initial guess 
    u using the eqns in Zhang and Xu (2003). Call with optimizer until error is zero
    
    u  = initial velocity guess
    rad= particle radius
    mu = kinematic viscosity
    Re = Reynolds number
    g  = gravity at ICB
    drho = solid-liquid density difference
    rhol = liquid density
    """
    if rad < 0.0:
        return np.nan # See above.
    re, cd = _fzhang_re_cd(u, rad, mu)

    result = (8.0 * g * rad * drho) / (3.0 * rhol * cd)
    result = np.sqrt(result)
    result = result - u

    return result