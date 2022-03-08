#!/usr/bin/env python

import warnings
import numpy as np
import scipy.optimize

@np.vectorize
def zhang_particle_dynamics(radius, kinematic_viscosity, gravity, 
                            delta_density, fluid_density, thermal_diffusivity,
                            chemical_diffusivity, brunt_vaisala=None, warn_reynolds=True):
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
    * warn_reynolds: generate python warning if the calculated Reynolds number / falling
          velocity / drag coefficent is too large for the emprical scaling. Optional 
          argument. Default is True (check and generate warning).
    
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
    # First calculate Re and the falling velocity 
    falling_velocity, re, drag_coefficient = self_consistent_falling_velocity(radius, 
                                  kinematic_viscosity, gravity, delta_density, fluid_density)
    
    if warn_reynolds and (re >= 3.0E5):
        warnings.warn(
            "Calculated Re {:g} too high for drag / velocity parameterisation. Treat results with care".format(re))
        
    # Dimensionless numbers
    pr, pe_t, sc, pe_c, fr = dimensionless_numbers(radius, re, falling_velocity, kinematic_viscosity, 
                          chemical_diffusivity, thermal_diffusivity, brunt_vaisala)    
    # Boundary layer analysis
    delta_u, delta_c, delta_t = boundary_layers(radius, re, pe_c, pe_t, sc, pr, fr)
    
    return falling_velocity, drag_coefficient, re, pe_t, pe_c, fr, delta_u, delta_t, delta_c


@np.vectorize
def self_consistent_falling_velocity(radius, kinematic_viscosity, gravity, delta_density, fluid_density):
    """
    Emprical falling velocity to Re 3E5 
    
    Calculate Re and the falling velocity - pg 139 of Zhang, point "1"
    uses _opt_zhang function to do optimisation. Assume solution changes sign
    between -1 and 100 m
    
    Input arguments:
    * radius: particle radius (m)
    * kinematic_viscosity: viscosity of fluid (m^2/s)
    * gravity: accelration due to gravity (m/s^2)
    * delta_density: difference in density between particle and fluid (kg/m^3)
    * fluid_density: density of fluid (kg/m^3)

    Returns:
    * falling_velocity: velocity of particle, positive downwards (m/s)
    * drag_coefficient: empricial drag coefficent based on Reynolds number scaling (-)
    * re: Reynolds number (-)
    """
    falling_velocity = scipy.optimize.brentq(_fzhang_opt, -1.0, 100.0,
                            args=(radius, kinematic_viscosity, gravity, delta_density,
                                  fluid_density))
    
    # Recalculate drag and Re from solution velocity
    re, drag_coefficient = _fzhang_re_cd(falling_velocity, radius, kinematic_viscosity)
    
    return falling_velocity, re, drag_coefficient


def stokes_falling_velocity(radius, kinematic_viscosity, gravity, delta_density, fluid_density):
    """
    Falling velocity for low Re

    In the paper we write this in terms of dynamic viscosity.
    
    Input arguments:
    * radius: particle radius (m)
    * kinematic_viscosity: viscosity of fluid (m^2/s)
    * gravity: accelration due to gravity (m/s^2)
    * delta_density: difference in density between particle and fluid (kg/m^3)
    * fluid_density: density of fluid (kg/m^3)

    Returns:
    * falling_velocity: velocity of particle, positive downwards (m/s)
    """
    stokes_velocity = (2 * delta_density * gravity * radius**2) /  (9 * kinematic_viscosity 
                                                                    * fluid_density)
    return stokes_velocity


@np.vectorize
def boundary_layers(radius, re, pe_c, pe_t, sc, pr, fr):
    """
    Dimensional analysis of boundary layer thicknesses for falling particle
    
    Input arguments:
    * radius: particle radius (m)
    * re: Reynolds number, input as it has to be self consistently calculated (-)
    * pr: Prandtl number
    * sc: Schmidt number (-)
    * pe_c: Chemical Péclet number (-)
    * pe_t: Thermal Péclet number (-)
    * fr: Froude number (-)

    Returns:
    * delta_u: thickness of mementum boundary layer (m)
    * delta_t: thickness of thermal boundary layer (m)
    * delta_c: thickness of chemical boundary layer (m)
    """
    if fr >= 10.0:
        if re < 1.0e-2:
            delta_u = 2.0 * radius
            delta_c = 2.0 * radius
            delta_t = 2.0 * radius
        elif re < 1.0e2 and re >= 1.0e-2:               # Intermediate Re case
            delta_u = 2.0 * radius
            delta_t = 2.0 * radius                  # For T, apply low Re limit as Pe is low till Re~100
            # Prefactor for delta_c must match low and intermediate regimes at re=1e-2 and this is depends
            # on pe_c, which is re.sc, so use sc to build prefactor
            delta_c_prefac = (1.0E-2*sc)**(1/3)
            delta_c = delta_c_prefac * pe_c**(-1/3) * 2.0 * radius
        elif re >= 1.0e2:                             # High Re case
            delta_u_prefac = 10.0 # Only depends on Re... sqrt(100)
            delta_u = 10.0 * re**(-0.5) * 2.0 * radius
            delta_t_prefac = 10.0 * pr**(0.5) # But this just leads to delta_u scaling. 
            delta_t = delta_t_prefac * re**(-0.5) * (pr)**(-0.5) * 2.0 * radius
            delta_c_prefac = (1.0E-2*sc)**(1/3) * (1.0E2*sc)**(-1/3) * (1.0E2)**(1/2) * sc**(1/3)
            delta_c = delta_c_prefac * re**(-0.5) * (sc)**(-1/3) * 2.0 * radius
            
    else: # Low Fr...
        
        if re < 1.0e-2: # Same argument as for high Fr...
            delta_u = 2.0 * radius
            delta_c = 2.0 * radius
            delta_t = 2.0 * radius
        else:
            # How do I get the pre-factor? fr is a function of the falling velocity!
            delta_u = (fr/re)**(0.5) * 2.0 * radius # Assumption in para above eq. 3.11 in Inman
            delta_c = fr**(1/6) / (re**(1/6) * pe_c**(1/3)) * 2.0 * radius # eq. 3.12
            delta_t = fr**(1/6) / (re**(1/6) * pe_t**(1/3)) * 2.0 * radius # Assume t works like c.
            warnings.warn("Don't know pre-factor for in Re, low Fr")
        
    return delta_u, delta_c, delta_t


def dimensionless_numbers(radius, re, falling_velocity, kinematic_viscosity, chemical_diffusivity, 
                          thermal_diffusivity, brunt_vaisala=None):
    """
    Calculate dimensionless numbers assoceated with a falling particle
    
    Input arguments:
    * radius: particle radius (m)
    * re: Reynolds number, input as it has to be self consistently calculated (-)
    * falling_velocity: velocity of particle, positive downwards (m/s)
    * kinematic_viscosity: viscosity of fluid (m^2/s)
    * thermal_diffusivity: thermal diffusivity of fluid (m^2/s)
    * chemical_diffusivity: chemical diffusivity of fluid (m^2/s)
    * brunt_vaisala: Brunt–Väisälä frequency. Optional argument. If None (the default)
          an estimate for the outer core is used. (Hz)
    
    Returns:
    * pr: Prandtl number
    * pe_t: Thermal Péclet number (-) 
    * sc: Schmidt number (-)
    * pe_c: Chemical Péclet number (-)
    * fr: Froude number (-)
    """

    sc = kinematic_viscosity / chemical_diffusivity
    pe_c = re * sc
    pr = kinematic_viscosity / thermal_diffusivity
    pe_t = re * pr
    
    if brunt_vaisala is None:
        # Default - use PREM estimates... using prem4derg
        # implementation and denstiy at and 100m above ICB
        rho_0 = 12166.33 # density above ICB (kg/m^3
        drho_dz = -0.0005 # density grad above ICB (kg/m^2)
        icb_g = 4.4 # gravity above ICB
        brunt_vaisala = np.sqrt(-(icb_g / rho_0) * drho_dz)
    fr = np.abs(falling_velocity) / (brunt_vaisala * radius)

    return pr, pe_t, sc, pe_c, fr


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


def find_critical_radii(kinematic_viscosity, gravity, 
                        delta_density, fluid_density, thermal_diffusivity,
                        chemical_diffusivity, brunt_vaisala=None):
    """
    Find critical radii for the falling sphere where boundary layer scaling changes
    
    This should only be used as a helper for plotting the double optimisation is
    probably quite expensive
    
    Input arguments:
    * kinematic_viscosity: viscosity of fluid (m^2/s)
    * gravity: accelration due to gravity (m/s^2)
    * delta_density: difference in density between particle and fluid (kg/m^3)
    * fluid_density: density of fluid (kg/m^3)
    * thermal_diffusivity: thermal diffusivity of fluid (m^2/s)
    * chemical_diffusivity: chemical diffusivity of fluid (m^2/s)
    * brunt_vaisala: Brunt–Väisälä frequency. Optional argument. If None (the default)
          an estimate for the outer core is used. (Hz)
          
    Output:
    * re_low_radius: largest particle radius in the low Re regime (m)
    * re_int_radius: largest particle radius in the intermediate Re regime (m)
    * re_cdmax_radius: largest particle radius where drag parameterisation is valid (m)
    * low_fr_radius: largest particle radius in the low Re regime (m)
    """
    critical_re_values = [1.0E-2, 1.0E2, 3.0E5]
    critical_re_radii = []
    critical_fr_values = [10.0]
    critical_fr_radii = []
    for re in critical_re_values:
        r_crit = scipy.optimize.brentq(_re_error, 1.0E-8, 1.0E3,
                            args=(kinematic_viscosity, gravity, delta_density, fluid_density, re))
        critical_re_radii.append(r_crit)
    for fr in critical_fr_values:
        r_crit = scipy.optimize.brentq(_fr_error, 1.0E-8, 1.0E3, args=(kinematic_viscosity, gravity, 
                    delta_density, fluid_density, chemical_diffusivity, thermal_diffusivity, 
                                                                       brunt_vaisala, fr))
        critical_fr_radii.append(r_crit)
        
    return critical_re_radii[0], critical_re_radii[1], critical_re_radii[2], critical_fr_radii[0]
        
        
def _re_error(radius, kinematic_viscosity, gravity, delta_density, fluid_density, re_target):
       
        _, re, _ = self_consistent_falling_velocity(radius, kinematic_viscosity, gravity, 
                                                    delta_density, fluid_density)
        return re - re_target
    

def _fr_error(radius, kinematic_viscosity, gravity, delta_density, fluid_density, chemical_diffusivity, 
                          thermal_diffusivity, brunt_vaisala, fr_target):
       
        falling_velocity, re, _ = self_consistent_falling_velocity(radius, kinematic_viscosity, gravity, 
                                                                   delta_density, fluid_density)
        
        _, _, _, _, fr = dimensionless_numbers(radius, re, falling_velocity, kinematic_viscosity, 
                                               chemical_diffusivity, thermal_diffusivity, brunt_vaisala)
        return fr - fr_target
