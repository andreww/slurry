import numpy as np

# Functions to setup and analyse the layer without the particle calculation
#
# This module provides functions to create the temperature and composition
# functions and function factories needed as input to the slurry layer
# problem. It also provides functions to investigate the properties of the
# layer (e.g. its stability against convection and its heat budget).


def estimate_brunt_vaisala_frequency(r_top, r_bot, tfunc, atfunc, xfunc, gfunc, pfunc):
    """
    Derive an estimate for the Brunt Vaisala frequency for a layer by taking differences
    
    We assume we can write:
    
       N_BV^2 = -(g_m/rho_m) (rho'_t - rho'_b)/(R_t - R_b)
       
    where t, m and b are at the top, middle and bottom of the layer and rho' is the difference
    in density between an adiabatic state and the real density. For the adiabatic state we follow
    the temperature down an adiabat and assume the composition is well mixed (i.e. assume it is 
    the same as the bulk outer core).
    
    Arguments
    r_top: radius (in m) of top of the layer 
    r_bot: radius (in m) of bottom of the layer
    tfunc: function that returns the temperature (K)
    atfunc: function that returns adiabatic temperature profile through the layer (K)
    xfunc: function that returns the liquid composition of the layer (mol. frac. Fe)
    gfunc: function that returns the gravity through the layer (m/s)
    pfunc: function that returns the pressure through the layer (GPa)
    
    Returns: (Nbv, N2)
    Nbv: Brunt Vaisala frequency (Hz)
    N2: Squared Brunt Vaisala frequency 
    If N22 is negative we return a complex frequency.
    """
    r_m = r_bot + (r_top - r_bot)/2.0
    # Don't need the absolute density at the top of the layer
    rho_b, _, _, _, _, _ = feot.densities(xfunc(r_bot), pfunc(r_bot), tfunc(r_bot))
    rho_m, _, _, _, _, _ = feot.densities(xfunc(r_m), pfunc(r_m), tfunc(r_m))
    # Our reference at the top is the same as the real density (because we construct
    # the layer that way), the reference at the bottom follows the adiabatic temperature
    # but uses the composition from the top (well mixed)
    ref_rho_b, _, _, _, _, _ = feot.densities(xfunc(r_top), pfunc(r_bot), atfunc(r_bot))
        
    N2 = -1.0 * (gfunc(r_m) / rho_m) * (0.0 - (rho_b - ref_rho_b)) / (r_top - r_bot)
    
    if N2 >= 0.0:
        Nbv = np.sqrt(N2)
    else:
        Nbv = np.sqrt((N2 + 0j))
    return Nbv, N2


def fit_quad_func_boundaries(r_icb, r_ftop, v_icb, v_ftop):
    """
    Find a quadratic function to match two values and zero derivative
    
    Find the coefficents of a quadratic equation for particualar values 
    (e.g. temperature) at the top and bottom of a layer and for zero
    derivative on the inner boundary, then return the function.
    
    Finds a, b and c for:
    
        Y(r) = ar^2 + br + c; Y(r_icb) = v_icb; Y_(r_ftop) = v_ftop
        dY/dr |_r_icb = 2ar + b = 0
    
    a, b and c come from solving:
    
        | r_icb^2    r_icb    1 |  | a |     | v_icb  |
        | r_ftop^2   r_ftop   1 |  | b | =   | v_ftop |
        | 2 r_icb      1      0 |  | c |     | 0      |
        
    returns a function (of r)
    """
    denom = (r_icb - r_ftop)**2
    a = (v_ftop - v_icb)/denom
    b = (-2.0*r_icb * (v_ftop - v_icb))/denom
    c = r_icb**2 * v_ftop - 2 * r_icb * r_ftop * v_icb + r_ftop**2 * v_icb
    c = c / denom
    def quad_func(r):
        v = a * r**2 + b * r + c
        return v
    return quad_func