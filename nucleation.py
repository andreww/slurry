import numpy as np
import feo_thermodynamics

"""
Python module to handle nucleation

This module contains the functions used to calculate the nucleation rate in a
Fe-O liquid. It follws the 'well mixed' approximation in new_nucleation.ipynb
(i.e. accounts for the change in chemical potential of Fe in the liquid with 
composition, but neglects any contribution to the change in oxgygen content
of the liquid as the neucleus forms. For fixed undercooling (below the liquidus)
this gives a very small decrease in the critical radius and increase in the 
nucleation rate compared to the approximation in Davies et al. 2019.
"""

def calc_nucleation(x, p, t, gamma, i0, theta=180.0):
    """
    Return the nucleation rate for liquid Fe-O using CNT and the well mixed approximation
    
    The volumetic energy is the difference in chemical potential of Fe in HCP Fe and 
    Fe-O liquid of some specified compositon at some temperature and pressure calculated
    using the Komabashi thermodynamic model. Input parameters are a surface energy term
    and a pre-factor.
    
    Input parameters:
    x - liquid composition (mol frac Fe)
    p - pressure (GPa)
    t - temperature (K)
    gamma - surface energy (J m^-2)
    i0 - nucleation pre-factor (s^-1 m^-3)
    theta - wetting angle (degrees). Default of 180 corresponds to homogenious nucleation
    
    Returns:
    rc - critical radius (m)
    i - nucleation rate (s^-1 m^-3)
    both are set to np.nan for cases above the liquidus
    """
    g_sl = well_mixed_gsl(x, p, t)
    rc, gc = well_mixed_nucleation(gamma, g_sl, theta)
    i = nucleation_rate(t, i0, gc)
    return rc, i, gc


def well_mixed_nucleation(gamma, gsl, theta=180.0):
    """
    Calculate CNT parameters assuming pure phase or well mixed liquid
    
    gamma: surface energy (J m^-2)
    gsl: difference between free energy of solid and liquid (J m^-3)
    i0: pre-factor / attempt rate (s^-1 m^-3)
    theta: wetting angle in degrees. 180 (default) corresponds to homogenious nuc
    
    if t > melting temperature, returns np.nan
    
    returns critical radii (in m), free energy barrier (in J), 
    nucleation rate (s^-1 m^3), and waiting time (s m^3)
    """
    if gsl >= 0.0:
        # On the liquidus or above Tm. return np.nan, this propogates
        # and is handled by the F-layer solver.
        rc = np.nan
        gc = np.nan
    else:
        theta = np.radians(theta)
        s_theta = (2.0 - 3.0*np.cos(theta) + np.cos(theta)**3) / 4.0
        rc = -2*gamma / gsl
        gc = (4.0/3.0 * np.pi * rc**3 * gsl + 4.0 * np.pi * rc**2 * gamma) * s_theta
    
    return rc, gc


def nucleation_rate(t, i0, gc):
    """
    Calculate nucleation rate for CNT given free energy at critical radius
    
    t: temperature (in K)
    i0: pre-factor / attempt rate (s^-1 m^-3)
    gc: free energy of critical radius (J)
    
    returns I (in s^-1 m^-3)
    """
    kb = 1.38064852E-23
    # nuc rate
    i = i0 * np.exp(-gc / (t * kb))
    return i


def waiting_time(t, prefac, gc):
    """
    Calculate waiting time for CNT given free energy at critical radius
    prefac: pre-factor / attempt rate (s m^3) [NB: inverse of i0 above!!!]
    gc: free energy of critical radius (J)
    
    returns tauv (in s m3)
    """
    # FIXME: this is likly to overflow in the exp, even if we precompute
    # prefac = 1/(2 i0). Break up the exp bit?
    kb = 1.38064852E-23
    
    # waiting time
    tauv = prefac*np.exp(gc / (t * kb))
    return tauv


def well_mixed_gsl(x, p, t):
    """
    Free energy change when making solid Fe in liquid FeO
    
    Assumes no O in solid, ideal mixing in liquid...
    
    """
    # All in J/mol
    _, g_fe, _ = feo_thermodynamics.solid_free_energies(x, p, t)
    g_lq = feo_thermodynamics.fe_liquid_chemical_potential(x, p, t)
    _, mol_vol_solid, _ = feo_thermodynamics.solid_molar_volume(1.0, p, t)
    _, partial_mol_vol_liquid, _ = feo_thermodynamics.liquid_molar_volume(1.0, p, t)
    g_sl = g_fe - g_lq # J/mol
    
    g_sl = g_sl  / (mol_vol_solid * 1.0E-6) # J / m^3 of solid
    return g_sl