#!/usr/bin/env python

import numpy as np
import scipy.optimize

from params import secinyr

def fzhang(u, r, mu, g, drho, rhol):
    """
    Zhang dynamics
    
    For given velocity outputs the Reynolds number Re and drag coefficient Cd. 
    Used to check fzhang_opt: if fzhang_opt has found the velocity that solves the equations
    then input u = output u
    
    r  = particle radius
    u  = initial velocity guess
    mu = kinematic viscosity
    Re = Reynolds number
    Cd = drag coefficient
    g  = gravity at ICB
    drho = solid-liquid density difference
    rhol = liquid density
    """
    
    Re  = np.abs(2*r * u / mu)                                              # Eqn 1  of ZX02
    Cd  = (24.0/Re) * (1.0 + 0.15*Re**0.687) + 0.42/(1.0 + 42500*Re**-1.16) # Eqn 19 of ZX02
    unum= 8.0 * g * r * drho                                                # Eqn 20 of ZX02
    uden= 3.0 * rhol * Cd                                                   # Eqn 20 of ZX02
    u   = np.sqrt(unum/uden)
    return Re, Cd, u


def fzhang_opt(u, rad, mu, g, drho, rhol):
    """
    Finds the velocity, "result", based on an initial guess u using the eqns in Zhang and Xu (2003)
    rad= particle radius
    u  = initial velocity guess
    mu = kinematic viscosity
    Re = Reynolds number
    g  = gravity at ICB
    drho = solid-liquid density difference
    rhol = liquid density
    """
    
    re = np.abs(2*rad*u/mu)

    result = (8.0 * g * rad * drho) / (3.0 * rhol * 
          ( (24.0/re ) * (1.0 + 0.15*re**0.687) + 
                                    0.42/(1.0 + 42500*re**-1.16) ) )
    result = np.sqrt(result)
    result = result - u

    return result


def calculate_boundary_layers(r, mu, g, drho, rhol, kappa, Dliq, rf, ri):
    """
    Obtain the timescales that depend on the flow velocity: 
    1. The Stokes flow timescale, tVh
    2. Thermal  diffusion in the liquid, tt_liq, based on the BL thickness. 
    3. Chemical diffusion in the liquid, tl_liq, based on the BL thickness. 
    """

    tVh  = np.zeros(len(r))
    PeT  = np.zeros(len(r))
    PeC  = np.zeros(len(r))
    deltaT = np.zeros(len(r))
    deltaC = np.zeros(len(r))
    tt_liq = np.zeros(len(r))  # Thermal diffusion time
    tl_liq = np.zeros(len(r))  # Chemical diffusion in liquid

    j = 0 # TODO: learn about enumerate
    for i in r:
        rad        = np.float(i)
        Vs         = scipy.optimize.brentq(fzhang_opt, -1.0, 100.0, args=(rad, mu, g, drho, rhol), disp=True)
        Re, Cd, uX = fzhang(Vs, rad, mu, g, drho, rhol)
        Sc = mu / Dliq
        Pr = mu / kappa
        PeT[j]     = Re * mu / kappa
        PeC[j]     = Re * mu / Dliq
     
        if Re < 1e-2: 
            deltaC[j] = 2*i
            deltaT[j] = 2*i
        if Re < 1e2 and Re > 1e-2:               # Intermediate Re case
            deltaT[j] = 2*i                      # For T, apply low Re limit as Pe is low till Re~100
            deltaC[j] = 2*PeC[j]**(-0.33333)  * 2*i
            #deltaT[j] = PeT[j]**(-0.5)      * 2*i
        if Re > 1e2:     # High Re case
            deltaC[j] = 4.5*Re**(-0.5)     * (Sc)**(-0.33333) * 2*i
            deltaT[j] = 3*Re**(-0.5)     * (Pr)**(-0.5    ) * 2*i        
            
        tt_liq[j] = deltaT[j]**2/kappa/secinyr
        tl_liq[j] = deltaC[j]**2/Dliq/secinyr

        #print(np.round(Re,9), np.round(Cd,5), np.round(uX,9), np.round(Vs,9))
        tVh[j] = (rf-ri)/Vs/secinyr
        
        j = j + 1
        
    return tVh, PeT, PeC, deltaT, deltaC, tt_liq, tl_liq


def calculate_diffusion_and_freefall_times(r, mu, g, drho, rhol, kappa, Dliq, Dsol, rf, ri):
    
    # Calculate diffusion and freefall timescales
    tt = np.zeros(len(r))  # Thermal diffusion time
    ts = np.zeros(len(r))  # Chemical diffusion in solid
    tl = np.zeros(len(r))  # Chemical diffusion in liquid
    tVl= np.zeros(len(r))  # Timescale based on Stokes velocity Vs

    # All timescales in yrs
    j = 0
    for i in r:    
        tt[j] = i**2/kappa/secinyr
        ts[j] = i**2/Dsol/secinyr
        tl[j] = i**2/Dliq/secinyr
    
        Vs    = 2.0 * drho * g * i**2 / (9.0 * mu * rhol)
        tVl[j]= (rf-ri)/Vs/secinyr
    
        j = j + 1
        
    return tt, ts, tl, tVl