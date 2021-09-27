import numpy as np

def sun_d_mu(dt, tm, dhm):
    """
    Sun's approximation to the difference in chemical potential between solid and liquid
    
    Linear approximation to the difference in chemical potential between the solid
    and liquid for temperature dt K below the melting point (tm, in K) with heat of
    fusion dhm. Returns the chemical potential difference in J.
    """
    t = tm - dt
    d_mu = (tm - t) * (dhm / tm) # K13 / first para of Sun S
    return d_mu


def sun_growth_velocity(dmu, k0, t):
    """
    Sun's linear model 
    dmu is the difference in chemical potentials (solid - liquid) in J/mol
    k0 is a rate constant
    t is the absolute temperature
    """
    kB = 1.380649e-23 #J/K
    v = k0 * (1.0 - np.exp(-dmu/(params.kb * t)))
    return v