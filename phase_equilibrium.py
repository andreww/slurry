import numpy as np
import numba

@numba.jit(nopython=True)
def phase_relations_komabayashi2004(t, x):
    """
    Returns the phase proportions and compositions for Fe-FeO
    
    This is a simple implementation of the 330 GPa binary phase
    diagram from Figure 6 of Komabayashi (2014) assuming the
    liquidus is a straight line, and Fe and FeO are solid. For
    a given temperature (t, in Kelvin) and bulk composition (x, mass
    fraction oxygen), the function will return the mass fraction liquid
    (phi_lq), the mass fraction Fe (phi_fe), the mass fraction FeO 
    (phi_feo), and the liquid composition (as a mass fraction, x_lq).
    
    This version only works for x < eutectic
    """
    fe_melt = 6400.0
    eutectic_t = 4400.0
    eutectic_x = 0.09 # Wt frac, 9 wt%

    # Check input... x > 0 and < 1, t positive etc.
    assert (t > 0), "Temperature must be positive"
    assert ((x >= 0.0) and (x <= eutectic_x)), "Wt fraction must be between 0 and eutectic"
    
    # liquidus_grad = dc/dTl
    # liquidus_int  = 
    
    liquidus_grad = eutectic_x / (eutectic_t-fe_melt) # wt frac / K
    liquidus_int = eutectic_x - liquidus_grad * eutectic_t  
    x_liquidus = liquidus_grad * t + liquidus_int
    t_liquidus = fe_melt + x * (eutectic_t - fe_melt) / (eutectic_x)
    if t <= eutectic_t:
        # Just solid...
        x_fe = 0.0
        x_lq = np.nan
        x_feo = 0.22270
        phi_fe = (x - x_feo) / (x_fe - x_feo)
        phi_feo = 1.0 - phi_fe
        phi_lq = 0.0
    elif x >= x_liquidus:
        # RHS of liquidus, just liquid
        x_fe = 0.0
        x_lq = x
        x_feo = 0.0
        phi_fe = 0.0
        phi_lq = 1.0
        phi_feo = 0.0
    else:
        # Must be two phase. Lever rule
        x_fe = 0.0
        x_lq = x_liquidus
        x_feo = 0.0
        phi_fe = (x - x_lq) / (x_fe - x_lq)
        phi_lq = 1.0 - phi_fe
        phi_feo = 0.0
    
    return(x_fe, x_lq, x_feo, phi_fe, phi_lq, phi_feo, x_liquidus, t_liquidus)


@numba.vectorize([numba.float64(numba.float64, numba.float64)],
                nopython=True)
def total_solid_fraction(t,x):
    "Just a helper wrapper (for making graph)"
    x_fe, x_lq, x_feo, phi_fe, phi_lq, phi_feo, x_liquidus, t_liquidus = \
    phase_relations_komabayashi2004(t,x)
    return(phi_fe+phi_feo)


@numba.vectorize([numba.float64(numba.float64, numba.float64)],
                nopython=True)
def liquid_o_wt_fraction(t,x):
    "Just a helper wrapper (for making graph)"
    x_fe, x_lq, x_feo, phi_fe, phi_lq, phi_feo, x_liquidus, t_liquidus = \
    phase_relations_komabayashi2004(t,x)
    return(x_lq)


def volume_fraction_solid(phi_fe, phi_lq, x_lq):
    """
    Convert from weight fraction Fe to volume fraction Fe
    
    Just work out the volume of a kg of stuff, then
    return the volume fraction that is sold. We need 
    to know the density of Fe and the density of an FeO
    melt. The latter comes from Figure 7d of Komabayashi
    (2014) - so is at 330 GPa and 5000 K. Assume that this
    does not change much with P or T. We also need the density
    of pure liquid Fe (which comes from table IV of Alfe; 2002)
    and pure solid Fe (where we assume a fraction change in
    volume on melting of 0.02, see Figure 2 of the same paper).
    """
    density_pure_fe_melt = 12844 + 30*((13315-12844)/50) # kg/m^3
    density_pure_fe_solid = density_pure_fe_melt * 1.02 # kg/m^3
    density_melt = (1.0+(x_lq*(-0.825/0.22270))) * density_pure_fe_melt # Fig 7d
    volume_melt = phi_lq / density_melt
    volume_fe_solid = phi_fe / density_pure_fe_solid
    vol_frac_solid = volume_fe_solid / (volume_fe_solid + volume_melt)
    return vol_frac_solid