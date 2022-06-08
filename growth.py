import numpy as np
import scipy.optimize as spo
import feo_thermodynamics

"""
Python module to handle growth

General approach is to implement the simplest model from Sun et al.
(2018; Nature Materials 17:881-886, https://doi.org/10.1038/s41563-018-0174-6)
where growth rate (a velocity perendicular to a surface) is described 
by a chemical potential difference between the solid and the liquid with
a (constant) pre-factor. This does a decent job of fitting their simulation
data (but other models that make the pre-factor temperature dependent are
available). For the most part we have to guess at a pre-factor and make use
of the Komabayashi (2013) thermodynamic model to find the chemical potential
differences. This is what is implemented in 'growth_velocity_feo'. Other functions
can be used to reproduce sun (for example).
"""

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


def growth_velocity_sun(dmu, k0, t):
    """
    Sun's linear model (model 1) for the growth rate as a function of temperature
    
    dmu is the difference in chemical potentials (solid - liquid) in J/mol
    NB: FIXME - I think the equations define positive to give growth, which I think is liquid - solid...
    k0 is a rate constant in m/s
    t is the absolute temperature in K
    
    returns the growth velocity in m/s
    """
    kB = 1.380649e-23 #J/K
    v = k0 * (1.0 - np.exp(-dmu/(kB * t))) # Sun eq. 1
    return v


def growth_velocity_feo(x, p, t, k0):
    """
    Fe growth velocity in an Fe-FeO system sollowing K14 thermodynamics and S18's model 1
    
    This function returns the growth velocity of an Fe particle (expressed as a velocity
    of the interface perpendicular to the interface) from an Fe-O melt with thermodynamics
    described by Komabayashi's thermodynamic model. The approach is to find the difference
    in the Fe chemical potential in the melt and the solid and use this (toghether with an
    assumed pre-factor) with Sun's growth model. If the user keeps track of the oxygen content
    in the liquid (which should increase as the solid Fe grows) the growth rate will approach
    zero as the system approaches phase equilibrium. More complex models will allow diffusion
    boundary layers to develop in the melt, which would allow disequilibrium to be maintained.
    
    Input parameters:
    x: oxygen content of liquid in contact with solid (mol fraction Fe)
    p: pressure (in GPa)
    t: temperature (in K)
    k0: growth velocity pre-factor (in m/s)
    
    We have very little idea of appropriate values for k0 for HCP Fe under core conditions.
    Sun's simulations on FCC metals under low pressure conditions suggest values between
    130 m/s (for Pb) and 750 m/s (for Al) *may* make sense. As this just acts as a scaling
    factor we should probably take this as an unknown parameter. Atomic scale simulations 
    (with LAMMPS, using our model) could resolve this.
    
    Returns:
    v: growth velocity in m/s
    
    Positive velocities indicate the crystal grows under these conditions, negative velocity
    indicates dissolution. However, the physics of dissolution at the interface may be different 
    to the physics of growth. This would imply (at least) the need for a different k0.
    """
    # Chemical potentials
    mu_fe_solid, _, _ = feo_thermodynamics.solid_free_energies(1.0, p, t)
    mu_fe_liquid = feo_thermodynamics.fe_liquid_chemical_potential(x, p, t)
    v = growth_velocity_sun( (mu_fe_liquid - mu_fe_solid)/feo_thermodynamics.avogadro, k0, t)
    return v


def diffusion_growth_velocity(xl, delta, pressure, temperature, dl, k0):
    """
    Solve for growth rate given liquid compositon outside boundary layer and boundary
    layer thickness.
    
    Input:
    xl - liquid composition (mol frac Fe)
    delta - boundary layer thickness (m) 
    temperature - temperature (K)
    pressure - pressure (GPa)
    dl - diffusion coeffcent of O in Fe
    k0 - growth pre-factor (m/s)
    
    Returns:
    v - growth velocity (m/s)
    xp - composition at interface (mol frac Fe)
    
    """

    # Particle growth rate. This needs a self consistent solution as it depends on the composition
    # at the interface, which depends on the growth rate and boundary layer thickness. We optimise for
    # the composition at the inside of the boundary laer
    
    if temperature > 6000.0:
        print(f"Temperature is too high ({temperature} K), capping to 7000 K")
        temperature = 6000.0
    
    # Check bounds
    low_fe_err = _optimise_boundary_composition(1.0E-26, xl, delta, temperature, pressure, dl, k0)
    high_fe_err = _optimise_boundary_composition(1.0-1.0E-26, xl, delta, temperature, pressure, dl, k0)
    if np.sign(low_fe_err) == np.sign(high_fe_err):
        print("About to crash in growth rate root finding")
        print(f"xp low err = {low_fe_err}, xp high err = {high_fe_err}")
        print(f"xl = {xl}, delta = {delta}, t = {temperature}, pressure = {pressure}")
        print(f"dl = {dl}, k0 = {k0}")
    
    xp, root_result = spo.brentq(_optimise_boundary_composition, 1.0E-26, 1.0-1.0E-26, 
                                 args=(xl, delta, temperature, pressure, dl, k0),
                                 xtol=2.0e-13, disp=True, full_output=True)
    assert root_result.converged, "Failed to converge in diffusion_growth_velocity"
    # and use this to calculate the growth velocity
    v = growth_velocity_feo(xp, pressure, temperature, k0)
    
    return v, xp


def _optimise_boundary_composition(xp, xl, delta, temperature, pressure, dl, k0):
    """
    For a given xp compute the difference between the composition at the edge of
    the boundary layer and the composition of the liquid. When this is zero we 
    have a consistent solution (i.e. the boundary layer has a gradient suggested
    by the growth rate and a composition at the boundary that matches the bulk
    liquid composition
    """
    # Compute growth rate for this composition 
    v = growth_velocity_feo(xp, pressure, temperature, k0)
    
    # This gives us the composition at the edge of the boundary layer
    # because the gradient at the boundary (and hence in the layer)
    # is set by the expulsion rate of O from the growing Fe
    xl_calc = xp + (delta*xp)/dl * v 
    
    # Return the difference between the calculated and intended composition
    error = xl_calc - xl
    return error