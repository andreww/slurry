import numpy as np
import numba
import scipy.integrate as spi
import scipy.optimize as spo

import growth
import falling
import feo_thermodynamics

# Functions for mass conservation

@numba.vectorize()
def x_r_well_mixed (r, rp, xl):
    """Implements eqn (1) above"""
    if r < rp:
        x = 1.0
    else:
        x = xl
    return x

@numba.jit()
def intergrand_well_mixed(r, rp, xl):
    """Integrand of eqn 2 above"""
    return 4.0 * np.pi * r**2 * x_r_well_mixed(r, rp, xl)

def total_composition_well_mixed(rp, xl, r_max):
    """Integral on RHS of eqn (2)"""
    y, err, infodict = spi.quad(intergrand_well_mixed, 0.0, r_max, args=(rp, xl), full_output=1)
    
    return y

def _total_composition_well_mixed_error(xl, rp, xi, rtot):
    """
    For a given xl compute the difference between the composition and the pure
    liquid composition. When this is zero we have a consistent solution
    """
    # Calculate total composition for this configuration
    xtot = total_composition_well_mixed(rp, xl, rtot)
    xtot_pure_melt = 4/3 * np.pi * rtot**3 * xi
    error = xtot - xtot_pure_melt
    return error

def well_mixed_growth_rate(rp, xi, rtot, temperature, pressure, k0):
    """
    Compute growth rate of a particle of Fe from an FeO liquid accounting in a well mixed liquid
    
    For an Fe particle of radius rp (m) in a spherical container of radius rtot (m)
    calculate drp/dt (in m/s) assuming a well mixed liquid and fixed total composition xi 
    (mol frac Fe) pressute (in GPa) and temperature (K). We also need a prefactor for growth 
    (k0, in m/s).
    
    Returns the growth rate and the self-consistent composition aof the liquid.
    """
    xl = spo.brentq(_total_composition_well_mixed_error, 0.00000001, 0.999999999, 
                    args=(rp, xi, rtot))
    error = _total_composition_well_mixed_error(xl, rp, xi, rtot)
    v = growth.growth_velocity_feo(xl, pressure, temperature, k0)
    return v, xl, error

@numba.vectorize()
def x_r (r, rp, delta, xl, xp):
    if r < rp:
        comp = 1.0
    elif r < (rp + delta):
        comp = (xl - xp)*(r - rp)/delta + xp
    else:
        comp = xl
    return comp

@numba.jit()
def intergrand(r, rp, delta, xl, xp):
    return 4.0 * np.pi * r**2 * x_r(r, rp, delta, xl, xp)

def total_composition(rp, delta, xl, xp, r_max):
    # specifing function discontiunities as breakpoints reduces errors here.
    y, err, infodict = spi.quad(intergrand, 0.0, r_max, args=(rp, delta, xl, xp), points=(rp, rp+delta), full_output=1)
    return y, err

def _total_composition_error(xp, rp, xi, delta, rtot, temperature, pressure, dl, k0, debug=False):
    """
    For a given cl compute the difference between the composition and the pure
    liquid composition. When this is zero we have a consistent solution
    """
    # Compute growth rate for this composition 
    v = growth.growth_velocity_feo(xp, pressure, temperature, k0)
    # This gives us the composition at the edge of the boundary layer
    # because the gradient at the boundary (and hence in the layer)
    # is set by the expulsion rate of O from the growing Fe
    xl = xp + (delta*xp)/dl * v 
    # but oxygen content has changed sign.
    # Calculate total composition for this configuration
    xtot, integration_error = total_composition(rp, delta, xl, xp, rtot)
    xtot_pure_melt = 4/3 * np.pi * rtot**3 * xi
    error = xtot - xtot_pure_melt
    if debug:
        print("Composition error:", error, "intgration error:", integration_error)
    return error

def diffusion_growth_rate(rp, xi, delta, rtot, temperature, pressure, dl, k0, debug=False):
    """
    Compute growth rate of a particle of Fe from an FeO liquid accounting for a diffusional boundary layer
    
    For an Fe particle of radius rp (m) in a spherical container of radius rtot (m)
    calculate drp/dt (in m/s) assuming the presence of a linear boundary layer of
    thickness delta (m) and total composition ci (mol frac Fe) pressute (in GPa) and
    temperature (K). We also need two material properties, the diffusivity of FeO in 
    the liquid (dl, in m^2s^-1) and prefactor for growth (k0, in m/s). We also need
    an initial guess for the liquid composition next to the particle (cl_guess).
    
    Returns the growth rate, the self-consistent composition at the interface
    and the self consistent composition at the outer side of the boundary layer.
    """
    xp, root_result = spo.brentq(_total_composition_error, 1.0E-12, 1.0-1.0E-12, 
                                 args=(rp, xi, delta, rtot, temperature, pressure, dl, k0, debug),
                                 xtol=2.0e-13, disp=True, full_output=True)
    if debug:
        print(root_result)
    v  = growth.growth_velocity_feo(xp, pressure, temperature, k0)
    if debug:
        print("Thermodynamic growth rate:", v)
    xl = xp + (delta*xp)/dl * v
    error = _total_composition_error(xp, rp, xi, delta, rtot, temperature, pressure, dl, k0)
    return v, xp, xl, error

def dy_by_dt(t, y, xin, rtotin, temperaturein, pressurein, dl, k0, gin, mu, icbr, verbose=False):
    """
    Find the growth rate and sinking rate of a particle at time t
    
    This is to be used with scipy.integrate.solve_ivp. t is the time
    (not used, everything is constant in time), y[0] is the current 
    particle radius and y[1] is the current height above the ICB.
    Returns growth rate of particle and vertical velocity of particle
    (positive is upwards). 
    """
    rp = y[0]
    z = y[1]
    
    # Composition, temperature and pressure could be functions of radius (z)
    # but they will not change during a call (i.e. during an ODE time step)
    if callable(xin):
        xi = xin(z)
    else:
        xi = xin
        
    if callable(temperaturein):
        temperature = temperaturein(z)
    else:
        temperature = temperaturein
        
    if callable(pressurein):
        pressure = pressurein(z)
    else:
        pressure = pressurein
        
    if callable(gin):
        g = gin(z)
    else:
        g = gin
        
    if callable(rtotin):
        rtot = rtotin(z)
    else:
        rtot = rtotin    
    
    if verbose:
        print('Derivative evaluation at', t, 's, at z=', z, ',rp = ', rp)
        print('At this point T =', temperature, 'K, P =', pressure, 'GPa, xi =', xi, '(mol frac Fe), g =', g, 'm/s^2')
    # Find liquid composition assuming no boundary layer (or we could do the whole thing
    # inside yet another self conssitent loop - no thanks).
    xl_no_bl = spo.brentq(_total_composition_well_mixed_error, 0.00000001, 0.999999999, 
                          args=(rp, xi, rtot))
    
    # Density calculation (with composition at previous step...)
    rho_liq, _, _, rho_hcp, _, _ = feo_thermodynamics.densities(xl_no_bl, pressure, temperature)
    delta_rho = rho_hcp - rho_liq
    
    # Falling velocity
    v_falling  = -1.0 * spo.brentq(falling.fzhang_opt, -1.0, 100.0, 
                             args=(rp, mu, g, delta_rho, rho_liq))
    # Boundary layers - last two arguments not needed - should clean this function! FIXME!
    _, _, _, _, delta, _, _ = falling.calculate_boundary_layers([rp], mu, g, delta_rho, rho_liq, 
                                                    1.0, dl, 1.0, 2.0)
    
    delta = delta[0] # Proper vectoriziation of calc boundary layers needed
    # Particle growth rate
    v, xp, xl, error = diffusion_growth_rate(rp, xi, delta, rtot, temperature, pressure, dl, k0)
    if verbose:
        print('Initial guess at liquid composition:', xl_no_bl)
        print('Gives density contrast of:', delta_rho, 'kg/m^3, falling velocity', v_falling, 
              'm/s and BL thickness', delta, 'm')
        print('Gives growth rate', v, 'm/s, boundary composition', xp, 'bulk composition', 
              xl, 'and error', error)
    
    return [v, v_falling]

def hit_icb(t, y, xi, rtot, temperature, pressure, dl, k0, g, mu, icbr, verbose=False):
    """
    Return height above inner core
    
    As this is zero when the particle hits the inner core, this can be
    patched to use as a stop condition
    """
    z = y[1]
    height = z - icbr
    return height

def dissolved(t, y, xi, rtot, temperature, pressure, dl, k0, g, mu, icbr, verbose=False):
    """
    Return the radius of the particle. 
    
    As this is zero if a particle dissolves, this can be patched to use
    as a stop condition
    """
    rp = y[0]
    return rp

def make_event(r_event):
    """
    Returns a function that can be used to trigger an event if particle is at some radous r_event (m)
    """
    def event_func(t, y, xi, rtot, temperature, pressure, dl, k0, g, mu, icbr, verbose=False):
        """
        Return height above r_event
    
        As this is zero when the particle hits the depth of interest
        """
        z = y[1]
        height = z - r_event
        return height
    return event_func
    

def falling_growing_particle_solution(start_time, max_time, initial_particle_size, initial_particle_position,
                                      xi, rtot, t, p, dl, k0, g, mu, radius_inner_core, analysis_radii=None):
    """
    Solve falling particle problem as IVP and return solution bunch object
    
    This is really just a thin wrapper around the IVP solver with some error
    checking
    """
    # We need to stop if we hit the ICB or disolve. Patch the function
    # with a 'terminal' property...
    hit_icb.terminal = True
    dissolved.terminal = True
    event_handles = [hit_icb, dissolved]
    
    # Set up depths for post-run analysis (e.g. for self consistent loop)
    if analysis_radii is not None:
        for r_event in analysis_radii:
            event_handles.append(make_event(r_event))

    # Solve the IVP
    sol = spi.solve_ivp(dy_by_dt, [start_time, max_time], [initial_particle_size, initial_particle_position], 
                    args=(xi, rtot, t, p, dl, k0, g, mu, radius_inner_core),
                    events=event_handles, dense_output=True)
    
    # Check solution
    assert sol.status >= 0, "IVP failed"
    assert sol.status == 1, "Out of time, did not hit ICB"
    
    return sol