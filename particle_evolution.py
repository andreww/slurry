import numpy as np
import numba
import scipy.integrate as spi
import scipy.optimize as spo
import matplotlib.pyplot as plt

import growth
import falling
import feo_thermodynamics

# Functions to describe the evolution (in terms of 
# growth velocity and falling velocity) of a particle
# of Fe in a two phase region. This assumes the composition
# of the liquid is fixed (so it should get updated outside
# the particle falling ODE solution) which makes things
# easer and quicker, and avoids the 'interaction radius'

def falling_growing_particle_solution(start_time, max_time, initial_particle_size, initial_particle_position,
                                      xl, t, p, dl, k0, g, mu, radius_inner_core, analysis_radii=None):
    """
    Solve the coupled ODE's for a falling particle an IVP and return the solution bunch object
    
    We have the situation where a particle is falling and growing in the F-layer (or elsewhere).
    The falling rate depends on the particle size and (T-P-x dependent) density difference between
    the solid and the liquid. The growth rate depends on the falling velocity because this sets the
    thickness of a boundary layer (and thus the composition on the interface). This defines two coupled
    ODE's. We solve these using the SciPy ODE solver as the timestep has to be very short for small 
    particles but needs to increase as things get going. This is really just a thin wrapper around 
    the IVP solver with some error checking.
    
    For this version of the code, we impose the liquid composition and assume a linear composition in
    the boundary layer. To maintain mass balance this means we need to wrap the IVP problem up in an
    outer loop.
    
    Input arguments:
    * start_time - initial condition for t in the IVP. Almost always 0. (s)
    * max_time - termination time for the IVP. Should be long enough to avoid termination (s)
    * initial_particle_size - initial condiction for R in the IVP. Approximatly crit nuc size (m)
    * initial_particle_position - initial condition for r in the IVP (distace from center of Earth (m)
    * xl - liquid composition. Can be function (of r) or scalar. (Mol frac Fe)
    * t - temperature. Can be function (of r) or scalar. (K)
    * p - pressure. Can be function (of r) or scalar. (GPa)
    * dl - chemical diffusivity in liquid. Scalar. (UNITS)
    * k0 - pre-exponential parameter for crystal growth. Scalar. (m/s)
    * g - acceleration due to gravity. Can be function (of r) or scalar. (m/s^2)
    * mu - viscosity of liquid. Scalar. (UNITS)
    * radius_inner_core - radius of inner core. Defines termination condition for IVP. (m)
    * analysis_radii - list of radii where t, R and r are recorded as IVP 'events'. (m) 
                       Default of None gives no events. Note analysis_radii[i] will be 
                       reported as event i+2 (i = 0 is impact with inner core, i = 1 is
                       dissolution, both terminate the solution.
    
    Output arguments:
    * sol - output object returned from IVP solver. See 
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    Briefly:
    ** sol.t - time points, t, of solution (s)
    ** sol.y - R and r (particle radius and position) of solution at each t. (both m)
    ** sol.t_events; sol.y_events - the events (see above)
    ** sol.sol - interpolator for the solution.
    """
    # We need to stop if we hit the ICB or disolve. Patch the function
    # with a 'terminal' property...
    _hit_icb.terminal = True
    _dissolved.terminal = True
    event_handles = [_hit_icb, _dissolved]
    
    # Set up depths for post-run analysis (e.g. for self consistent loop)
    if analysis_radii is not None:
        for r_event in analysis_radii:
            event_handles.append(_make_event(r_event))

    # Solve the IVP
    sol = spi.solve_ivp(_derivatives_of_ode, [start_time, max_time], [initial_particle_size, initial_particle_position], 
                    args=(xl, t, p, dl, k0, g, mu, radius_inner_core),
                    events=event_handles, dense_output=True)
    
    # Check solution
    assert sol.status >= 0, "IVP failed"
    assert sol.status == 1, "Out of time, did not hit ICB"
    
    return sol


def interpolate_particle_evolution(sol, xl, t, p, dl, k0, g, mu, numpoints=500, inttimes=None):

    if inttimes is not None:
        times = inttimes
    else:
        # Interpolate solution (using 5th order polynomial interpolation)
        times = np.linspace(sol.sol.ts[0], sol.sol.ts[-1], numpoints)
    
    rps = sol.sol(times)[0]
    lps = sol.sol(times)[1]
    
    # second order finite difference to find growth rate at these times
    v_growth = np.gradient(rps, times)
    v_falling = np.gradient(lps, times)
    
    initial_position = sol.y[1][0]
    initial_radius = sol.y[0][0]
    
    # Make arrays of input parameters (or optimised liquid composition)
    # for each point where we have time, depth and radius of particle
    if callable(xl):
        xl = xl(lps)
    else:
        xl = np.ones_like(lps) * xl
        
    if callable(t):
        temperature = t(lps)
    else:
        temperature = np.ones_like(lps) * t
        
    if callable(p):
        pressure = p(lps) 
    else:
        pressure = np.ones_like(lps) * p
        
    if callable(g):
        gravity = g(lps)
    else:
        gravity = np.ones_like(lps) * g   
    
    
    rho_liq, _, _, rho_hcp, _, _ = feo_thermodynamics.densities(xl, pressure, temperature)
    delta_rho = rho_hcp - rho_liq
    
    # Should be able to solve for this without optimisation as I know falling velocity...
    # but this needs inverse of drag coeffcient calculation to get re I think.
    falling_velocity, drag_coefficient, re, pe_t, pe_c, fr, delta_u, delta_c = \
        falling.zhang_particle_dynamics(rps, mu, gravity, 
                            delta_rho, rho_liq, 100.0, dl, warn_peclet=False)
    
    # Should also be able to get xp from boundary layer analysis without optimisation,
    # but need to look at equations
    xp = np.zeros_like(xl)
    growth_velocity = np.zeros_like(xl)
    for i, (xl_i, delta_c_i, temperature_i, pressure_i) in enumerate(zip(xl, delta_c, temperature, pressure)):
        growth_velocity[i], xp[i] = growth.diffusion_growth_velocity(xl_i, delta_c_i, pressure_i, temperature_i, dl, k0)
        
    return times, rps, v_growth, lps, v_falling, re, pe_c, fr, delta_c, xl, xp

        
        
def plot_particle_evolution_time(sol, xl, t, p, dl, k0, g, mu, 
                                 include_solution_points=False):
    
    times, rps, v_growth, lps, v_falling, re, pe_c, fr, \
        delta_c, xl, xp = interpolate_particle_evolution(sol, xl, t, p, dl, k0, g, mu)
    
    times = times / (60*60*24) # Days seems like a generaly sensible unit for us.

    # Plot
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(12,8), sharex='col')
    fig.subplots_adjust(hspace=0, wspace=0.1)
    
    ax = axs[0,0]
    ax.plot(times, lps/1000)
    if include_solution_points:
        ax.plot(sol.t/ (60*60*24), sol.y[1]/1000, 'bo')
    ax.set_ylabel('Particle position (km)')
    
    ax = axs[0,1]
    ax.plot(times, -1*v_falling)
    ax.set_ylabel('Particle velocity (m/s)')
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position("right")
    
    ax = axs[1,0]
    ax.plot(times, rps)
    if include_solution_points:
        ax.plot(sol.t/ (60*60*24), sol.y[0], 'bo')
    ax.set_ylabel('particle radius (m)')
    
    ax = axs[1,1]
    ax.plot(times[1:], v_growth[1:])
    ax.set_ylabel('Growth velocity (m/s)')
    ax.yaxis.set_major_formatter(_sciformat)
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position("right")


    ax = axs[2,0]
    ax.plot(times[1:], re[1:], label="Re")
    ax.plot(times[1:], pe_c[1:], label="Pe_c")
    ax.plot(times[1:], fr[1:], label="Fr")
    ax.set_yscale("log")
    ax.set_ylabel('Re, Fr, Pe_c (-)')
    ax.legend()
       
    ax = axs[2,1]
    ax.plot(times, delta_c/rps)
    ax.set_ylabel('Scaled BL thickness (-)')
    ax.set_yscale("log")
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position("right")

    ax = axs[3,0]
    ax.plot(times, delta_c)
    ax.set_ylabel('Physical BL thickness (m)')
    ax.set_xlabel('Time (days)')

    
    ax = axs[3,1]
    ax.plot(times, xp, label="at boundary")
    ax.plot(times, xl, label="in bulk")
    ax.set_ylabel('Composition \n (mol frac Fe)')
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position("right")
    ax.legend()
    

    ax.set_xlabel('Time (days)')

    plt.show()



# ODE function
# This is the function used by the IVP solver to integrate the ODE. This basically 
# takes the time (t), radius of the particle (R) and it's position (r) and must 
# return dR/dt and dr/dt (both happen to be velocities) as a list. R and r are
# packed into a list on input. The other arguments set the conditions.
def _derivatives_of_ode(t, y, xln, temperaturein, pressurein, dl, 
                        k0, gin, mu, icbr, verbose=False):
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
    
    # Evaluate composition of liquid, temperature and pressure. All could be
    # functions of radius (z) but they will not change during a call (i.e. 
    # during an ODE time step)
    if callable(xln):
        xl = float(xln(z)) # Not sure why this is single element numpy array
    else:
        xl = xln
        
    if callable(temperaturein):
        temperature = temperaturein(z)
    else:
        temperature = temperaturein
        
    if callable(pressurein):
        pressure = float(pressurein(z)) # Not sure why this is single element numpy array
    else:
        pressure = pressurein
        
    if callable(gin):
        g = gin(z)
    else:
        g = gin   
    
    if verbose:
        print('Derivative evaluation at', t, 's, at z=', z, ', rp = ', rp)
        print('At this point T =', temperature, 'K, P =', pressure, 'GPa, xl =', xl, '(mol frac Fe), g =', g, 'm/s^2')
        print(type(pressure), type(xl))
        
    
    # Density calculation
    rho_liq, _, _, rho_hcp, _, _ = feo_thermodynamics.densities(xl, pressure, temperature)
    delta_rho = rho_hcp - rho_liq
    
    # Falling velocity. Needs self consistent solution as the falling velocity
    # depends on the Reynolds number, which depends on the velocity...
    # this also calculates the boundary layer thickness from scaling analysis.
    # we don't need all output and the thermal diffusivity is irrelevent.
    v_falling, _, _, _, _, _, _, delta = falling.zhang_particle_dynamics(
        rp, mu, g, delta_rho, rho_liq, 1.0, dl, warn_peclet=False) 
    v_falling  = -1.0 * v_falling # V = -dr/dt
    
    # Particle growth rate. This needs a self consistent solution as it depends on the composition
    # at the interface, which depends on the growth rate and boundary layer thickness. We optimise for
    # the composition at the inside of the boundary laer
    v, xp = growth.diffusion_growth_velocity(xl, delta, pressure, temperature, dl, k0)
    
    return [v, v_falling]





# IVP event handling
# These functions either act as or return functions that act as event triggers for the
# IVP solver. Events are triggered when any of these return zero. Call sig must match 
# ODE function (but these are all very simple events and ignore most of the arguments).
def _hit_icb(t, y, xl, temperature, pressure, dl, k0, g, mu, icbr, verbose=False):
    """
    Return height above inner core
    
    As this is zero when the particle hits the inner core, this can be
    patched to use as a stop condition
    """
    z = y[1]
    height = z - icbr
    return height


def _dissolved(t, y, xl, temperature, pressure, dl, k0, g, mu, icbr, verbose=False):
    """
    Return the radius of the particle. 
    
    As this is zero if a particle dissolves, this can be patched to use
    as a stop condition
    """
    rp = y[0]
    return rp


def _make_event(r_event):
    """
    Returns a function that can be used to trigger an event if particle is at some radous r_event (m)
    """
    def event_func(t, y, xl, temperature, pressure, dl, k0, g, mu, icbr, verbose=False):
        """
        Return height above r_event
    
        As this is zero when the particle hits the depth of interest
        """
        z = y[1]
        height = z - r_event
        return height
    return event_func

def _sciformat(x, pos=None):
    if x == 0:
        return "0.0"
    scistr = "{:E}".format(x)
    vals = scistr.split('E')
    fmttick = "${:.1f}".format(float(vals[0])) + r"\times 10^{" + "{}".format(int(vals[1])) + "}$"
    return fmttick