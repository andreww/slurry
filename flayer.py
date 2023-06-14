import warnings

import numpy as np
import scipy.optimize as spo
import scipy.interpolate as spi

import particle_evolution
import feo_thermodynamics as feot
import earth_model
import nucleation
import layer_diffusion


# Functions to build and evaluate models fo the F-layer
# assuming a prescribed total composition and temperature
# profile. 


def evaluate_flayer(tfunc, xfunc, pfunc, gfunc, start_time, max_time,
                    k0, dl, k, mu, i0, surf_energy, wetting_angle, hetrogeneous_radius,
                    nucleation_radii, analysis_radii, radius_inner_core, 
                    radius_top_flayer, verbose=False, silent=False):
    """
    Create a single solution for the F-layer assuming non-equilibrium growth and falling
    of iron crystals
    
    This involves integration of the results from falling-growing crystal calculations with the
    integration being with respect to the nucleation radius. We thus need to perform the calculations
    at a set of (provided) nucleation radii for the integration. Each of thse is assoceated with 
    a nucleation rate, which is calculated first. We calculate the density of falling particles (and thus their seperation)
    at a number of analysis radii. Finally, we calcuate the new temperautre and liquid composition
    profiles consistnet with the solid production. In general these will not be self consisent as
    they were not used to compute the crystal nucleation, growth or falling.
    
    Input arguments:
    
    tfunc: function returing temperature in K as a function of radius in m (callable), or scalar (K)
    xfunc: function returing total composition in mol. frac Fe as a function of radius in m 
           (callable), or scalar (in mol. frac Fe)
    pfunc: function returing pressure in GPa as a function of radius in m (callable), or scalar (GPa)
    gfunc: function returning acceleration due to gravity in m/s as a function of radius in m (callable)
           or scalar (m/s)
    start_time: initial time condition for IVP solver (s)
    max_time: maximum time for the IVP solver (s)
    k0: prefactor for particle growth rate (m/s)
    dl: diffusivity of oxygen in liquid iron (UNITS?)
    k: thermal conductivity (W/m/K)
    mu: kinematic viscosity of liquid iron(UNITS)
    i0: prefactor for CNT (s^-1 m^-3)
    surf_energy: surface energy for CNT (J m^-2)
    nucleation_radii: The radii where we explicity consider nuclation to take place. These are used
        as integration points for a numerical integration over the whole nucleating layer so we need
        enough of thse to converge. However, as the solution seems to be well behaved O(10) seems to
        be enough. A 1D array of radii (measured outwards from the center of the Earth) in m.
    nucleation_rates: The nucleation rate (in events per m^3 per s) at each nucleation_radii. 1D array
    analysis_radii: locations where the particle density and other output parameters are calculated.
        1D array in m. No particular relationship with nucleation radii assumes. This is also the place
        where the self conssitency of the solution is checked.
    radius_inner_core: inner core boundary radius (m)
    radius_top_flayer: radius to top of F-layer (m)
    verbose: optional bool default false. If true print lots of output
    silent: optional bool default false. If true suppress all output. Default should be sensible as a log of a run to a terminal.
        
    Returns:
    solutions: list of IVP solution objects, one per nucleation_radii
    particle_densities: array of total particle number densities (particles per m^3) evaluated at the
        analysis_radii
    calculated_interaction_radii: mean seperation between particles, evaluated at the analysis_radii (in m)
    icb_growth_rate: calculated growth rate of the inner core (in m/s)
    solid_vf: volume fraction solid evaluated at each analysis_radii
    particle_radius_unnormalised: 2D numpy array of size (analysis_radii, nucleation_radii) giving 
        giving the crystal radius measured at an analysis radius which nucleated at a nucleation radius.
    particle_radius_histogram: 2D numpy array of size (analysis_radii, nucleation_radii) giving the
        product of the particle radius and corresponding density at each analysis radius for each nucleation
        radius.
    crit_nuc_radii: CNT critical radii for the solution (m)
    nucleation_rates: CNT nucleation rates for the solution (events s^-1 m^-3)
    t_points: new evaluation of temperature on grid points (K)
    xl_points: new evaluation of iron content on gris points (mol frac Fe)
    """
    
    # Calculate nucleation rates and radii at each radius
    if not silent:
        print("Nucleation calculation")
        print(f"Prefactor: {i0:.3g} s^-1 m^-3, surface energy {surf_energy:.3g}")
        if wetting_angle == 180.0:
            print("Homogenious nucleation")
        else:
            print(f"Hetrogenious nucleation, wetting angle {wetting_angle:.3g} degrees")
    crit_nuc_radii = np.zeros_like(analysis_radii)
    nucleation_rates = np.zeros_like(analysis_radii)
    crit_nuc_energy = np.zeros_like(analysis_radii)
    if verbose or (not silent):
        print("Radius (km), P (GPa), T (K), X (mol. frac Fe), I (s^-1m^-3), r0 (m)")
    for i, r in enumerate(analysis_radii):
        crit_nuc_radii[i], nucleation_rates[i], crit_nuc_energy[i]  = nucleation.calc_nucleation(
            float(xfunc(r)), float(pfunc(r)), float(tfunc(r)), surf_energy, i0, theta=wetting_angle)
        if hetrogeneous_radius is not None:
            crit_nuc_radii[i] = hetrogeneous_radius
        if verbose or (not silent and len(analysis_radii) <= 10):
            print(f"{r/1000.0:4g} {pfunc(r):3g} {tfunc(r):4g} {xfunc(r):.3g} {nucleation_rates[i]:.3g} {crit_nuc_radii[i]:.3g}")
        elif not silent:
            if i%(len(analysis_radii)//10) == 0: 
                print(f"{r/1000.0:4g} {pfunc(r):3g} {tfunc(r):4g} {xfunc(r):.3g} {nucleation_rates[i]:.3g} {crit_nuc_radii[i]:.3g}")

    if not silent:
        print(f"Finding {len(analysis_radii)} IVP solutions")
    # Calculate an initial guess using the provided liquid compositioon
    solutions, particle_densities, growth_rate, solid_vf, \
        particle_radius_unnormalised, partial_particle_densities, \
        solid_volume_production_rate, mean_particle_velocities = integrate_snow_zone(
        analysis_radii, radius_inner_core, radius_top_flayer, nucleation_radii, 
        nucleation_rates, tfunc, xfunc, pfunc, gfunc,
        start_time, max_time, crit_nuc_radii, k0, dl, mu, verbose=verbose)
    
    liquid_density, _, _, fe_density, _, _ = feot.densities(xfunc(analysis_radii),
                                                            pfunc(analysis_radii),
                                                            tfunc(analysis_radii))
    
    # Calculate solid flux term - del rho u phi
    solid_mass_fraction = (fe_density * solid_vf) / (fe_density * solid_vf +
                                                     liquid_density * (1.0 - solid_vf))
    print("Calculating solid flux term thingy")
    print(f"solid mass fraction: {solid_mass_fraction}")
    print(f"mean velocities: {mean_particle_velocities}")
    print(f"solid densities: {fe_density}")
    print(f"r : {analysis_radii}")
    
    function_to_finite_difference = analysis_radii**2 * fe_density * \
                                    mean_particle_velocities * solid_mass_fraction
    
    flux_term = (1.0 / analysis_radii**2) * np.gradient(function_to_finite_difference,
                                                        analysis_radii)
    print(f"flux term: {flux_term}")
    
    # Solution for latent heat

    latent_heat = 0.75 * 1000000.0 # J/kg - from Davies 2015, could calculate this from the thermodynamics I think (FIXME). 0.75E6
    mass_production_rate = solid_volume_production_rate * fe_density
    heat_production_rate = mass_production_rate * latent_heat
    top_bc = tfunc(analysis_radii[-1])
    bottom_bc = 0.0
    if verbose or (not silent):
        print("Finding T to match heat production rate")
        print(f"Boundary conditions, top: {top_bc} K, bottom {bottom_bc} K/m")
    t_points_out = layer_diffusion.solve_layer_diffusion(analysis_radii, heat_production_rate, 
                                                         100.0, tfunc(analysis_radii),
                                                         top_value_bc=top_bc,
                                                         bottom_derivative_bc=bottom_bc)

    #print(f"volume production rate {solid_volume_production_rate}")
    #print(f"mass production rate {mass_production_rate}")
    #print(f"t_points_out {t_points_out}")
    #print(f"heat_production_rate {heat_production_rate}")
    #print(f"initial_t {tfunc(analysis_radii)}")
    #print(f"k {100}")

    # Solution for chemistry
    top_x_bc = xfunc(analysis_radii[-1])
    c_top = feot.mass_percent_o(top_x_bc)/100.0
    c_bottom = 0.0
    initial_c = feot.mass_percent_o(xfunc(analysis_radii))/100.0
    source_rate = initial_c * mass_production_rate
    dl = 1.0E-6
    print(f"With fake DL={dl}")
    if verbose or (not silent):
        print("Finding X to oxygen production rate")
        print(f"Boundary conditions, top: {c_top} kg(?), bottom {c_bottom} kg(?)/m")
    c_points_out = layer_diffusion.solve_layer_diffusion(analysis_radii, source_rate, 
                                                         dl*np.mean(fe_density), initial_c,
                                                         top_value_bc=c_top,
                                                         bottom_derivative_bc=c_bottom)
    #print(f"c_points_out {c_points_out}")
    #print(f"source_rate {source_rate}")
    #print(f"initial_c {initial_c}")
    #print(f"dl rho {dl*np.mean(fe_density)}")
    xl_points_out = feot.mol_frac_fe(c_points_out * 100.0)
    
    # Report
    if verbose or (not silent and len(analysis_radii) <= 10):
        print("Radius (km), P (GPa), Guess T (K), Guess X, dm/dt (kg/s), Q (W/m^3), O prod rate, Calculated T (K), Calculated X")
        for i, r in enumerate(analysis_radii):
            print(f"{r/1000.0:4g} {pfunc(r):3g} {tfunc(r):4g} {xfunc(r):4g} {mass_production_rate[i]:.3g} {heat_production_rate[i]:.3g} {source_rate[i]:.3g} {t_points_out[i]:4g} {xl_points_out[i]:4g}")
    elif not silent:
        print("Radius (km), P (GPa), Guess T (K), Guess X, dm/dt (kg/s), Q (W/m^3), O prod rate, Calculated T (K), Calculated X")
        for i, r in enumerate(analysis_radii):
            if i%(len(analysis_radii)//10) == 0: 
                print(f"{r/1000.0:4g} {pfunc(r):3g} {tfunc(r):4g} {xfunc(r):4g} {mass_production_rate[i]:.3g} {heat_production_rate[i]:.3g} {source_rate[i]:.3g} {t_points_out[i]:4g} {xl_points_out[i]:4g}")     
        
    return solutions, particle_densities, growth_rate, solid_vf, \
        particle_radius_unnormalised, partial_particle_densities, \
        crit_nuc_radii, nucleation_rates, t_points_out, xl_points_out


def analyse_flayer(solutions, integration_radii, analysis_radii, nucleation_rates, radius_inner_core,
                   particle_densities, growth_rate, solid_vf,
                   particle_radius_unnormalised, partial_particle_densities, tfunc, xfunc, 
                   pfunc, verbose=True):
    """
    Perform post-processing analysis of solution
    
    Note that we do quite a lot of analysis in the solver to reach a self consistent solution
    and the results of this analysis must be passed into this function
    """

    # Seperation betwen particles
    calculated_seperation = evaluate_particle_seperation(particle_densities, analysis_radii, verbose=verbose)

    # core growth rate in km/Myr
    growth_rate = evaluate_core_growth_rate(solutions, integration_radii, nucleation_rates, 
                                            radius_inner_core, verbose=verbose)
    
    
    # solid volume fraction ratio (calculated / equilibrium)
    vf_ratio =  solid_vf / feot.volume_fraction_solid(xfunc(analysis_radii), 
                                                      pfunc(analysis_radii), 
                                                      tfunc(analysis_radii))
    
    return calculated_seperation, growth_rate, vf_ratio
    
    
def report_all_solution_events(sol, analysis_depths):
    """
    Print out what happend to a single ODE IVP falling calculation
    
    We are careful to check when things don't happen. And check for 
    double crossings of depths (say we have upward falling) but just
    treat that as an error for now.
    """
    if not sol.t_events[0].size > 0:
        print('did not reach icb')
    else:
        assert sol.t_events[0].size == 1, "Double crossing detected"
        print('icb at t = ', sol.t_events[0][0], 's, with particle radius = ', sol.y_events[0][0][0])
    
    if not sol.t_events[1].size > 0:
        print('did not dissolve')
    else:
        print('particle dissolved at t = ', sol.t_events[1][0], 's')
    for i, r in enumerate(analysis_depths):
        if sol.t_events[i+2].size > 0:
            if sol.t_events[i+2].size != 1:
                warnings.warn( "Double crossing detected")
            print("reached r = ", r, 'm at t = ', sol.t_events[i+2][0], 's, with particle radius = ', sol.y_events[i+2][0][0])
        else:
            print('did not reach r = ', r, 'm')
    print("")
    
    
# Calculate parile seperation and 'partial' density
def partial_particle_density(ivp_solution, event_index, nucleation_rate, nucleation_volume, num_areas, verbose=True):
    """
    Evaluate the 'partial' particle density and particle growth rate at a given radius 
    
    Given a single solution to the IVP and a nucleation rate this function computes the
    partical number density for this IVP at a radius marked by an IVP "event" (i.e. at a
    solution time marked by the particle passing through a depth; this is tracked during
    the IVP solution process). These partial densities can be summed over nuclation depths
    (i.e. over IVP solutions) to give the total number density of particles. At the same
    time the growth rate of the particle is calculated. This can be used to calculate the
    solid production rate (and thus calculate the production of latent heat and oxygen at
    this depth for a self consistent solution).

    ivp_solution: solution object for IVP. Must have events attached. Must also have dense 
            output (i.e. a smooth interpolator) attached
    event_index: index into list of events in ivp_solution where analysis is to be perfomed. 
            This must correspond to the particle crossing the depth of interest and a sutible
            'event' must thus be attached to the IVP solution object. The event need not have
            been triggered (e.g. the particles could dissolve before reaching the radius of
            interest). 
    nucleation_rate: the rate of nucleation for the radius corresponding to t=0 for the IVP
            This is where the particles we are forming before they fall through our radius
            of interest (in particles per m^3 per s).
    nucleation_volume: the volume where nucleation takes place (in m^3)
    
    Returns the partial particle density in particles per m^3 nucleating in the volume
    represented by the start of the IVP. This must be integrated to find the total density.
    (See notes in the notebook re. statistical meaning of this given CNT!) and the
    production rate of solid in m^3 s^-1 per particle. 
    """
    if ivp_solution is None:
        if verbose:
            print("No ivp solution - particle not formed")
        return 0.0, 0.0, 0.0
    
    # Calculate the average time between nucleation events, this is the 'waiting time'
    # of Davies et al. 2019 and includes a factor of 1/2 to account for half of the 
    # particles reaching r_c then dissolving.
    tau = 1.0/(2.0*nucleation_rate*nucleation_volume)
    delta_t = 2.0 # Time step for FD calculation
    
    # Find the time where a particle that nucleated at t=0 reached the radius of interest
    # by searching through the IVP events. This cannot be the first (hit ICB) or second
    # (dissolved) event. We need to check even if it did dissolve as that could be below the
    # radius of interest.
    assert event_index > 1, "Cannot processes hit ICB or dissolved data"
    if ivp_solution.t_events[event_index].size > 0:
        assert ivp_solution.t_events[event_index].size == 1, "Double crossing detected"
        # NB: solution y_events indexed by event, then a list of times where the event is
        # seen (we want the first one - index 0) then a list of ys (i.e. particle radius, position)
        # and want the position which is index 1 
        analysis_radius = ivp_solution.y_events[event_index][0][1]
        particle_radius = ivp_solution.y_events[event_index][0][0]
        analysis_time = ivp_solution.t_events[event_index][0]
        analysis_area = (4.0*np.pi*analysis_radius**2)/num_areas

        # We'll take the distances between this particle (nucleated at t=0) and the
        # one before (nucleated at t = -tau) and the one after (t = tau). Because we 
        # have a steady state solution the whole IVP solution is just shifted in time
        # so we can do the analysis from just this solution and use the dense output
        # to get the distance We just need to be careful about falling off the ends of
        # the solution. 
        assert ivp_solution.sol(analysis_time)[1] == analysis_radius, "event / interpolator missmatch"
        if (analysis_time - tau) > 0.0:
            distance_above = ivp_solution.sol(analysis_time - tau)[1] - analysis_radius
            radius_before = ivp_solution.sol(analysis_time - delta_t)[0]
            travel_time = tau
        elif (analysis_time - delta_t) > 0.0:
            distance_above = ((ivp_solution.sol(analysis_time - delta_t)[1] - analysis_radius) / delta_t) * tau
            radius_before = ivp_solution.sol(analysis_time - delta_t)[0]
            travel_time = delta_t
        else:
            if verbose:
                print("cannot process if next particle has yet to form")
            return 0.0, 0.0, 0.0
        if (analysis_time + tau) < ivp_solution.t[-1]:
            distance_below = analysis_radius - ivp_solution.sol(analysis_time + tau)[1]
            radius_after = ivp_solution.sol(analysis_time + delta_t)[0]
            travel_time = travel_time + tau
        elif (analysis_time + delta_t) < ivp_solution.t[-1]:
            distance_below = ((analysis_radius - ivp_solution.sol(analysis_time + delta_t)[1]) / delta_t) * tau
            radius_after = ivp_solution.sol(analysis_time + delta_t)[0]
            travel_time = travel_time + delta_t
        else:
            if verbose:
                print("cannot process if previous particle has gone")
            return 0.0, 0.0, 0.0
        s_v = (0.5 * (distance_below + distance_above))
        particle_volume_growth_rate = ((4/3) * np.pi * (radius_after**3 - radius_before**3)) / (2.0 * delta_t) 
        partial_density = 1/(analysis_area * s_v) # /m^3 - see notebook!
        particle_velocity = (distance_below + distance_above) / travel_time
        if verbose:
            print("Nucleation rate = ", nucleation_rate, "nuc_vol = ", nucleation_volume)
            print("At time t = ", analysis_time, "s, and tau = ", tau, "s")
            print("Previous particle is", distance_below, "m below, next particle is", distance_above, "m above")
            print("s_v = ", s_v, "s_h = ", np.sqrt(analysis_area), "m")  
            print("Partial particle densituy is", partial_density, "particles / m^3") 
            print(f"Particle growth rate {particle_volume_growth_rate:.3g} m^3/s")
    else:
        # No particles at this depth (above nucleation depth or dissolved)
        # partial density is zero
        partial_density = 0.0
        particle_volume_growth_rate = 0.0
        particle_velocity = 0.0
        if verbose:
            print("No event data (e.g. dissolved) so partical density is zero")
        
    return partial_density, particle_volume_growth_rate, particle_velocity


# Total particle density and solid volume fraction calculation
def evaluate_partcle_densities(solutions, analysis_depths, integration_depths, nucleation_rates, 
                               radius_inner_core, radius_top_flayer, num_areas=1000000000,
                               verbose=True):
    # FIXME: other parameters should be arguments
    if verbose:
        print("ODE solved for all nuclation depths... calculating integrals over nuclation depth for particle density")
    particle_densities = np.zeros_like(analysis_depths)
    solid_vf = np.zeros_like(analysis_depths)
    solid_volume_production_rate = np.zeros_like(analysis_depths)
    partial_particle_densities = np.zeros((analysis_depths.size, integration_depths.size))
    particle_radius_unnormalised = np.zeros((analysis_depths.size, integration_depths.size))
    mean_particle_velocities = np.zeros_like(analysis_depths)
    for i, analysis_r in enumerate(analysis_depths):
        analysis_index = i + 2
        # Particle density at this depth is 'just' the partial density
        # (density from nucleation depth) integrated over nuclation depths
        # It's okay to integrate nuc depths below int depth as this will
        # return zero density. This is a 1D integral (see notebook)
        partial_densities = np.zeros_like(integration_depths)
        partial_radius = np.zeros_like(integration_depths)
        particle_volume_growth_rate = np.zeros_like(integration_depths)
        partial_velocities = np.zeros_like(integration_depths)
        for j, int_r in enumerate(integration_depths):
            # Skip if this will be zero - avoid noise
            if analysis_r > int_r:
                partial_densities[j] = 0.0
                next
                
            nuc_rate = nucleation_rates[j]
            nuc_area = (4.0*np.pi*int_r**2)/num_areas
            if j == 0:
                # Innermost layer
                top = integration_depths[j] + (0.5 * (integration_depths[j+1] - integration_depths[j]))
                bot = radius_inner_core
                
            elif (j + 1) == integration_depths.size:
                # Outermost layer
                top = radius_top_flayer
                bot = integration_depths[j] - (0.5 * (integration_depths[j] - integration_depths[j-1]))
                
            else:
                # Inside - top and bottom are half way between point and point above or below.
                top = integration_depths[j] + (0.5 * (integration_depths[j+1] - integration_depths[j]))
                bot = integration_depths[j] - (0.5 * (integration_depths[j] - integration_depths[j-1]))
             
            nuc_height = top - bot
            nuc_vol = nuc_area * nuc_height
            if verbose:
                print("\nPartial density calc for r =", analysis_r, "with nuc at r =", int_r)
            partial_densities[j], particle_volume_growth_rate[j], partial_velocities[j] = partial_particle_density(solutions[j],
                                            analysis_index, nuc_rate, nuc_vol, num_areas, verbose=verbose)
            
            # Put radius at this radius and nuc radius in radius histogram
            if solutions[j] is None:
                particle_radius_unnormalised[i,j] = 0.0
                partial_particle_densities[i,j] = 0.0
                partial_radius[j] = 0.0
            elif solutions[j].t_events[analysis_index].size > 0:
                # Triggered event - no check for double crossing as partial_particle_density will have done this
                particle_radius_unnormalised[i,j] = solutions[j].y_events[analysis_index][0][0]
                partial_particle_densities[i,j] = partial_densities[j]
                partial_radius[j] = particle_radius_unnormalised[i,j]
            else:
                # Melted etc
                particle_radius_unnormalised[i,j] = 0.0
                partial_particle_densities[i,j] = 0.0
                partial_radius[j] = 0.0
            
        # Number density of particles at this radius
        particle_density = np.trapz(partial_densities, integration_depths)
        #particle_density = np.sum(partial_densities)
        if verbose:
            print("\nTotal particle density at r = ", analysis_r, "is", particle_density, "particles per m^3")
        particle_densities[i] = particle_density
        
        # Solid volume fraction of particles at this radius - this is partial number density multiplied
        # by particle volume (which gives the solid volume fraction of particles that formed at R_N),
        # then summed over all R_Ns. No issue with layers where particles dissolved or from below our
        # analysis radius (outer loop) as these get set to zero in both terms.
        solid_vf[i] = np.sum(4/3*np.pi*partial_radius**3 * partial_densities)
        if verbose:
            print("Solid volume fraction at r = ", analysis_r, " is ", solid_vf[i])
            
        # Calculate total solid volume production rate. This is summed over nucleation points
        solid_volume_production_rate[i] = np.sum(particle_volume_growth_rate * partial_densities)
        if verbose:
            print(f"Solid production rate is {solid_volume_production_rate[i]} m^3 s^-1 / m^3")
            
        # Solid flux and velocity
        # Take the mean velocity as the sum of the particle velocities time the 
        # number density normalised by the sum of the number densities
        mean_particle_velocities[i] = np.mean((partial_densities * partial_velocities) /
                                          np.sum(partial_densities))
        
    return particle_densities, solid_vf, particle_radius_unnormalised, \
                   partial_particle_densities, solid_volume_production_rate, \
                   mean_particle_velocities


def evaluate_particle_seperation(particle_densities, analysis_depths, verbose=True):

    # Evaluate seperation of particles
    calculated_seperation = np.zeros_like(analysis_depths)
    for i, rad in enumerate (analysis_depths):
        rho = particle_densities[i]
        if rho > 0.0:
            sep = (3.0/(4.0*np.pi*rho))**(1.0/3.0)
        else:
            # Assign nominal seperation based on min density
            sep = np.inf
        calculated_seperation[i] = sep
        if verbose:
            print("Ar r =", rad, "m, particle density = ", rho, 'per m^3, calculated seperation radius = ', sep, 'm')
            
    return calculated_seperation
    
# Just calculate the core growth rate.
def evaluate_core_growth_rate(solutions, integration_depths, nucleation_rates, radius_inner_core, verbose=True):
    if verbose:
        print("\nODE solved for all nuclation  depths... calculating integrals over nuclation depth for inner core growth")
    
    # IC growth rate should be okay
    # We build up an array of solid volume as a function
    # of nuc depth and integrate
    particle_volumes = np.zeros_like(nucleation_rates)
    for i, sol in enumerate(solutions):
        if sol is None:
            # nucleation rate too low
            p_radius = 0.0
            particle_volumes[i] = 0.0
        elif not sol.t_events[0].size > 0:
            # Disolved before reaching ICB
            p_radius = 0.0
            particle_volumes[i] = 0.0
        elif sol.t_events[0].size == 1:
            # Particle reached ICB exactly once
            p_radius = sol.y_events[0][0][0]
            particle_volumes[i] = 4/3 * np.pi * p_radius**3
        else:
            # Impossible multiple crossing of ICB
            raise NotImplementedError
            
    area_icb = 4.0 * np.pi * radius_inner_core**2
    growth_rate = np.trapz(particle_volumes * np.nan_to_num(nucleation_rates, nan=0.0) 
                           * integration_depths**2 * 4.0 * np.pi, 
                           integration_depths) / area_icb
    
    if verbose:
        print("Inner core growth rate:", growth_rate, "m/s")
    secinMyr = 60.0*60.0*24.0*365.0*1000000.0
    growth_rate = growth_rate/1000.0 * secinMyr
    if verbose:
        print("Inner core growth rate:", growth_rate, "km/Myr (or mm/yr)")
    return growth_rate
    
    
# Run a single set of IVP problems and do the analysis
# Fitst create a interpolator for the ineteraction radius.
def integrate_snow_zone(analysis_depths, radius_inner_core, radius_top_flayer, integration_depths, 
                        nucleation_rates, tfunc, xl_func, pfunc, gfunc,
                        start_time, max_time, initial_particle_size, k0, dl, mu, verbose=True):
    """
    For a single liquid composition, run the IVPs and do the minimum analysis needed
    
    This should be called by the self-consistent solver
    """
    solutions = []
    for i, int_depth in enumerate(integration_depths):
        if (nucleation_rates[i] < 1.0E-120) or (np.isnan(nucleation_rates[i])):
            if verbose:
                print("Skipping this solution as no crystals form")
            sol = None
            
        else:
            int_depth = int_depth + 1.0E-3
            if verbose:
                print("Starting ODE IVP solver for nuclation at", int_depth, "with r0", initial_particle_size[i])
            sol = particle_evolution.falling_growing_particle_solution(start_time, max_time, initial_particle_size[i], 
                                                       int_depth, xl_func, tfunc, pfunc,
                                                       dl, k0, gfunc, mu, radius_inner_core, analysis_depths)
            assert sol.success, "No ODE solution found!"
            if verbose:
                report_all_solution_events(sol, analysis_depths)
        solutions.append(sol)
    
    particle_densities, solid_vf, particle_radius_unnormalised, \
    partial_particle_densities, solid_volume_production_rate, \
    mean_particle_velocities = evaluate_partcle_densities(
        solutions, analysis_depths, integration_depths, nucleation_rates,
        radius_inner_core, radius_top_flayer, verbose=verbose)
    
    growth_rate = evaluate_core_growth_rate(solutions, integration_depths, nucleation_rates, radius_inner_core, verbose=verbose)
    
    return solutions, particle_densities, growth_rate, solid_vf, \
           particle_radius_unnormalised, partial_particle_densities, \
           solid_volume_production_rate, mean_particle_velocities
    
    
