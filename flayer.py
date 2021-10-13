import numpy as np
import scipy.optimize as spo
import scipy.interpolate as spi

import particle_evolution
import feo_thermodynamics as feot
import earth_model

# Functions to build and evaluate models fo the F-layer
# assuming a prescribed total composition and temperature
# profile. 

def setup_flayer_functions(r_icb, r_cmb, r_flayer_top, gamma, delta_t_icb, xfe_adiabatic, xfe_icb):
    """
    This defines the radial functions we are going to need to model the f-layer
    
    Our input parameters are the (total) oxygen content at the top and bottom of 
    the layer, the subadiabatic cooling through the layer, a value for gamma
    for the adiabatic mantle and a thickness of the layer. We first uuse this 
    to find the liquidus temperature at the top of the F-layer. We then find
    the CMB temperature to generate an adiabat that intersects the liquidus at the
    top of the f-layer.
    
    Arguments are:
    
    r_icb: ICB radius in m
    r_cmb: core mantle boundary radius in m
    r_flayer_top: radius of the top of the f-layer in m
    gamma: grunisen parameter for adiabatic core, dimensionless (1.5 is a typical value)
    delta_t_icb: subadiabatic temperature depression at the ICB in K, 10 K is a sensible value
    xfe_adiabatic: oxygen content (as mol frac Fe) of the adiabatic core (and top of the F-layer)
                    8-17 mol % O (i.e. 0.92-0.83) are sensible values
    xfe_icb: oxygen content (as mol frac Fe) at the bottom of the F-layer. NB: this is the total oxygen 
             content. As O partititons into the liquid the liquid will be enriched. 
    
    We end up returning a functions that can give the total oxygen content as a function
    of radius, the temperature as a function of radius, and the pressure (from PREM) as
    a function of radius. In the F-layer we assume linear temperature and composition
    profiles. These functions are (numpy) vectorized and take the radii in m. 
    """
    # Base P and g on PREM...
    prem = earth_model.Prem()
    
    # First find the liquidus temperature at the top of the F-layer... we know P (from PREM)
    # and X (from our input). NB: my PREM module works in km and does not like vector input.
    # This is quite slow and could be optimised by avoiding the double brentq calls!
    tl_top_flayer = feot.find_liquidus(xfe_adiabatic, prem.pressure(r_flayer_top/1000.0))
    
    # Now we need to work out the adiabatic temperature profile that intersects the liquidus 
    # at the top of the F-layer. 
    rho_cmb = prem.density((r_cmb-0.1)/1000.0) # -0.1 for core, not mantle rho
    rho_top_flayer = prem.density(r_flayer_top/1000.0)
    def _t_error_top_flayer(tcmb):
        """
        For a given CMB temperature, calculate the difference between the temperature
        at the top of the f-layer and the liquidus temperature. We'll need to set 
        tcmb such that this is zero!
        """
        adabat_t_top_flayer = tcmb * (rho_top_flayer/rho_cmb)**gamma
        t_error = adabat_t_top_flayer - tl_top_flayer
        return t_error
    t_cmb = spo.brentq(_t_error_top_flayer, 1000, 8000)
    
    # We can now build our function to give the adiabatic temperature
    rho_icb = prem.density(r_icb/1000.0)
    adabat_t_top_flayer = t_cmb * (rho_top_flayer/rho_cmb)**gamma
    adabat_icb = t_cmb * (rho_icb/rho_cmb)**gamma
    @np.vectorize
    def adiabatic_temperature_function(r):
        temp = t_cmb * (prem.density(r/1000.0)/rho_cmb)**gamma
        return temp
    
    # And the function to give the 'real' temperature (including a subadiabatic layer)
    # We'll base temperature on an adiabat (temperature at top of F-layer)
    # and then assume it's linear to a subadiabatic ICB temperature
    # Do note that we cannot use the ICB as a tie point as (1) the F-layer
    # is not adiabatic and (2) we are not assuming phase equilibrium at the ICB
    # which means the ICB need not be at any particular melting temperature. 
    # This is an important point (probably the most important point if we introduce
    # non-equilibrium processes).
    temperature_icb = adabat_icb - delta_t_icb
    @np.vectorize
    def temperature_function(r):
        if r >= r_cmb:
            temp = t_cmb # density is odd at disconts, just fix T outside core.
        elif r > r_flayer_top:
            temp = t_cmb * (prem.density(r/1000.0)/rho_cmb)**gamma
        else: # Will give value inside inner core, but we may need that for IVP solver...
            temp = temperature_icb + (r - r_icb)*(
                (adabat_t_top_flayer-temperature_icb)/(r_flayer_top-r_icb))
        return temp
    

    # Finally, a function to give the composition everywhere
    @np.vectorize
    def composition_function(r):
        if r > r_flayer_top:
            xfe = xfe_adiabatic
        else:
            xfe = xfe_icb + (r - r_icb)*(
                (xfe_adiabatic-xfe_icb)/(r_flayer_top-r_icb))
        return xfe
    
    @np.vectorize           
    def pressure_function(r):
        return prem.pressure(r/1000.0)
    
    @np.vectorize
    def gravity_function(r):
        return prem.gravity(r/1000.0)
    
    return temperature_function, adiabatic_temperature_function, composition_function, \
        pressure_function, gravity_function


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
            assert sol.t_events[i+2].size == 1, "Double crossing detected"
            print("reached r = ", r, 'm at t = ', sol.t_events[i+2][0], 's, with particle radius = ', sol.y_events[i+2][0][0])
        else:
            print('did not reach r = ', r, 'm')
    print("")
    
    
# Calculate parile seperation and 'partial' density
def partial_particle_density(ivp_solution, event_index, nucleation_rate, nucleation_volume, verbose=True):
    """
    Evaluate the 'partial' particle density at a given radius given a single solution to the IVP
    

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
    (See notes in the notebook re. statistical meaning of this given CNT!)
    """
    # Calculate the average time between nucleation events, this is the 'waiting time'
    # of Davies et al. 2019 and includes a factor of 1/2 to account for half of the 
    # particles reaching r_c then dissolving.
    tau = 1.0/(2.0*nucleation_rate*nucleation_volume)
    
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
        analysis_time = ivp_solution.t_events[event_index][0]
        # We'll take the distances between this particle (nucleated at t=0) and the
        # one before (nucleated at t = -tau) and the one after (t = tau). Because we 
        # have a steady state solution the whole IVP solution is just shifted in time
        # so we can do the analysis from just this solution and use the dense output
        # to get the distance
        assert ivp_solution.sol(analysis_time)[1] == analysis_radius, "event / interpolator missmatch"
        if (analysis_time - tau) < 0.0:
            print("cannot process if next particle has yet to form")
            return 0.0
        if (analysis_time + tau) > ivp_solution.t[-1]:
            print("cannot process if previous particle has gone")
            return 0.0
        distance_below = analysis_radius - ivp_solution.sol(analysis_time + tau)[1]
        distance_above = ivp_solution.sol(analysis_time - tau)[1] - analysis_radius
        partial_density = 1.0 / (0.5 * (distance_below + distance_above)) # /m^3 - see notebook!
        if verbose:
            print("Partial density calculation at r = ", analysis_radius, "m")
            print("At time t = ", analysis_time, "s, and tau = ", tau, "s")
            print("Previous particle is", distance_below, "m below, next particle is", distance_above, "m above")
            print("Partial particle densituy is", partial_density, "particles / m^3")     
    else:
        # No particles at this depth (above nucleation depth or dissolved)
        # partial density is zero
        partial_density = 0.0
        if verbose:
            print("No event data (e.g. dissolved) so partical density is zero")
        
    return partial_density


# Actually run the IVPs
def run_ivps_for_snow_zone(analysis_depths, integration_depths, radius_inner_core, nucleation_rates, 
                           tfunc, xl_func, pfunc, gfunc,
                           k0, dl, mu, start_time, max_time, initial_particle_size):
    # FIXME: other parameters should be arguments
    solutions = []
    for int_depth in integration_depths:
        print("Starting ODE IVP solver for nuclation at", int_depth)
        sol = particle_evolution.falling_growing_particle_solution(start_time, max_time, initial_particle_size, 
                                                       int_depth, xl_func, tfunc, pfunc,
                                                       dl, k0, gfunc, mu, radius_inner_core, analysis_depths)
        assert sol.success, "No ODE solution found!"
        report_all_solution_events(sol, analysis_depths)
        solutions.append(sol)
        
    return solutions


# Total particle density and solid volume fraction calculation
def evaluate_partcle_densities(solutions, analysis_depths, integration_depths, nucleation_rates, 
                               radius_inner_core, radius_top_flayer):
    # FIXME: other parameters should be arguments

    print("ODE solved for all nuclation depths... calculating integrals over nuclation depth for particle density")
    particle_densities = np.zeros_like(analysis_depths)
    solid_vf = np.zeros_like(analysis_depths)
    particle_radius_histogram = np.zeros((analysis_depths.size, integration_depths.size))
    particle_radius_unnormalised = np.zeros((analysis_depths.size, integration_depths.size))
    for i, analysis_r in enumerate(analysis_depths):
        analysis_index = i + 2
        # Particle density at this depth is 'just' the partial density
        # (density from nucleation depth) integrated over nuclation depths
        # It's okay to integrate nuc depths below int depth as this will
        # return zero density. This is a 1D integral (see notebook)
        partial_densities = np.zeros_like(integration_depths)
        partial_radius = np.zeros_like(integration_depths)
        for j, int_r in enumerate(integration_depths):
            # Skip if this will be zero - avoid noise
            if analysis_r > int_r:
                partial_densities[j] = 0.0
                next
                
            nuc_rate = nucleation_rates[j]
            nuc_area = 1000.0*1000.0
            if j == 0:
                # Innermost layer
                nuc_vol = nuc_area * (integration_depths[0] + 0.5 * (
                                      integration_depths[1] - integration_depths[0])
                                     ) - radius_inner_core 
            elif (j + 1) == integration_depths.size:
                # Outermost layer
                nuc_vol = nuc_area * radius_top_flayer - (integration_depths[-2] + 0.5 * (
                                      integration_depths[-1] - integration_depths[-2])
                                     )
            else:
                nuc_vol = nuc_area * ((integration_depths[j] + 0.5 * (
                                      integration_depths[j+1] - integration_depths[j]))
                                  - (integration_depths[j-1] + 0.5 * (
                                      integration_depths[j] - integration_depths[j-1])
                                    ))
            print("\nPartial density calc for i = ", i, "and j = ", j, "nuc_rate = ", nuc_rate, "nuc_vol = ", nuc_vol)
            print("Analysis r = ", analysis_r, "int r = ", int_r)
            partial_densities[j] = partial_particle_density(solutions[j], analysis_index, 
                                                            nuc_rate, nuc_vol, verbose=True)
            
            # Put radius at this radius and nuc radius in radius histogram
            if solutions[j].t_events[analysis_index].size > 0:
                # Triggered event - no check for double crossing as partial_particle_density will have done this
                particle_radius_unnormalised[i,j] = solutions[j].y_events[analysis_index][0][0]
                particle_radius_histogram[i,j] = particle_radius_unnormalised[i,j]*partial_densities[j]
                partial_radius[j] = particle_radius_unnormalised[i,j]
            else:
                # Melted etc
                particle_radius_unnormalised[i,j] = 0.0
                particle_radius_histogram[i,j] = 0.0
                partial_radius[j] = 0.0
            
        # Number density of particles at this radius
        particle_density = np.trapz(partial_densities, integration_depths) / nuc_area # See units note...
        print("\nTotal particle density at r = ", analysis_r, "is", particle_density, "particles per m^3")
        particle_densities[i] = particle_density
        
        # Solid volume fraction of particles at this radius - this is partial number density multiplied
        # by particle volume integrated over nucleation height. While we are here also build a grain size
        # distribution histogramme at each radius
        solid_vf[i] = np.trapz(4/3*np.pi*partial_radius**3 * (partial_densities / nuc_area), integration_depths)
        print("Solid volume fraction at r = ", analysis_r, " is ", solid_vf[i])
        
    # Evaluate seperation of particles (to allow self consistent solution)
    calculated_seperation = np.zeros_like(analysis_depths)
    for i, rad in enumerate (analysis_depths):
        rho = particle_densities[i]
        if rho > 0.0:
            sep = (3.0/(4.0*np.pi*rho))**(1.0/3.0)
        else:
            # Assign nominal seperation based on min density
            sep = np.inf
        calculated_seperation[i] = sep
        print("Ar r =", rad, "m, particle density = ", rho, 'per m^3, calculated seperation radius = ', sep, 'm')
       
    return particle_densities, calculated_seperation, solid_vf, particle_radius_unnormalised, particle_radius_histogram


# Just calculate the core growth rate.
def evaluate_core_growth_rate(solutions, integration_depths, nucleation_rates, radius_inner_core):
    # FIXME: other parameters should be arguments
    print("\nODE solved for all nuclation depths... calculating integrals over nuclation depth for inner core growth")
    
    # IC growth rate should be okay
    # We build up an array of solid volume as a function
    # of nuc depth and integrate
    particle_volumes = np.zeros_like(nucleation_rates)
    for i, sol in enumerate(solutions):
        if not sol.t_events[0].size > 0:
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
    growth_rate = np.trapz(particle_volumes * nucleation_rates * integration_depths**2 * 4.0 * np.pi, 
                           integration_depths) / area_icb
    
    secinMyr = 60.0*60.0*24.0*365.0*1000000.0
    print("Inner core growth rate:", growth_rate, "m/s")
    print("Inner core growth rate:", growth_rate/1000.0 * secinMyr, "km/Myr (or mm/yr)")
    return growth_rate
    
    
# Run a single set of IVP problems and do the analysis
# Fitst create a interpolator for the ineteraction radius.
def integrate_snow_zone(analysis_depths, radius_inner_core, radius_top_flayer, integration_depths, 
                        nucleation_rates, tfunc, xl_func, pfunc, gfunc,
                        start_time, max_time, initial_particle_size, k0, dl, mu,):
    # FIXME: other parameters should be arguments
    
    solutions = run_ivps_for_snow_zone(analysis_depths, integration_depths, radius_inner_core, 
                                       nucleation_rates, tfunc, xl_func, pfunc, gfunc,
                                       k0, dl, mu, start_time, max_time, initial_particle_size)
    
    particle_densities, calculated_seperation, solid_vf, \
        particle_radius_unnormalised, particle_radius_histogram = evaluate_partcle_densities(solutions, 
                                        analysis_depths, integration_depths, nucleation_rates, radius_inner_core, 
                                                                                             radius_top_flayer)
    
    growth_rate = evaluate_core_growth_rate(solutions, integration_depths, nucleation_rates, radius_inner_core)
    
    return solutions, particle_densities, calculated_seperation, growth_rate, solid_vf, particle_radius_unnormalised, particle_radius_histogram
    
    
# This controls the self consistnet solution
def solve_flayer(tfunc, xfunc, pfunc, gfunc, start_time, max_time, initial_particle_size,
                 k0, dl, mu, nucleation_radii, nucleation_rates, analysis_radii, radius_inner_core, 
                 radius_top_flayer, max_rel_error=0.01, max_absolute_error=0.001):
    """
    FIX THE DOCSTRING!
    Create a self consistent solution for the F-layer assuming non-equilibrium growth and falling
    of iron crystals
    
    This involves integration of the results from falling-growing crystal calculations with the
    integration being with respect to the nucleation radius. We thus need to perform the calculations
    at a set of (provided) nucleation radii for the integration. Each of thse is assoceated with 
    a nucleation rate. We calculate the density of falling particles (and thus their seperation)
    at a number of analysis radii. Part of the calculation needs us to know an interaction region
    around each particle. This is found self consistently, but an initial guess can be provided.
    
    Input arguments:
    
    nucleation_radii: The radii where we explicity consider nuclation to take place. These are used
        as integration points for a numerical integration over the whole nucleating layer so we need
        enough of thse to converge. However, as the solution seems to be well behaved O(10) seems to
        be enough. A 1D array of radii (measured outwards from the center of the Earth) in m.
    nucleation_rates: The nucleation rate (in events per m^3 per s) at each nucleation_radii. 1D array
    analysis_radii: locations where the particle density and other output parameters are calculated.
        1D array in m. No particular relationship with nucleation radii assumes. This is also the place
        where the self conssitency of the solution is checked.
    initial_interaction_radii: an initial guess at the interaction radius at each analysis_radii. Interpolatied
        for all other radii (using linear interpolator). Defaults to 1m.
    maximum_seperation: how big can we let the seperation get. Defaults to 3m.
    max_rel_error: maximum relative error on any particle radius at any analysis position to be considered converged. 
        Defaults to 1%
    max_rel_error: maximum absolute error on any particle radius at any analysis position to be considered converged. 
        Defaults to 1 mm.
        
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
    """
    
    # Set initial liquid compositioon # TODO: pass xfunc in as argument 
    xl_func = spi.interp1d(analysis_radii, xfunc(analysis_radii), fill_value='extrapolate')
    
    # Calculate an initial guess using the provided liquid compositioon (TODO: pass in xl)
    solutions, particle_densities, calculated_interaction_radii, growth_rate, solid_vf, \
        particle_radius_unnormalised, particle_radius_histogram = integrate_snow_zone(
        analysis_radii, radius_inner_core, radius_top_flayer, nucleation_radii, 
        nucleation_rates, tfunc, xl_func, pfunc, gfunc,
        start_time, max_time, initial_particle_size, k0, dl, mu)
    
    # Work out the updated liquid composition to maintain mass
    xl_points = np.zeros_like(analysis_radii)
    for i, analysis_r in enumerate(analysis_radii):
        _, mol_vol_solid, _ = feot.solid_molar_volume(1.0, pfunc(analysis_r), tfunc(analysis_r))
        mol_vol_liquid, _, _ = feot.liquid_molar_volume(xl_func(analysis_r), pfunc(analysis_r), tfunc(analysis_r))
        moles_solid = (solid_vf[i] * 100.0**3) / mol_vol_solid
        moles_liquid = ((1.0 - solid_vf[i]) * 100.0**3) / mol_vol_liquid
        mol_frac_solid = moles_solid / (moles_solid + moles_liquid)
        xl_points[i] = xfunc(analysis_r) / (1.0 - mol_frac_solid)
    xl_func = spi.interp1d(analysis_radii, xl_points, fill_value='extrapolate')
    
    converged = False
    while not converged:
        # Recalculate solution with updated (TODO: xl...)
        solutions, particle_densities, calculated_interaction_radii, growth_rate, solid_vf, \
             new_particle_radius_unnormalised, particle_radius_histogram = integrate_snow_zone(
             analysis_radii, radius_inner_core, radius_top_flayer, 
             nucleation_radii, nucleation_rates, tfunc, xl_func, pfunc, gfunc,
             start_time, max_time, initial_particle_size, k0, dl, mu,)
        converged = np.allclose(particle_radius_unnormalised, new_particle_radius_unnormalised,
                                atol=max_absolute_error, rtol=max_rel_error)
        particle_radius_unnormalised = new_particle_radius_unnormalised
        
        # Work out the updated liquid composition to maintain mass  
        xl_points = np.zeros_like(analysis_radii)
        for i, analysis_r in enumerate(analysis_radii):
            _, mol_vol_solid, _ = feot.solid_molar_volume(1.0, pfunc(analysis_r), tfunc(analysis_r))
            mol_vol_liquid, _, _ = feot.liquid_molar_volume(xl_func(analysis_r), pfunc(analysis_r), tfunc(analysis_r))
            moles_solid = (solid_vf[i] * 100.0**3) / mol_vol_solid
            moles_liquid = ((1.0 - solid_vf[i]) * 100.0**3) / mol_vol_liquid
            mol_frac_solid = moles_solid / (moles_solid + moles_liquid)
            xl_points[i] = xfunc(analysis_r) / (1.0 - mol_frac_solid)
        xl_func = spi.interp1d(analysis_radii, xl_points, fill_value='extrapolate')
        
    return solutions, particle_densities, calculated_interaction_radii, growth_rate, solid_vf, \
        particle_radius_unnormalised, particle_radius_histogram 