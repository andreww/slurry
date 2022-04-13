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
# profile. For most cases the flayer_case function does
# everything, taking input that people may want to change
# and returning most useful output. However, for some cases
# a combination of setup_flayer_functions, evaluate_flayer
# and analyse_flayer may be more useful (e.g. for making 
# more involved plots, or showing how things work. flayer_case 
# is really just a thin wrapper around these three functions
# but it is set up so we can process cases in parallel with
# multiprocessing (see process_cases.py).

def flayer_case(f_layer_thickness, delta_t_icb, xfe_outer_core, xfe_icb,
                growth_prefactor, chemical_diffusivity, thermal_conductivity,
                kinematic_viscosity, i0, surf_energy, 
                number_of_analysis_points,
                wetting_angle=180.0, hetrogeneous_radius=None,
                r_icb=1221.5E3, r_cmb=3480.0E3, gruneisen_parameter=1.5,
                start_time=0.0, max_time=1.0E12, max_rel_error=1.0E-5,
                max_absolute_error=1.0E-8, verbose=False):
    """
    Setup, run and analyse a non-equilibrium F-layer model
    
    We define the F-layer as a region above the ICB with a temperature
    that varies linearly with radius and is pinned to the adiabat and
    Fe-FeO liquidus at the top of the F-layer and is colder (or hotter) 
    than the adiabat at the ICB. The F-layer thickness is chosen along
    with the oxygen content of the outer core and a Grüneisen parameter 
    and the temperature of the CMB varied until the temperature at top
    of the F-layer corresponds to the liquidus. Pressure (and gravity)
    is taken from PREM. Inside the F-layer the total oxygen content is 
    permitted to vary linearly between the bulk core value and a value
    at the ICB (these are commonly chosen to be equal). As defined the
    F-layer is in the Fe-FeO two phase region so we allow iron to nucleate,
    sink and grow within it and evaluate the solid content.
    
    Input arguments:
    * f_layer_thickness: thickness of F-layer (m)
    * delta_t_icb: difference between adiabat and F-layer
      temperature at ICB, more positive is colder (K)
    * xfe_outer_core: composition at top of F-layer (mol frac Fe)
    * xfe_icb: composition at bottom of F-layer (mol frac Fe)
    * initial_particle_size: particle size at nucleation (m)
    * growth_prefactor: prefactor for particle growth rate (m/s)
    * chemical_diffusivity: diffusivity of oxygen in liquid iron (UNITS?)
    * kinematic_viscosity: kinematic viscosity of liquid iron(UNITS)
    * i0: prefactor for CNT (s^-1 m^-3)
    * surf_energy: surface energy for CNT (J m^-2)
    * number_of_analysis_ponts: how many points to perform liquid composition
      and other analysis (-)
    
    Optional input arguments:
    * wetting_angle: set to < 180.0 degrees to turn on hetrogeneous nucleation
    * hetrogeneous_radius: a fixed initial particle radius (rather than the critical radius)
    * r_icb: inner core boundary radius, default is 1221.5e3 (m)
    * r_cmb: core mantle boundary radius, default is 3480.0e3 (m)
    * gruneisen_parameter: Grüneisen parameter for outer core adiabat 
      calculation, default is 1.5 (-)
    * start_time: initial time condition for IVP solver, default 0.0 (s)
    * max_time: maximum time for the IVP solver, default 1.0E12 (s)
    * max_rel_error: maximum relative error on liquid composition in
      self consistent solution, default 0.01 (-)
    * max_absolute_error: maximum absolute error on liquid composition in
      self consistent solution, default 0.001 (mol frac Fe)
    * inner_core_offset: start discretisation of F-layer this far above
      ICB, default is 1000.0 (m)
      
      
    Output arguments:
    * t_flayer_top: temperature at top of the F-layer
    * t_flayer_bottom: temperature at ICB
    * t_cmb: temperature of core mantle boundary
    """
    # Derived values of use
    r_flayer_top = r_icb + f_layer_thickness
    
    # Make nterpolation functions for each input property (matching liquidus and
    # adiabat
    tfunc, tafunc, xfunc, pfunc, gfunc = setup_flayer_functions(r_icb, r_cmb,
         f_layer_thickness, gruneisen_parameter, delta_t_icb, xfe_outer_core, xfe_icb)
    
    # Discretisation points
    nucleation_radii = np.linspace(r_icb, r_flayer_top, number_of_analysis_points)
    analysis_radii = np.linspace(r_icb, r_flayer_top, number_of_analysis_points)
    
    # doit!
    solutions, particle_densities, growth_rate, solid_vf, \
        particle_radius_unnormalised, partial_particle_densities, opt_xlfunc, \
        crit_nuc_radii, nucleation_rates, out_x_points, out_t_points  = solve_flayer(tfunc, xfunc, 
        pfunc, gfunc, start_time, max_time, growth_prefactor, 
        chemical_diffusivity, thermal_conductivity, kinematic_viscosity, i0, surf_energy,
        wetting_angle, hetrogeneous_radius,
        nucleation_radii, analysis_radii, r_icb, 
        r_flayer_top, max_rel_error=max_rel_error, max_absolute_error=max_absolute_error,
                                                                                 verbose=verbose)


    # Post-solution analysis
    xfunc = spi.interp1d(analysis_radii, out_x_points, fill_value='extrapolate')
    tfunc = spi.interp1d(analysis_radii, out_t_points, fill_value='extrapolate')
    calculated_seperation, growth_rate, vf_ratio = analyse_flayer(solutions, 
                   nucleation_radii, analysis_radii, nucleation_rates, r_icb,
                   particle_densities, growth_rate, solid_vf,
                   particle_radius_unnormalised, partial_particle_densities, tfunc, xfunc, 
                   pfunc, verbose=verbose)
    
    
    return solutions, analysis_radii, particle_densities, calculated_seperation, solid_vf, \
        particle_radius_unnormalised, partial_particle_densities, growth_rate, crit_nuc_radii, nucleation_rates, \
        vf_ratio, out_x_points, out_t_points


def setup_flayer_functions(r_icb, r_cmb, f_layer_thickness, gruneisen_parameter, delta_t_icb, xfe_outer_core, xfe_icb, **kwargs):
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
    r_flayer_top = r_icb + f_layer_thickness
    # Base P and g on PREM...
    prem = earth_model.Prem()
    
    # First find the liquidus temperature at the top of the F-layer... we know P (from PREM)
    # and X (from our input). NB: my PREM module works in km and does not like vector input.
    # This is quite slow and could be optimised by avoiding the double brentq calls!
    tl_top_flayer = feot.find_liquidus(xfe_outer_core, prem.pressure(r_flayer_top/1000.0))
    
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
        adabat_t_top_flayer = tcmb * (rho_top_flayer/rho_cmb)**gruneisen_parameter
        t_error = adabat_t_top_flayer - tl_top_flayer
        return t_error
    t_cmb = spo.brentq(_t_error_top_flayer, 1000, 8000)
    
    # We can now build our function to give the adiabatic temperature
    rho_icb = prem.density(r_icb/1000.0)
    adabat_t_top_flayer = t_cmb * (rho_top_flayer/rho_cmb)**gruneisen_parameter
    adabat_icb = t_cmb * (rho_icb/rho_cmb)**gruneisen_parameter
    @np.vectorize
    def adiabatic_temperature_function(r):
        temp = t_cmb * (prem.density(r/1000.0)/rho_cmb)**gruneisen_parameter
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
            temp = t_cmb * (prem.density(r/1000.0)/rho_cmb)**gruneisen_parameter
        else: # Will give value inside inner core, but we may need that for IVP solver...
            temp = temperature_icb + (r - r_icb)*(
                (adabat_t_top_flayer-temperature_icb)/(r_flayer_top-r_icb))
        return temp
    

    # Finally, a function to give the composition everywhere
    @np.vectorize
    def composition_function(r):
        if r > r_flayer_top:
            xfe = xfe_outer_core
        else:
            xfe = xfe_icb + (r - r_icb)*(
                (xfe_outer_core-xfe_icb)/(r_flayer_top-r_icb))
        return xfe
    
    @np.vectorize           
    def pressure_function(r):
        if r < 0.0:
            r = 0.0
        return prem.pressure(r/1000.0)
    
    @np.vectorize
    def gravity_function(r):
        if r < 0.0:
            r = 0.0
        return prem.gravity(r/1000.0)
    
    return temperature_function, adiabatic_temperature_function, composition_function, \
        pressure_function, gravity_function


def solve_flayer(tfunc, xfunc, pfunc, gfunc, start_time, max_time,
                k0, dl, k, mu, i0, surf_energy, wetting_angle, hetrogeneous_radius,
                nucleation_radii, analysis_radii, radius_inner_core, 
                radius_top_flayer, max_rel_error=1.0E-7, max_absolute_error=1.0E-10,
                verbose=False, silent=False):
    """
    Create a self consistent solution for the F-layer assuming non-equilibrium growth and falling
    of iron crystals
    
    This involves integration of the results from falling-growing crystal calculations with the
    integration being with respect to the nucleation radius. This function acts as a wrapper and
    driver for evaluate_flayer, which solves the nucleation, particle falling, and flux balance
    problems for a given temperarture and liquid composition profile. evaluate_flayer is called
    repeatedly with updated profiles until convergence.
    
    Input arguments:
    
    tfunc: function returing initial temperature in K as a function of radius in m (callable), or scalar (K)
    xfunc: function returing initial liquid composition in mol. frac Fe as a function of radius in m 
           (callable), or scalar (in mol. frac Fe)
    pfunc: function returing pressure in GPa as a function of radius in m (callable), or scalar (GPa)
    gfunc: function returning acceleration due to gravity in m/s as a function of radius in m (callable)
           or scalar (m/s)
    start_time: initial time condition for IVP solver (s)
    max_time: maximum time for the IVP solver (s)
    k0: prefactor for particle growth rate (m/s)
    dl: diffusivity of oxygen in liquid iron (UNITS?)
    k: thermal conductivity of the layer (W/m/K)
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
    max_rel_error: maximum relative error on any particle radius at any analysis position to be considered converged. 
        Defaults to 1%
    max_rel_error: maximum absolute error on any particle radius at any analysis position to be considered converged. 
        Defaults to 1 mm.
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
    t_points: Optimised (self consistent) temperature profile through layer evaluated at points (K)
    xl_points: Optimised (self consistent) liquid composition through layer evaluated at points (mol frac Fe)
    """
    
    if not silent:
        print("Running optimisation to find self conssitent temperature")

    # Optimiser takes array of temperatures, not function. So evaluate for initial guess
    input_t_points = tfunc(analysis_radii)
    
    res = spo.minimize(evaluate_flayer_wrapper_func, input_t_points, args=(xfunc, pfunc, 
            gfunc, start_time, max_time, k0, dl, k, mu, i0, 
            surf_energy, wetting_angle, hetrogeneous_radius, nucleation_radii, 
            analysis_radii, radius_inner_core, radius_top_flayer),
            options={'disp': True})
    
    if not silent:
        print("Optimisation done. Results are:")
        print(res)
    if not res.success:
        print("***************************************************************************")
        print("***************************************************************************")
        print("NB: Optimiser failed! Will run final point (and store) for possible restart")
        print("***************************************************************************")
        print("***************************************************************************")
    
    # Function of self consistent temperatures
    tfunc = spi.interp1d(analysis_radii, res.x, fill_value='extrapolate')
    
    # Do calculation for self conssitent location
    if not silent:
        print("Recaculating F-layer solution at optimum temperature")
    solutions, particle_densities, growth_rate, solid_vf, \
    particle_radius_unnormalised, partial_particle_densities, \
    crit_nuc_radii, nucleation_rates, out_t_points, out_x_points = evaluate_flayer(
            tfunc, xfunc, pfunc, gfunc, start_time, max_time, k0, dl, k, mu, i0, 
            surf_energy, wetting_angle, hetrogeneous_radius, nucleation_radii, 
            analysis_radii, radius_inner_core, radius_top_flayer, verbose, silent)
    
    # We should report the temperatures we used for the final calculation
    # (we should also report the output temperatures too, but they will match 
    # iff the convergence has converged!
    out_t_points = tfunc(analysis_radii)
        
    return solutions, particle_densities, growth_rate, solid_vf, \
        particle_radius_unnormalised, partial_particle_densities, \
        xfunc, crit_nuc_radii, nucleation_rates, out_x_points, out_t_points 


def evaluate_flayer_wrapper_func(tpoints, xfunc, pfunc, gfunc, start_time, max_time,
                    k0, dl, k, mu, i0, surf_energy, wetting_angle, hetrogeneous_radius,
                    nucleation_radii, analysis_radii, radius_inner_core, 
                    radius_top_flayer):
    """
    Wrapper function around evaulate_flayer which we can pass to scipy optimise
    
    returns the sum of the squared difference between tpoints and calculated temperatures
    """
    tfunc = spi.interp1d(analysis_radii, tpoints, fill_value='extrapolate')
    
    solutions, particle_densities, growth_rate, solid_vf, \
        particle_radius_unnormalised, partial_particle_densities, \
        crit_nuc_radii, nucleation_rates, t_points_out, xl_points = \
            evaluate_flayer(tfunc, xfunc, pfunc, gfunc, start_time, max_time,
                    k0, dl, k, mu, i0, surf_energy, wetting_angle, hetrogeneous_radius,
                    nucleation_radii, analysis_radii, radius_inner_core, 
                    radius_top_flayer, verbose=False, silent=False)
    
    sse = np.sqrt(np.sum((tpoints - t_points_out)**2)/len(analysis_radii))
    print(f"Mean abs error = {sse:4g} (K)")
    return sse
    

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
        particle_radius_unnormalised, partial_particle_densities, solid_volume_production_rate = integrate_snow_zone(
        analysis_radii, radius_inner_core, radius_top_flayer, nucleation_radii, 
        nucleation_rates, tfunc, xfunc, pfunc, gfunc,
        start_time, max_time, crit_nuc_radii, k0, dl, mu, verbose=verbose)
    
    
    latent_heat = 0.75 * 1000000.0 # J/kg - from Davies 2015, could calculate this from the thermodynamics I think (FIXME).
    _, _, _, fe_density, _, _ = feot.densities(1.0, pfunc(analysis_radii), tfunc(analysis_radii))
    mass_production_rate = solid_volume_production_rate * fe_density
    heat_production_rate = mass_production_rate * latent_heat
    # Here we should calculate the updated composion. For now just return input so we can test without self consistent loop
    # FIXME!
    top_bc = tfunc(analysis_radii[-1])
    bottom_bc = 0.0
    if verbose or (not silent):
        print("Finding T to match heat production rate")
        print(f"Boundary conditions, top: {top_bc} K, bottom {bottom_bc} K/m")
        print("Radius (km), P (GPa), Guess T (K), dm/dt (kg/s), Q (W/m^3), Calculated T (K)")
    t_points_out = layer_diffusion.solve_layer_diffusion(analysis_radii, heat_production_rate, 
                                                         100.0, tfunc(analysis_radii),
                                                         top_value_bc=top_bc,
                                                         bottom_derivative_bc=bottom_bc)
    if verbose or (not silent and len(analysis_radii) <= 10):
        for i, r in enumerate(analysis_radii):
            print(f"{r/1000.0:4g} {pfunc(r):3g} {tfunc(r):4g} {mass_production_rate[i]:.3g} {heat_production_rate[i]:.3g} {t_points_out[i]:4g}")
    elif not silent:
        for i, r in enumerate(analysis_radii):
            if i%(len(analysis_radii)//10) == 0: 
                print(f"{r/1000.0:4g} {pfunc(r):3g} {tfunc(r):4g} {mass_production_rate[i]:.3g} {heat_production_rate[i]:.3g} {t_points_out[i]:4g}")     
              
              
    xl_points = xfunc(analysis_radii)
    if not silent:
        print("NOT doing the diffusion calculation for composition")
        
    return solutions, particle_densities, growth_rate, solid_vf, \
        particle_radius_unnormalised, partial_particle_densities, \
        crit_nuc_radii, nucleation_rates, t_points_out, xl_points


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
        return 0.0, 0.0
    
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
        elif (analysis_time - delta_t) > 0.0:
            distance_above = ((ivp_solution.sol(analysis_time - delta_t)[1] - analysis_radius) / delta_t) * tau
            radius_before = ivp_solution.sol(analysis_time - delta_t)[0]
        else:
            if verbose:
                print("cannot process if next particle has yet to form")
            return 0.0, 0.0
        if (analysis_time + tau) < ivp_solution.t[-1]:
            distance_below = analysis_radius - ivp_solution.sol(analysis_time + tau)[1]
            radius_after = ivp_solution.sol(analysis_time + delta_t)[0]
        elif (analysis_time + delta_t) < ivp_solution.t[-1]:
            distance_below = ((analysis_radius - ivp_solution.sol(analysis_time + delta_t)[1]) / delta_t) * tau
            radius_after = ivp_solution.sol(analysis_time + delta_t)[0]
        else:
            if verbose:
                print("cannot process if previous particle has gone")
            return 0.0, 0.0
        s_v = (0.5 * (distance_below + distance_above))
        particle_volume_growth_rate = ((4/3) * np.pi * (radius_after**3 - radius_before**3)) / (2.0 * delta_t) 
        partial_density = 1/(analysis_area * s_v) # /m^3 - see notebook!
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
        if verbose:
            print("No event data (e.g. dissolved) so partical density is zero")
        
    return partial_density, particle_volume_growth_rate


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
    for i, analysis_r in enumerate(analysis_depths):
        analysis_index = i + 2
        # Particle density at this depth is 'just' the partial density
        # (density from nucleation depth) integrated over nuclation depths
        # It's okay to integrate nuc depths below int depth as this will
        # return zero density. This is a 1D integral (see notebook)
        partial_densities = np.zeros_like(integration_depths)
        partial_radius = np.zeros_like(integration_depths)
        particle_volume_growth_rate = np.zeros_like(integration_depths)
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
            partial_densities[j], particle_volume_growth_rate[j] = partial_particle_density(solutions[j],
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
        
    return particle_densities, solid_vf, particle_radius_unnormalised, \
                   partial_particle_densities, solid_volume_production_rate


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
    
    particle_densities, solid_vf, \
        particle_radius_unnormalised, partial_particle_densities, solid_volume_production_rate = evaluate_partcle_densities(solutions, 
                                        analysis_depths, integration_depths, nucleation_rates, radius_inner_core, 
                                                                                             radius_top_flayer, verbose=verbose)
    
    growth_rate = evaluate_core_growth_rate(solutions, integration_depths, nucleation_rates, radius_inner_core, verbose=verbose)
    
    return solutions, particle_densities, growth_rate, solid_vf, particle_radius_unnormalised, \
                                          partial_particle_densities, solid_volume_production_rate
    
    
