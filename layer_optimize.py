import numpy as np
import scipy.optimize as spo
import scipy.interpolate as spi

import layer_setup
import flayer

# Code to setup, run and optimise an f-layer solution
# This seeks to match the heat production in the layer
# with it's flux out of the layer, and the oxygen production
# in the layer with the flux out of the layer (i.e. assume
# a steady state for the liquid temperature and composition.
#
# NB: the advective 'j-flux' term for two phase flow in a
# slurry is not currently implemented. This is left for 
# future work. See (copious) notes.
#
# The actual slurry solution (for a single case) uses
# flayer.py, layer setup is in layer_setup.py.
#
# For most cases the flayer_case function does
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
                number_of_analysis_points, number_of_knots,
                wetting_angle=180.0, hetrogeneous_radius=None,
                r_icb=1221.5E3, r_cmb=3480.0E3, gruneisen_parameter=1.5,
                start_time=0.0, max_time=1.0E12, max_rel_error=1.0E-5,
                max_absolute_error=1.0E-8, verbose=False, opt_mode='temp'):
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
        
    # Discretisation points
    nucleation_radii = np.linspace(r_icb, r_flayer_top, number_of_analysis_points)
    analysis_radii = np.linspace(r_icb, r_flayer_top, number_of_analysis_points)
    knott_radii = np.linspace(r_icb, r_flayer_top, number_of_knots)

    
    # Make interpolation functions for each input property (matching liquidus and
    # adiabat
    tfunc, tafunc, ftfunc, tfunc_creator, xfunc, pfunc, gfunc, \
         xfunc_creator = layer_setup.setup_flayer_functions(r_icb, r_cmb, f_layer_thickness,
                    gruneisen_parameter, delta_t_icb, xfe_outer_core, xfe_icb, knott_radii)

    
    t_flayer_top = tfunc(r_flayer_top)
    print(f"Fixed temperature at top of F-layer {t_flayer_top} K")
    
    # doit!
    solutions, particle_densities, growth_rate, solid_vf, \
        particle_radius_unnormalised, partial_particle_densities, \
        crit_nuc_radii, nucleation_rates, out_x_points, out_t_points, \
        opt_tfunc, opt_xfunc, opt_params, n_params = solve_flayer(ftfunc, tfunc_creator, xfunc, xfunc_creator, 
        pfunc, gfunc, start_time, max_time, growth_prefactor, 
        chemical_diffusivity, thermal_conductivity, kinematic_viscosity, i0, surf_energy,
        wetting_angle, hetrogeneous_radius,
        nucleation_radii, analysis_radii, knott_radii, r_icb, 
        r_flayer_top, t_flayer_top, max_rel_error=max_rel_error, max_absolute_error=max_absolute_error,
                                                                                 verbose=verbose, opt_mode=opt_mode)


    # Post-solution analysis
    calculated_seperation, growth_rate, vf_ratio = flayer.analyse_flayer(solutions, 
                   nucleation_radii, analysis_radii, nucleation_rates, r_icb,
                   particle_densities, growth_rate, solid_vf,
                   particle_radius_unnormalised, partial_particle_densities, 
                   opt_tfunc, opt_xfunc, 
                   pfunc, verbose=verbose, diffusion_problem=True)
    
    
    return solutions, analysis_radii, particle_densities, calculated_seperation, solid_vf, \
        particle_radius_unnormalised, partial_particle_densities, growth_rate, crit_nuc_radii, nucleation_rates, \
        vf_ratio, out_x_points, out_t_points, opt_params



def solve_flayer(ftfunc, tfunc_creator, xfunc, xfunc_creator, pfunc, gfunc, start_time, max_time,
                k0, dl, k, mu, i0, surf_energy, wetting_angle, hetrogeneous_radius,
                nucleation_radii, analysis_radii, knott_radii, radius_inner_core, 
                radius_top_flayer, t_top_flayer, max_rel_error=1.0E-7, max_absolute_error=1.0E-10,
                verbose=False, silent=False, opt_mode='temp'):
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
    t_top_flayer: temperature at top of F-layer (K), this is fixed
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

    # Optimiser takes array of temperature parameters, not function. 
    # So we need to build these for the initial guess
    t_icb_init = ftfunc(radius_inner_core)
    tgrad = (t_top_flayer - t_icb_init) / (radius_top_flayer - radius_inner_core)
    t_params_guess = np.zeros(knott_radii.size-1)
    t_params_guess[0] = tgrad
    
    # Bounds for temperature
    lbt = np.ones_like(t_params_guess)
    # Temperature must get hotter downwards, negative grad
    lbt[0] = 10.0*t_params_guess[0]
    lbt[1:] = -25.0
    ubt = np.ones_like(t_params_guess)
    ubt[0] = 0.0
    ubt[1:] = 25.0
    
    # Initial composition parameters
    x_icb_init = xfunc(radius_inner_core)
    x_top_flayer = xfunc(radius_top_flayer)
    xgrad = (x_top_flayer - x_icb_init) / (radius_top_flayer - radius_inner_core)
    x_params_guess = np.zeros(knott_radii.size-1)
    x_params_guess[0] = xgrad
    
    # Bounds for composition
    lbx = np.ones_like(x_params_guess)
    # Maximum gradient should be 0 - composition must get more O rich down 
    lbx[0] = 0.0
    lbx[1:] = -1.0E-3
    ubx = np.ones_like(t_params_guess)
    # Minimim gradient should give 10 mol% O over layer - 
    # composition must get more O rich down 
    ubx[0] = 0.1 / (radius_top_flayer - radius_inner_core)
    ubx[1:] = 1.0E-3
    
    if opt_mode == 'temp':
        params_guess = t_params_guess
        param_bounds = spo.Bounds(lbt, ubt)
        xfunc_creator = xfunc # we don't need to pass in the generator
    elif opt_mode == 'comp':
        params_guess = x_params_guess
        param_bounds = spo.Bounds(lbx, ubx)
        tfunc_creator = ftfunc
    elif opt_mode == 'both':
        # For the 'both' case we order the parameters so they go
        # grad of temp, grad of comp, then all the dts then all the dxs
        # this is because that's the order we want a powell solver to 
        # do line searches in...
        params_guess = np.concatenate((t_params_guess[0:1], x_params_guess[0:1],
                                       t_params_guess[1:], x_params_guess[1:]))
        param_bounds = spo.Bounds(np.concatenate((lbt[0:1],lbx[0:1],lbt[1:],lbx[1:])), 
                                  np.concatenate((ubt[0:1],ubx[0:1],ubt[1:],ubx[1:])))
        
    else:
        raise ValueError('Unknown mode')
    
    print(f"Params_guess = {params_guess}")
    print(f"Params_bounds = {param_bounds}")
    res = spo.minimize(evaluate_flayer_wrapper_func, params_guess, bounds=param_bounds, 
                       args=(tfunc_creator, xfunc_creator, pfunc, 
            gfunc, start_time, max_time, k0, dl, k, mu, i0, 
            surf_energy, wetting_angle, hetrogeneous_radius, nucleation_radii, 
            analysis_radii, radius_inner_core, opt_mode, knott_radii.size-1, radius_top_flayer, t_top_flayer),
            options={'disp': True, 'xtol': 0.0001, 'ftol': 0.1}, method='Powell')
    
    if not silent:
        print("Powell optimisation done. Results are:")
        print(res)
    if not res.success:
        print("***************************************************************************")
        print("***************************************************************************")
        print("NB: Powell ptimiser failed! Storing results anyway")
        print("***************************************************************************")
        print("***************************************************************************")
    
    
    # Unpack optimised parameters
    n_params = knott_radii.size-1
    print(f"Optimised parameters are {res.x}")
    if opt_mode == 'temp':
        tfunc = tfunc_creator(res.x)
        tpoints = tfunc(analysis_radii)
        xfunc = xfunc_creator # No building needed
        xl_points_in = xfunc(analysis_radii)
    elif opt_mode == 'comp':
        xfunc = xfunc_creator(res.x)
        tfunc = tfunc_creator
        tpoints = tfunc(analysis_radii)
        xl_points_in = xfunc(analysis_radii)
        # pass temperature function in as the creator.
    elif opt_mode == 'both':
        t_params = np.concatenate((res.x[0:1], res.x[2:n_params+1]))
        x_params = np.concatenate((res.x[1:2], res.x[n_params+1:]))
        tfunc = tfunc_creator(t_params)
        tpoints = tfunc(analysis_radii)
        xfunc = xfunc_creator(x_params)
        xl_points_in = xfunc(analysis_radii)
        
        # break up arguments and make both functions
    else:
        raise ValueError('Unknown mode')
    
    # Do calculation for self conssitent location
    if not silent:
        print("Recaculating F-layer solution at optimum temperature")
    solutions, particle_densities, growth_rate, solid_vf, \
    particle_radius_unnormalised, partial_particle_densities, \
    crit_nuc_radii, nucleation_rates, out_t_points, out_x_points = flayer.evaluate_flayer(
            tfunc, xfunc, pfunc, gfunc, start_time, max_time, k0, dl, k, mu, i0, 
            surf_energy, wetting_angle, hetrogeneous_radius, nucleation_radii, 
            analysis_radii, radius_inner_core, radius_top_flayer, verbose, silent,
            diffusion_problem=True)
    
    # We should report the temperatures we used for the final calculation
    # (we should also report the output temperatures too, but they will match 
    # iff the convergence has converged!
        
    return solutions, particle_densities, growth_rate, solid_vf, \
        particle_radius_unnormalised, partial_particle_densities, \
        crit_nuc_radii, nucleation_rates, out_x_points, out_t_points, \
        tfunc, xfunc, res.x, n_params


def evaluate_flayer_wrapper_func(params, tfunc_creator, xfunc_creator, pfunc, gfunc, start_time, max_time,
                    k0, dl, k, mu, i0, surf_energy, wetting_angle, hetrogeneous_radius,
                    nucleation_radii, analysis_radii, radius_inner_core, mode, n_params,
                    radius_top_flayer, t_top_flayer):
    """
    Wrapper function around evaulate_flayer which we can pass to scipy optimise
    
    returns the sum of the squared difference between tpoints and calculated temperatures
    """
    print(f"Wrapper called with {params}")
    if mode == 'temp':
        tfunc = tfunc_creator(params)
        tpoints = tfunc(analysis_radii)
        xfunc = xfunc_creator # No building needed
        xl_points_in = xfunc(analysis_radii)
    elif mode == 'comp':
        xfunc = xfunc_creator(params)
        tfunc = tfunc_creator
        tpoints = tfunc(analysis_radii)
        xl_points_in = xfunc(analysis_radii)
        # pass temperature function in as the creator.
    elif mode == 'both':
        t_params = np.concatenate((params[0:1], params[2:n_params+1]))
        x_params = np.concatenate((params[1:2], params[n_params+1:]))
        tfunc = tfunc_creator(t_params)
        tpoints = tfunc(analysis_radii)
        xfunc = xfunc_creator(x_params)
        xl_points_in = xfunc(analysis_radii)
        
        # break up arguments and make both functions
    else:
        raise ValueError('Unknown mode')
        
    
    solutions, particle_densities, growth_rate, solid_vf, \
        particle_radius_unnormalised, partial_particle_densities, \
        crit_nuc_radii, nucleation_rates, t_points_out, xl_points_out = \
            flayer.evaluate_flayer(tfunc, xfunc, pfunc, gfunc, start_time, max_time,
                    k0, dl, k, mu, i0, surf_energy, wetting_angle, hetrogeneous_radius,
                    nucleation_radii, analysis_radii, radius_inner_core, 
                    radius_top_flayer, verbose=False, silent=False,
                    diffusion_problem=True)
    
    sset = np.sqrt(np.sum((tpoints - t_points_out)**2)/len(analysis_radii)) \
                / tpoints[-1] # Normalise by temperatrue at top of F-layer
    ssex = np.sqrt(np.sum((xl_points_in - xl_points_out)**2)/len(analysis_radii)) \
                / xl_points_in[-1]
    # Should be in callback: 
    print(f"Normalised SSE temperature difference = {sset:4g}")
    print(f"Normalised SSE composition difference = {ssex:4g}")
    if mode == 'temp':
        sse = sset
    elif mode == 'comp':
        sse = ssex
    else:
        # Errors already caught. Must be 'both'
        sse = sset + ssex
    print(f"Error for solver = {sse:4g}")
    return sse