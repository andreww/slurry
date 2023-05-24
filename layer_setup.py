import numpy as np
import scipy.optimize as spo

import earth_model
import feo_thermodynamics as feot
import scipy.interpolate as spi

# Functions to setup and analyse the layer without the particle calculation
#
# This module provides functions to create the temperature and composition
# functions and function factories needed as input to the slurry layer
# problem. It also provides functions to investigate the properties of the
# layer (e.g. its stability against convection and its heat budget).


def setup_flayer_functions(r_icb, r_cmb, f_layer_thickness, gruneisen_parameter, delta_t_icb,
                           xfe_outer_core, xfe_icb, knott_radii, **kwargs):
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
    t_func_creator = make_new_t_func_creator(r_flayer_top, r_icb, adabat_t_top_flayer, knott_radii)
    t_grad_flayer = (adabat_t_top_flayer - temperature_icb) / f_layer_thickness
    assert t_grad_flayer < 1.0E-12, "Temperature gradient should be negative"
    tfunc_blank_params = np.zeros(knott_radii.size-1)
    tfunc_blank_params[0] = t_grad_flayer
    flayer_temperature_function = t_func_creator(tfunc_blank_params)
    
    # And composition. Input setup is slightly different
    x_func_creator = make_new_x_func_creator(r_flayer_top, r_icb, xfe_outer_core, knott_radii)
    xfunc_blank_params = np.zeros(knott_radii.size-1)
    x_grad = (xfe_outer_core - xfe_icb) / f_layer_thickness
    xfunc_blank_params[0] = x_grad
    flayer_composition_function = x_func_creator(xfunc_blank_params)
    
    @np.vectorize
    def temperature_function(r):
        if r >= r_cmb:
            temp = t_cmb # density is odd at disconts, just fix T outside core.
        elif r > r_flayer_top:
            temp = t_cmb * (prem.density(r/1000.0)/rho_cmb)**gruneisen_parameter
        else: # Will give value inside inner core, but we may need that for IVP solver...
            temp = flayer_temperature_function(r)
        return temp
    

    # Finally, a function to give the composition everywhere
    @np.vectorize
    def composition_function(r):
        if r > r_flayer_top:
            xfe = xfe_outer_core
        else:
            xfe = flayer_composition_function(r)
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
    
    return temperature_function, adiabatic_temperature_function, flayer_temperature_function, \
        t_func_creator, composition_function, \
        pressure_function, gravity_function, x_func_creator


def make_new_t_func_creator(radius_top_flayer, r_icb, t_top_flayer, knott_radii):
    """
    Create temperature function creator from model setup
    
    The idea is that this is called using the general parameters
    of the F-layer that remain fixed in any given model run. It
    returns a function that accepts a set of temperatue parameters
    that, when called, returns a cublic spline representation of the
    temperatue through the F-layer. This may be one too many layers of
    abstraction, but this way the raw temperature function (which is
    returned by the function that is returned by this function) is
    quick to evaluate and can easily be updated inside a optimisation
    loop.
    
    Arguments to this function are:
    
    radius_top_f_layer: in m
    r_icb: in m
    t_top_flayer: this will be fixed for all temperature models, in K
    knott_radii: the set of N points where the cubic spline knots will be
        located. This should include the ICB and 
        the top of the F-layer.
        
    Returns a function which returns a cubic spline represnetation of the
    temperature when called. This takes a single array of N-1 parameters
    which have the following meaning:
    
    parameters[0]: thermal gradient in K/m, should be negative (gets hotter
        downwards) and be about -0.001 
    parameters[1:N-1]: temperature purtubations (in K) at each analysis radus
        other than the inner boundary and the outer boundary.
        
    The returned cubic spline is forced to have zero gradient at the ICB and
    zero second derivative at the top of the F-layer. The temperatuer at the
    ICB is set by the overall thermal gradient. The temperature at the top
    of the F-layer cannot be changed once set. This setup matches the 'natural'
    boundary conditions of the thermal problem, but will need to be changed if
    we allow direct freezing at the ICB.
    
    Good luck!
    """
    
    layer_thickness = radius_top_flayer - r_icb
    
    def t_func_creator(params):
        # params contains dt_dr and a Dt for each point not at the ends 
        assert params.shape[0] == knott_radii.shape[0] - 1, "params radii mismatch"
        dt_dr = params[0]
        t_points = t_top_flayer - (radius_top_flayer - knott_radii) * dt_dr
        t_points[1:-1] = t_points[1:-1] + params[1:]
        return spi.CubicSpline(knott_radii, t_points, bc_type=((1, 0.0), (2, 0.0)))
    
    return t_func_creator


def make_new_x_func_creator(radius_top_flayer, r_icb, x_top_flayer, knott_radii):
    """
    Create composition function creator from model setup
    
    The idea is that this is called using the general parameters
    of the F-layer that remain fixed in any given model run. It
    returns a function that accepts a set of composition parameters
    that, when called, returns a cublic spline representation of the
    composition through the F-layer. This may be one too many layers of
    abstraction, but this way the raw composition function (which is
    returned by the function that is returned by this function) is
    quick to evaluate and can easily be updated inside a optimisation
    loop.
    
    Arguments to this function are:
    
    radius_top_f_layer: in m
    r_icb: in m
    x_top_flayer: this will be fixed for all composition models, in mol frac Fe
    knott_radii: the set of N points where the cubic spline knots will be
        located. This should include the ICB and 
        the top of the F-layer.
        
    Returns a function which returns a cubic spline represnetation of the
    composition when called. This takes a single array of N-1 parameters
    which have the following meaning:
    
    parameters[0]: composition gradient in mol-frac/m, should be negative (gets 
        more oxygen rich do downwards) and probably small 
    parameters[1:N-1]: composition purtubations (in mol frac) at each analysis radus
        other than the inner boundary and the outer boundary.
        
    The returned cubic spline is forced to have zero gradient at the ICB and
    zero second derivative at the top of the F-layer. The composition at the
    ICB is set by the overall composition gradient. The composition at the top
    of the F-layer cannot be changed once set. This setup matches the 'natural'
    boundary conditions of the compositional problem, but will need to be changed if
    we allow direct freezing at the ICB.
    """
    
    layer_thickness = radius_top_flayer - r_icb
    
    def x_func_creator(params):
        # params contains dt_dr and a Dt for each point not at the ends 
        assert params.shape[0] == knott_radii.shape[0] - 1, "params radii mismatch"
        dx_dr = params[0]
        x_points = x_top_flayer - (radius_top_flayer - knott_radii) * dx_dr
        x_points[1:-1] = x_points[1:-1] + params[1:]
        return spi.CubicSpline(knott_radii, x_points, bc_type=((1, 0.0), (2, 0.0)))
    
    return x_func_creator



def estimate_brunt_vaisala_frequency(r_top, r_bot, tfunc, atfunc, xfunc, gfunc, pfunc):
    """
    Derive an estimate for the Brunt Vaisala frequency for a layer by taking differences
    
    We assume we can write:
    
       N_BV^2 = -(g_m/rho_m) (rho'_t - rho'_b)/(R_t - R_b)
       
    where t, m and b are at the top, middle and bottom of the layer and rho' is the difference
    in density between an adiabatic state and the real density. For the adiabatic state we follow
    the temperature down an adiabat and assume the composition is well mixed (i.e. assume it is 
    the same as the bulk outer core).
    
    Arguments
    r_top: radius (in m) of top of the layer 
    r_bot: radius (in m) of bottom of the layer
    tfunc: function that returns the temperature (K)
    atfunc: function that returns adiabatic temperature profile through the layer (K)
    xfunc: function that returns the liquid composition of the layer (mol. frac. Fe)
    gfunc: function that returns the gravity through the layer (m/s)
    pfunc: function that returns the pressure through the layer (GPa)
    
    Returns: (Nbv, N2)
    Nbv: Brunt Vaisala frequency (Hz)
    N2: Squared Brunt Vaisala frequency 
    If N22 is negative we return a complex frequency.
    """
    r_m = r_bot + (r_top - r_bot)/2.0
    # Don't need the absolute density at the top of the layer
    rho_b, _, _, _, _, _ = feot.densities(xfunc(r_bot), pfunc(r_bot), tfunc(r_bot))
    rho_m, _, _, _, _, _ = feot.densities(xfunc(r_m), pfunc(r_m), tfunc(r_m))
    # Our reference at the top is the same as the real density (because we construct
    # the layer that way), the reference at the bottom follows the adiabatic temperature
    # but uses the composition from the top (well mixed)
    ref_rho_b, _, _, _, _, _ = feot.densities(xfunc(r_top), pfunc(r_bot), atfunc(r_bot))
        
    N2 = -1.0 * (gfunc(r_m) / rho_m) * (0.0 - (rho_b - ref_rho_b)) / (r_top - r_bot)
    
    if N2 >= 0.0:
        Nbv = np.sqrt(N2)
    else:
        Nbv = np.sqrt((N2 + 0j))
    return Nbv, N2


def fit_quad_func_boundaries(r_icb, r_ftop, v_icb, v_ftop):
    """
    Find a quadratic function to match two values and zero derivative
    
    Find the coefficents of a quadratic equation for particualar values 
    (e.g. temperature) at the top and bottom of a layer and for zero
    derivative on the inner boundary, then return the function.
    
    Finds a, b and c for:
    
        Y(r) = ar^2 + br + c; Y(r_icb) = v_icb; Y_(r_ftop) = v_ftop
        dY/dr |_r_icb = 2ar + b = 0
    
    a, b and c come from solving:
    
        | r_icb^2    r_icb    1 |  | a |     | v_icb  |
        | r_ftop^2   r_ftop   1 |  | b | =   | v_ftop |
        | 2 r_icb      1      0 |  | c |     | 0      |
        
    returns a function (of r)
    """
    denom = (r_icb - r_ftop)**2
    a = (v_ftop - v_icb)/denom
    b = (-2.0*r_icb * (v_ftop - v_icb))/denom
    c = r_icb**2 * v_ftop - 2 * r_icb * r_ftop * v_icb + r_ftop**2 * v_icb
    c = c / denom
    def quad_func(r):
        v = a * r**2 + b * r + c
        return v
    return quad_func