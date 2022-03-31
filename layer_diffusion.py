# This python module provides functions for solution of the
# temperature and composition of the f-layer assuming that
# heat and oxygen is produced by growth of the solid particles
# and this is balanced by diffusion out of the layer. The
# two problems are independent but can be solved using much
# of the same code.

import numpy as np
import scipy.integrate as spi
import scipy.interpolate as spinterp
import matplotlib

def solve_layer_diffusion(radii, sources, diffusion_coefficent, value_guess, *,
                          top_value_bc=None, top_derivative_bc=None,
                          bottom_value_bc=None, bottom_derivative_bc=None,
                          derivative_guess=None, fig_axs=None, fig_col='k'):
    """
    Temperature or composition profile balancing production and diffusion
    
    We treat the F-layer as having no liquid motion, so any heat or oxygen
    produced by the growth of solid is balanced by diffusion out of the
    layer. This amounts to the solution of a pair of first order coupled
    ODEs with specified boundary conditions. This function is flexible
    about those boundary conditions (fixed value or fixed flux on either
    boundary). There is also the need for an initial guess of the solution
    (value and derivative). The function requiers a guess for the value
    but the derivative guess is optional (it is calculated if not supplied).
    The guess need not match the boundary conditions (but it is probably
    sensible if it is close). Optionally, the function can accept a tuple
    containing Matplotlib axes objects for plotting the results.
    
    The function returns the values (e.g. temperatures) evaluated at each
    input radial point (as a numpy array).
    
    Input positional arguments:
    
    radii: a strictly increasing array of points for the F-layer, location of sources, and boundaries (m)
    sources: source rates at each radius (in W/m^3, or CHEMISTRY)
    diffusion_coefficent: the diffusion coefficent, assumed constant (in W.m^-3.K^-1 or CHEM)
    value_guess: initial guess for the solution value at each radius (in K or CHEM)
    
    Input optional arguments (kw only):
    
    top_value_bc: value boundary condition at the top of the F-layer (K or CHEM) [*]
    top_derivative_bc: derivative boundary condition at the top of the F-layer (K/m or CHEM) [*]
    bottom_value_bc: value boundary condition at the bottom of the F-layer (K or CHEM) [**]
    bottom_derivative_bc: derivative boundary condition at the bottom of the F-layer (K/m or CHEM) [**]
    derivative_guess: an initial guess for the derivative. If missing calculated from the value_guess (K/m or CHEM)
    fig_axs: a tuple of three Matplotlib figure axes. If present the solution and initial conditions are
             added to these axes,
    
    Note: exactly one top boundary condition (marked [*]) must be present and this corresponds to
    the radius given by radii[-1]. Exactly one bottom boundary condition (marked [**] must be present
    and this corresponds to the radius given by radii[0]. 
    
    """
    # Build initial guess array
    if derivative_guess is None:
        derivative_guess = np.gradient(value_guess, radii)
    guess = np.vstack((value_guess, derivative_guess))
    
    # Turn the source values into a function (linear interplation seems to 
    # work best. Raise an error if we are outside the input array. We need
    # this because the mesh changes in the solver.
    source_fun = spinterp.interp1d(radii, sources, kind='linear', bounds_error=True)
    
    # Set up the fuction defining the ODEs. Here x[:] is an 1D array 
    # of R, y[0,:] is a 1D array of T, and y[1,:] is a 1D array of 
    # dT/dR. We return the derivatives at each point. Note that the
    # size of x can change from call to call and y[1] is the row y[1,:].
    def fun(x, y):
        dy0_by_dx = y[1]
        dy1_by_dx = (- source_fun(x) - (2.0 * diffusion_coefficent / x) * y[1]
                    ) / diffusion_coefficent
        return np.vstack((dy0_by_dx, dy1_by_dx))
    
    # Set up boundary conditions. First check we have the right number
    # of conditions then create a local function that returns the 
    # error at the two boundaries.
    assert (top_value_bc is None) != (top_derivative_bc is None), "Exactly one top BC needed"
    assert (bottom_value_bc is None) != (bottom_derivative_bc is None), "Exactly one bottom BC needed"
    
    if (top_value_bc is not None) and (bottom_value_bc is not None):
        # e.g. Two values fixed
        def bc(ya, yb):
            return np.array([ya[0] - bottom_value_bc, yb[0] - top_value_bc])
    elif (top_value_bc is not None) and (bottom_derivative_bc is not None):
        def bc(ya, yb):
            return np.array([ya[1] - bottom_derivative_bc, yb[0] - top_value_bc])
    elif (top_derivative_bc is not None) and (bottom_value_bc is not None):
        def bc(ya, yb):
            return np.array([ya[0] - bottom_value_bc, yb[1] - top_derivative_bc])
    elif (top_derivative_bc is not None) and (bottom_derivative_bc is not None):
        def bc(ya, yb):
            return np.array([ya[1] - bottom_derivative_bc, yb[1] - top_derivative_bc])
    else:
        assert True, "Internal error in BC function setup code"
        
    # Run the solver
    result = spi.solve_bvp(fun, bc, radii, guess)
    
    if not result.success:
        print(f"Boundary value solver failed with status {result.status}")
        print(f"Solver message is \n{result.message}")
        assert True, "BVP solution error"
        
    # Plot the results etc.
    if fig_axs is not None:
        assert len(fig_axs) == 3, "Expected three Axes object to draw on"
        assert isinstance(fig_axs[0], matplotlib.axes.Axes), "Expect fig_axs[0] to be an Axes object"
        assert isinstance(fig_axs[1], matplotlib.axes.Axes), "Expect fig_axs[1] to be an Axes object"
        assert isinstance(fig_axs[2], matplotlib.axes.Axes), "Expect fig_axs[2] to be an Axes object"
        
        r_plot = np.linspace(radii[0], radii[-1], num=500)
        t_plot = result.sol(r_plot)[0]
        dt_dr_plot = result.sol(r_plot)[1]
    
        fig_axs[0].plot(r_plot/1000.0, t_plot, f"{fig_col}-")
        fig_axs[1].plot(r_plot/1000.0, dt_dr_plot, f"{fig_col}-")
        fig_axs[2].plot(radii/1000, sources, f"{fig_col}-")
    
        fig_axs[0].plot(radii/1000.0, guess[0], 'r:') 
        fig_axs[1].plot(radii/1000.0, guess[1], 'r:')
    
        if (top_value_bc is not None):
            fig_axs[0].plot(radii[-1]/1000.0, top_value_bc, 'ro')
        else:
            fig_axs[1].plot(radii[-1]/1000.0, top_derivative_bc, 'ro')
        if (bottom_value_bc is not None):
            fig_axs[0].plot(radii[0]/1000.0, bottom_value_bc, 'ro')
        else:
            fig_axs[1].plot(radii[0]/1000.0, bottom_derivative_bc, 'ro')
            
        fig_axs[0].set_xlabel('$R$ (km)')
        fig_axs[0].set_ylabel('$T(R)$ (K)')
        fig_axs[1].set_xlabel('$R$ (km)')
        fig_axs[1].set_ylabel('$\mathrm{d}T / \mathrm{d}R$ (K/m)')
        fig_axs[2].set_xlabel('$R$ (km)')
        fig_axs[2].set_ylabel('$Q(R)$ (W/m$^3$)')
        
    return result.sol(radii)[0]