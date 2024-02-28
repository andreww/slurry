# Collection of functions that are useful for plotting
# the output of the F-layer model. This is intended to 
# be imported as a module and used in a notebook.

import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import adjustText


# Function and supporting tools to make grid of points in dt dx space
# note _get... functions below are there to help with the lines

def plot_summary_figure(results_df, target_latent_heat=None, 
                        target_density_excess=None, 
                        fig=None, ax=None, marker_x=None,
                        marker_t=None, marker=None, marker_color=None):
    """
    Create a plot showing the heat production as a function of layer setup
    
    The layer setup here amounts to two parameters, dx, the composition 
    difference from well mixed at the ICB and dt, the temperature
    difference from an adiabat at the ICB. We filter (and show) models
    that are not viabale on the basis of being too hot (above the
    liquidus at the ICB), too cold (below the temperature at the top
    of the F-layer at the ICB), or not statified (imaginary N_BV).
    
    Adding target_latent_heat (in W) or target_density_excess (in kg/m^3)
    plots lines on top showing 'viable' solutions - where they cross is
    the 'Earth like' solution.
    """
    if (fig is None) != (ax is None):
        raise ValueError("Must specify both fig and ax or have them both created.")
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(figsize=(7,6))
            
    if (marker_t is None) != (marker is None) != (marker_x is None) != (marker_color is None):
        raise ValueError("Must none or four marker parameters.")
    
    light_blue = '#91bfdb'
    point_scale = 400
    
    # Filter inputd data...
    # snow_df is only the cases where we ran a snow calc
    snow_df = results_df[~results_df["total_latent_heat"].isna()]
    # unstable_df is all the cases where the liquid is not stratified
    unstable_df = results_df[results_df["N2"] < 0.0]
    # Wrong dtdr is too cold (colder at the bottom than the top)
    wrong_dtdr_df = results_df[results_df["dT_cond"] < 0.0]
    # melting_df is above liquidus at ICB
    melting_df = results_df[results_df["dT_liq"] < 0.0]
     
    ax.axhline(c='k', lw=0.5, zorder=0)
    ax.axvline(c='k', lw=0.5, zorder=0)
    ax.scatter(unstable_df["dx"], unstable_df["dt"], s=40, facecolors='none', 
               edgecolors='k')
    ax.scatter(wrong_dtdr_df["dx"], wrong_dtdr_df["dt"], s=15, facecolors='none', 
               edgecolors='b')
    ax.scatter(melting_df["dx"], melting_df["dt"], s=15, facecolors='none', 
               edgecolors='r')
    c = ax.scatter(snow_df["dx"], snow_df["dt"], 
                   c=np.array(snow_df["total_latent_heat"])/1.0E12,
                  norm=colors.Normalize(vmin=0.0, vmax=34), 
                  cmap='viridis',
                  s=snow_df["max_particle_radius"]*point_scale)  
    ax.set_ylabel("$\Delta T_{ICB}$")
    ax.set_xlabel("$\Delta X_{ICB}$")

    ax.set_xlim(-0.003, 0.0155)
    ax.set_ylim(-80, 105)
    fig.colorbar(c, label='Latent heat from snow (TW)', extend='max', 
                 location='right',
                 shrink=0.5, anchor=(0.0,0.9))
    
    # Size scale
    ax.scatter(0.0165, -20, s=0.001*point_scale, c='k', clip_on=False)
    ax.text(0.0175, -20, "1 mm")
    ax.scatter(0.0165, -30, s=0.01*point_scale, c='k', clip_on=False)
    ax.text(0.0175, -30, "1 cm")
    ax.scatter(0.0165, -40, s=0.1*point_scale, c='k', clip_on=False)
    ax.text(0.0175, -40, "10 cm")
    ax.scatter(0.0165, -50, s=1.0*point_scale, c='k', clip_on=False)
    ax.text(0.0175, -50, "1 m")
    
    # Why do the squares need to be offset?
    if marker is not None:
        ax.plot(marker_x, marker_t, marker=marker, markerfacecolor='none',
                markeredgecolor=marker_color, ms=np.sqrt(0.15*point_scale))
   
    
    if target_latent_heat is not None:
        heat_line_dt, heat_line_dx = _get_dt_dx_latent_heat(results_df, 
                                                           target_latent_heat)
        ax.plot(heat_line_dx, heat_line_dt, 'r:', zorder=-1)
        
    if target_density_excess is not None:
        density_line_dt, density_line_dx = _get_dt_dx_excess_density(results_df, 
                                                            target_density_excess)
        ax.plot(density_line_dx, density_line_dt, 'k:', zorder=-1)
        
        
    plt.show()
    
    
def _get_dt_dx_both(df, target_latent_heat, target_density_excess):
    
    results = []
    errors = []
    
    heat_line_dt, heat_line_dx = _get_dt_dx_latent_heat(df, target_latent_heat)
    density_line_dt, density_line_dx = _get_dt_dx_excess_density(df,
                                                                target_density_excess)
    
    for lx, lt in zip(heat_line_dx, heat_line_dt):
        for rhox, rhot in zip(density_line_dx, density_line_dt):
                if lx == rhox and lt == rhot:
                    rx = lx
                    rt = lt
                    
    return rx, rt
                    
    
def _get_dt_dx_latent_heat(df, target_latent_heat):
    df["dlat_abs"] = abs((df.total_latent_heat.fillna(0)) - target_latent_heat)
    heat_line_dt = []
    heat_line_dx = []
    for this_dx in df.dx.unique():
        this_dx_df = df[df.dx == this_dx]
        # Skip the places where we didn't run any
        if this_dx_df.dlat_abs.min() < target_latent_heat:
            heat_line_dt.append(float(this_dx_df[this_dx_df.dlat_abs == 
                                                 this_dx_df.dlat_abs.min()].dt))
            heat_line_dx.append(this_dx)
    return heat_line_dt, heat_line_dx

def _get_dt_dx_excess_density(df, target_density_excess):
    df["drho_abs"] = abs((df.liquid_density_excess + 
                          df.max_solid_excess_density.fillna(0)) 
                         - target_density_excess)
    density_line_dt = []
    density_line_dx = []
    for this_dt in df.dt.unique():
        this_dt_df = df[df.dt == this_dt]
        density_line_dx.append(float(this_dt_df[this_dt_df.drho_abs == 
                                                this_dt_df.drho_abs.min()].dx))
        density_line_dt.append(this_dt)
    return density_line_dt, density_line_dx