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

import flayer
import feo_thermodynamics as feot
import earth_model
import layer_setup
import bulk_case_runner


# Function and supporting tools to make grid of points in dt dx space
# note _get... functions below are there to help with the lines

def plot_matches(matches_df, fig=None, ax=None, noplot=False):
    """
    Create a summary plot.
    """
    if (fig is None) != (ax is None):
        raise ValueError("Must specify both fig and ax or have them both created.")
    if (fig is None) and (ax is None):
        # Remove 0.15 fraction of the width to match the other plot...
        fig, ax = plt.subplots(figsize=(7-(7.0*0.15),6))
 
    point_scale = 100
    
     
    ax.axhline(c='k', lw=0.5, zorder=0)
    ax.axvline(c='k', lw=0.5, zorder=0)
    ax.set_xlim(-0.003, 0.0155)
    ax.set_ylim(-80, 105)

    for index, row in matches_df.iterrows():
        # We need to reach into the full output for some info... grrr
        full_data = bulk_case_runner.load_case_data(row["full_file"])
        
        ax.plot(row.dx, row["dt"], marker=row.marker, markerfacecolor='none',
                markeredgecolor=row.color)#, ms=np.sqrt(full_data["growth_rate"]*point_scale))
        
    ax.set_ylabel("$\Delta T_{ICB}$")
    ax.set_xlabel("$\Delta X_{ICB}$")

    ax1 = inset_axes(ax, width="40%", height="60%", loc=2, borderpad=1)
    ax1.set_yscale('log')
    #ax1.xaxis.set_label_position('top')
    ax1.yaxis.set_label_position('right') 
    ax1.yaxis.tick_right()
    #ax1.xaxis.tick_top()
    ax1.set_ylabel("Max nucleation rate (m$^{-3}$s$^{-1}$)")
    ax1.set_xlabel("Maximum particle radius (m)")
    point_scale = 100
    
    texts = []
    for index, row in matches_df.iterrows():
        
        # We need to reach into the full output for some info... grrr
        full_data = bulk_case_runner.load_case_data(row["full_file"])        
        ax1.plot(row["max_particle_radius"], row["max_nucleation_rate"], marker=row.marker, markerfacecolor='none',
                markeredgecolor=row.color)#, ms=np.sqrt(full_data["growth_rate"]*point_scale))
        
        t = ax1.annotate(f"{full_data['growth_rate']:.1f}", (row["max_particle_radius"], row["max_nucleation_rate"]),
                        color=row.color)
        texts.append(t)
        
    adjustText.adjust_text(texts)


    if not noplot:
        plt.show()

def plot_summary_figure(results_df, target_latent_heat=None, 
                        target_density_excess=None, 
                        fig=None, ax=None, marker_x=None,
                        marker_t=None, marker=None, marker_color=None,
                        include_fails=True, noplot=False, cax=None):
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
    if include_fails:
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
    if cax is None:
        fig.colorbar(c, label='Latent heat from snow (TW)', extend='max', 
                     location='right',
                     shrink=0.5, anchor=(0.0,0.9))
    else:
        fig.colorbar(c, label='Latent heat from snow (TW)', extend='max', 
                     cax=cax, shrink=0.5, anchor=(0.0,0.9))
    
    # Size scale
    if include_fails:
        ax.scatter(0.0165, -20, s=0.001*point_scale, c='k', clip_on=False)
        ax.text(0.0175, -20, "1 mm")
        ax.scatter(0.0165, -30, s=0.01*point_scale, c='k', clip_on=False)
        ax.text(0.0175, -30, "1 cm")
        ax.scatter(0.0165, -40, s=0.1*point_scale, c='k', clip_on=False)
        ax.text(0.0175, -40, "10 cm")
        ax.scatter(0.0165, -50, s=1.0*point_scale, c='k', clip_on=False)
        ax.text(0.0175, -50, "1 m")
    else:
        ax.scatter(0.001, 90, s=0.001*point_scale, c='k', clip_on=False)
        ax.text(0.00175, 90, "1 mm")
        ax.scatter(0.001, 80, s=0.01*point_scale, c='k', clip_on=False)
        ax.text(0.00175, 80, "1 cm")
        ax.scatter(0.001, 70, s=0.1*point_scale, c='k', clip_on=False)
        ax.text(0.00175, 70, "10 cm")
        ax.scatter(0.001, 60, s=1.0*point_scale, c='k', clip_on=False)
        ax.text(0.00175, 60, "1 m")
    
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
        
    if not noplot:    
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


# Plotting for temperature and composition

def make_layer_plot(dt, dx, xfe_outer_core, r_icb=1221.5E3, 
                    gruneisen_parameter=1.5, r_cmb=3480.0E3, 
                    f_layer_thickness=200_000, number_of_analysis_points=100,
                    number_of_knots=5, fig=None, ax=None):

    # Setup the f-layer

    # Derived radii
    r_flayer_top = r_icb + f_layer_thickness
         
    # Discretisation points
    nucleation_radii = np.linspace(r_icb, r_flayer_top, number_of_analysis_points)
    analysis_radii = np.linspace(r_icb, r_flayer_top, number_of_analysis_points)
    knott_radii = np.linspace(r_icb, r_flayer_top, number_of_knots)

    # Reuse general layer setup code, but we don't need many of the functions
    # so we never need to rerun this (fairly expensive) function
    _, adiabatic_temperature_function, _, _, _, pressure_function, \
        gravity_function, _ = layer_setup.setup_flayer_functions(
        r_icb, r_cmb, f_layer_thickness, gruneisen_parameter, 10, xfe_outer_core,
            xfe_outer_core, knott_radii)
    
    temperature_function = layer_setup.fit_quad_func_boundaries(
            r_icb, r_flayer_top, adiabatic_temperature_function(r_icb)+dt, 
            adiabatic_temperature_function(r_flayer_top))
    composition_function = layer_setup.fit_quad_func_boundaries(
            r_icb, r_flayer_top, xfe_outer_core+dx, xfe_outer_core)
    Nbv, N2, _ = layer_setup.estimate_brunt_vaisala_frequency(
            r_flayer_top, r_icb, temperature_function, adiabatic_temperature_function,
            composition_function, gravity_function, pressure_function)

    print(f"BV freq = {Nbv}")
    rs = np.linspace(r_icb, r_flayer_top+100.0E3)

    # Check PREM works ... and print some interesting values
    prem = earth_model.Prem()
    print("Pressure at ICB:", prem.pressure(r_icb/1000.0), "GPa")
    print("Pressure at top of F-layer", prem.pressure(r_flayer_top/1000.0), "GPa")
    print("g at ICB:", prem.gravity(r_icb/1000.0), "m/s**2")
    print("g at top of F-layer", prem.gravity(r_flayer_top/1000.0), "m/s**2")

    # Find the liquidus
    @np.vectorize
    def full_temperature_function(r):
        if r < r_flayer_top:
            return temperature_function(r) 
        else:
            return adiabatic_temperature_function(r)
        
    @np.vectorize
    def full_composition_function(r):
        if r < r_flayer_top:
            return composition_function(r) 
        else:
            return xfe_outer_core 
    
    tl = feot.find_liquidus(full_composition_function(rs), pressure_function(rs))

    # Find the liquid density

    liquid_density, _, _, solid_density, _, _ = feot.densities(
        composition_function(rs), pressure_function(rs), temperature_function(rs))

    # This is for a well mixed core extended down 
    adiabatic_liquid_density, _, _, adiabatic_solid_density, _, _ = feot.densities(
        xfe_outer_core, pressure_function(rs), adiabatic_temperature_function(rs))
    
    print("Temperature at CMB:", full_temperature_function(r_cmb), "K")
    print("Liquid density at ICB:", liquid_density[0], "Kg/m^3")
    print("Liquid density at top of F-layer", liquid_density[-1], "Kg/m^3")
    print("Liquid density along adiabat at ICB:", 
          adiabatic_liquid_density[0], "Kg/m^3")
    print("Liquid density along adiabat at top of F-layer",
          adiabatic_liquid_density[-1], "Kg/m^3")
    print("Liquid density difference at ICB:", 
          liquid_density[0]-adiabatic_liquid_density[0], "Kg/m^3")

    # Plot the F-layer setup alongside the liquidus
    if (fig is None) != (ax is None):
        raise ValueError("Must specify both fig and ax or have them both created.")
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(figsize=(6,6))
    ax_in = inset_axes(ax, width="35%", height="35%", loc=1, borderpad=1)
    
    color = 'tab:red'
    ax.set_xlabel('Radius (km)')
    ax.set_ylabel('Temperature (K)', color=color)
    ax.plot(rs/1000.0, full_temperature_function(rs), color=color)
    ax.plot(rs/1000.0, adiabatic_temperature_function(rs), color=color, ls='--')
    ax.plot(rs/1000.0, tl, color='k', ls=':')

    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylim([4940, 5140])
 

    #color = 'tab:blue'
    #ax2.set_xlabel('Radius (km)')
    #ax2.set_ylabel('Pressure (GPs)', color=color)  
    #ax2.plot(rs/1000.0, pressure_function(rs), color=color)
    #ax2.tick_params(axis='y', labelcolor=color)
    #ax2.set_ylim([310, 330])

    color = 'tab:green'
    ax_in.set_xlabel('Radius (km)')
    ax_in.set_ylabel('Composition (mol. frac. Fe)', color=color)  
    ax_in.plot(rs/1000.0, full_composition_function(rs), color=color)
    ax_in.plot(rs/1000.0, np.ones_like(rs)*xfe_outer_core, color=color, ls='--')
    ax_in.tick_params(axis='y', labelcolor=color)
    ax_in.set_ylim([0.828, 0.842])

    #ax4.set_xlabel('Radius (km)')
    #ax4.set_ylabel('Liquid density (kg/m^3)')  
    #ax4.plot(rs/1000.0, liquid_density)
    #ax4.plot(rs/1000.0, adiabatic_liquid_density, ls='--')
    #ax4.plot(rs/1000.0, solid_density, ls=':')
    #ax4.tick_params(axis='y')

    plt.show()
    
    
def _sciformat(x, pos=None):
    if x == 0:
        return "0.0"
    scistr = "{:E}".format(x)
    vals = scistr.split('E')
    fmttick = "${:.1f}".format(float(vals[0])) + r"\times 10^{" + "{}".format(int(vals[1])) + "}$"
    return fmttick


def plot_case_figure(data):
    
    prem = earth_model.Prem()
    m_oc = prem.mass(3480.0, r_inner=1221.5)
    s_per_ga = 60.0*60.0*24.0*365.0*1.0E9
    
    fig, ax = plt.subplots(figsize=(7,6))
   
    plot_lines(ax, **data)
    
    ax2 = ax.twinx() 
    ax2.plot(data['analysis_radii'][1:]/1000.0, data['solid_vf'][1:], 'g--')
    ax2.set_ylabel('Volume fraction solid', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.yaxis.set_major_formatter(_sciformat)

    ax_ins = inset_axes(ax, width="30%", height="35%", loc=1, borderpad=1)
    im = plot_csd(ax_ins, **data)
    
    #cbax = fig.add_axes((0.5,0.5,0.1,0.1))
    cb = plt.colorbar(im, ax=ax, location='bottom',
                 shrink=0.5, anchor=(0.0,1.0))
    cb.set_label('Number of particles per m$^{3}$')
    
    plt.figtext(0.6, 0.21, f"Latent heat production: {data['total_latent_heat']/1E12:.1f} TW", figure=fig)
    #plt.figtext(0.6, 0.15, f"Oxygen production: {((data['total_o_rate']*s_per_ga)/m_oc)*100.0:.2g} wt.%/Ga", figure=fig)
    plt.figtext(0.6, 0.18, f"Inner core growth: {data['growth_rate']:.1f} km/Myr", figure=fig)
 
    plt.show()

def plot_lines(ax, analysis_radii, nucleation_rates, r_icb, f_layer_thickness, solid_vf,
               **other_data):
    ax.plot(analysis_radii/1000.0, nucleation_rates, 'k')
    ax.set_xlim(r_icb/1000.0, (r_icb+f_layer_thickness)/1000.0)
    ax.yaxis.set_major_formatter(_sciformat)
    ax.set_xlabel('Radius (km)')
    ax.set_ylabel('Nucleation rate (m$^{-3}$ s$^{-1}$)')
   
    
def plot_csd(ax, particle_radii, analysis_radii, partial_particle_densities, r_icb,
             f_layer_thickness, **other_data):
    
    
    
    max_particle_radius = particle_radii[particle_radii > 0.0].max()
    min_particle_radius = particle_radii[particle_radii > 0.0].min()
    #print("Particle radii between {:.3g} and {:.3g} m".format(max_particle_radius, min_particle_radius))

    particle_size_distributions = []
    edges = None
    binsin = np.linspace(0, max_particle_radius, 11)

    for i, r in enumerate(analysis_radii):
        csd, edg = np.histogram(particle_radii[i,:], 
                                weights=partial_particle_densities[i,:],
                                range=(min_particle_radius, max_particle_radius),
                                bins=binsin)
        if edges is None:
            edges = edg
        else:
            assert np.array_equal(edges, edg)
    
        particle_size_distributions.append(csd)

    csd = np.array(particle_size_distributions)

    im = ax.imshow(csd[1:,:].T, aspect='auto', interpolation='none', 
                   cmap=plt.get_cmap('Greys'), origin='lower', 
                   extent=[analysis_radii[1]/1000.0, analysis_radii[-1]/1000.0,
                           edges[0], edges[-1]], norm=colors.LogNorm())
        
    ax.set_ylabel('Particle radius (m)')
    ax.set_xlabel('Radius (km)')
    ax.set_xlim(r_icb/1000.0, (r_icb+f_layer_thickness)/1000.0)
    
    return im