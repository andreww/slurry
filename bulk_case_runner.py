import pickle
import warnings
# We have some annoying warnings - I think from llvm bug - fix then remove this
warnings.filterwarnings("ignore")

import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import matplotlib.colors as mplc

import layer_optimize
import particle_evolution
import feo_thermodynamics as feot

def case_handler(case_input):
    name = case_input[0]
    parameters = case_input[1]
    try:
        data = run_flayer_case(parameters, name+'.pickle')
        print(f"{name}: growth rate {data['growth_rate']} km/Myrm ax vf_ratio {data['vf_ratio'].max()}")
    except Exception as err:
        print(f"{name} failed! Exception was {err}") 
    
    
def run_flayer_case(input_data, filename=None):
    """
    Run a single f-layer case with input from dictionary, optionally save
    
    input_data: a dictionary of input data 
    filename: optional string, if present save case to file
    
    Returns
    output_data: a dictionary of input and output data"""
    
    solutions, analysis_radii, particle_densities, calculated_seperation, solid_vf, \
        particle_radii, partial_particle_densities, growth_rate, crit_nuc_radii, nucleation_rates, \
        vf_ratio, out_x_points, out_t_points, opt_params = layer_optimize.flayer_case(**input_data)
    
    output_data = dict(input_data)
    
    output_data["solutions"] = solutions
    output_data["analysis_radii"] = analysis_radii
    output_data["particle_densities"] = particle_densities
    output_data["calculated_seperation"] = calculated_seperation
    output_data["solid_vf"] = solid_vf
    output_data["particle_radii"] = particle_radii
    output_data["partial_particle_densities"] = partial_particle_densities
    output_data["growth_rate"] = growth_rate
    output_data["crit_nuc_radii"] = crit_nuc_radii
    output_data["nucleation_rates"] = nucleation_rates
    output_data["vf_ratio"] = vf_ratio
    output_data["out_x_points"] = out_x_points
    output_data["out_t_points"] = out_t_points
    output_data["opt_params"] = opt_params
    
    if filename is not None:
        with open(filename, 'wb') as f:
            pickle.dump(output_data, f)
    
    return output_data


def load_case_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_case_single_solution(index, data):
    """
    Create plot of time evolution for nucleation from a single depth, given case data and index
    
    index: integer nucleation index
    data: dictioary of compleated f-layer case. If a string read from pickle file
    """     
    tfunc, tafunc, xfunc, pfunc, gfunc = flayer.setup_flayer_functions(**data)
    xl_func = spi.interp1d(data["analysis_radii"], data["opt_xl"], fill_value='extrapolate')

    particle_evolution.plot_particle_evolution_time(data["solutions"][index], xl_func, tfunc, pfunc,            
                                                    data['chemical_diffusivity'],
                                                    data['growth_prefactor'], gfunc, data['kinematic_viscosity'])
    
 
def plot_case_csd_nuc(particle_radii, analysis_radii, partial_particle_densities,
                      crit_nuc_radii, nucleation_rates, logscale=False, nonuc=True, nosum=True, **other_data):

    max_particle_radius = particle_radii[particle_radii > 0.0].max()
    min_particle_radius = particle_radii[particle_radii > 0.0].min()
    print("Particle radii between {:.3g} and {:.3g} m".format(max_particle_radius, min_particle_radius))

    particle_size_distributions = []
    edges = None
    binsin = np.linspace(min_particle_radius, max_particle_radius, 10)

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

    def _sciformat(x, pos=None):
        if x == 0:
            return "0.0"
        scistr = "{:E}".format(x)
        vals = scistr.split('E')
        fmttick = "${:.1f}".format(float(vals[0])) + r"\times 10^{" + "{}".format(int(vals[1])) + "}$"
        return fmttick

    if nonuc:
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,8), sharex='col', tight_layout=True)
    else:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,8), sharex='col')
    fig.subplots_adjust(hspace=0, wspace=0.1)


    if nonuc:
        ax = axs[0]
    else:
        ax = axs[0,0]
    if logscale:
        im = ax.imshow(csd[1:,:].T, aspect='auto', interpolation='none', 
                   cmap=plt.get_cmap('Greys'), origin='lower', 
                   extent=[analysis_radii[1]/1000.0, analysis_radii[-1]/1000.0,
                           edges[0], edges[-1]], norm=mplc.LogNorm())
    else:
        im = ax.imshow(csd[1:,:].T, aspect='auto', interpolation='none', 
                   cmap=plt.get_cmap('Greys'), origin='lower', 
                   extent=[analysis_radii[1]/1000.0, analysis_radii[-1]/1000.0,
                           edges[0], edges[-1]])
    ax.set_ylabel('Particle radius (m)')
    #ax.set_xlabel('Radius (km)')
    ax.set_xlim(1221.5, 1471.5)
    #ax.set_yscale('log')

    cb = plt.colorbar(im, ax=ax, location='top')
    cb.set_label('Number of particles per m$^{3}$')
    #cb.ax.xaxis.set_major_locator()
    #cb.ax.xaxis.set_major_formatter(_sciformat)
    

    if nonuc:
        ax = axs[1]
    else:
        ax = axs[1,0]
    ax.plot(analysis_radii[1:-1]/1000.0, csd[1:-1,:].sum(axis=1))
    ax.set_xlabel('Radius (km)')
    ax.set_ylabel('Total number of particles per m$^3$')
    ax.yaxis.set_major_formatter(_sciformat)
    
    ax.set_xlim(1221.5, 1471.5)

    if not nonuc:
        ax = axs[0, 1]
        ax.plot(analysis_radii[1:-1]/1000.0, nucleation_rates[1:-1])
        ax.yaxis.set_major_formatter(_sciformat)
        ax.set_xlabel('Radius (km)')
        ax.set_ylabel('Nucleation rate (m$^{-3}$ s$^{-1}$)')
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position("right")
        #ax.set_yscale('log')

        ax = axs[1, 1]
        ax.plot(analysis_radii[1:-1]/1000.0, crit_nuc_radii[1:-1])
        ax.set_xlabel('Radius (km)')
        ax.set_ylabel('Critical radius for nucleation (m)')
        ax.set_yscale('log')
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position("right")

    plt.show()
    
def plot_case_setup(r_icb, r_cmb, f_layer_thickness, gruneisen_parameter, 
                    delta_t_icb, xfe_outer_core, xfe_icb, number_of_analysis_points, t_params, **kwargs):
    
    # Generate the functions for temperautre,
    # composition, pressure and gravity
    
    # Derived values of use
    r_flayer_top = r_icb + f_layer_thickness
        
    # Discretisation points
    nucleation_radii = np.linspace(r_icb, r_flayer_top, number_of_analysis_points)
    analysis_radii = np.linspace(r_icb, r_flayer_top, number_of_analysis_points)
    
    tfunc, atfunc, ftfunc, tfunc_creator, xfunc, pfunc, \
        gfunc = flayer.setup_flayer_functions(r_icb, r_cmb, f_layer_thickness, 
                                              gruneisen_parameter, delta_t_icb,
                                              xfe_outer_core, xfe_icb, analysis_radii)
    ftfunc = tfunc_creator(t_params)

    print("Temperature at CMB is", tfunc(r_cmb), "K")
    print("Temberature at top of F-layer is", tfunc(r_icb+f_layer_thickness), "K")
    print("Temberature at ICB is", ftfunc(r_icb), "K")

    # Interpolate onto radius for plotting
    rs = np.linspace(r_icb, r_icb+500.0E3)
    ts = ftfunc(rs)
    ats = atfunc(rs)
    ps = pfunc(rs)
    xs = xfunc(rs)
    # Find the P-X dependent liquidus (storing the temperature at each point)
    tl = feot.find_liquidus(xs, ps)


    # Plot the F-layer setup alongside the liquidus
    fig, ax1 = plt.subplots(figsize=(6,6), tight_layout=True)

    color = 'tab:red'
    ax1.set_xlabel('Radius (km)')
    ax1.set_ylabel('Temperature (K)', color=color)
    ax1.plot(rs/1000.0, ts, color=color)
    ax1.plot(rs/1000.0, ats, color=color, ls='--')
    ax1.plot(rs/1000.0, tl, color='k', ls=':')

    ax1.tick_params(axis='y', labelcolor=color)
    #ax1.set_ylim([5900, 6200])

    ax2 = ax1.twinx()  

    color = 'tab:blue'
    ax2.set_ylabel('Pressure (GPs)', color=color)  
    ax2.plot(rs/1000.0, ps, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([280, 330])

    fig.tight_layout()  
    plt.show()
    
def plot_case_solid_frac(analysis_radii, r_icb, r_cmb, f_layer_thickness, gruneisen_parameter, number_of_knots,
                    delta_t_icb, xfe_outer_core, xfe_icb, solid_vf, number_of_analysis_points, t_params, **kwargs):

    # Generate the functions for temperautre,
    # composition, pressure and gravity
    
    # Derived values of use
    r_flayer_top = r_icb + f_layer_thickness
        
    # Discretisation points
    nucleation_radii = np.linspace(r_icb, r_flayer_top, number_of_analysis_points)
    analysis_radii = np.linspace(r_icb, r_flayer_top, number_of_analysis_points)
    knott_radii = np.linspace(r_icb, r_flayer_top, number_of_knots)
    
    tfunc, atfunc, ftfunc, tfunc_creator, xfunc, pfunc, \
        gfunc = flayer.setup_flayer_functions(r_icb, r_cmb, f_layer_thickness, 
                                              gruneisen_parameter, delta_t_icb,
                                              xfe_outer_core, xfe_icb, knott_radii)
    tfunc = tfunc_creator(t_params)

    fig, axs = plt.subplots(ncols=2, figsize=(12,6), tight_layout=True)
    ax = axs[0]
    ax.plot(analysis_radii[1:-1]/1000.0, solid_vf[1:-1], label='Non-equilibrium')

    ax.plot(analysis_radii[1:-1]/1000.0, feot.volume_fraction_solid(xfunc(analysis_radii[1:-1]), 
                                                                    pfunc(analysis_radii[1:-1]), 
                                                                    tfunc(analysis_radii[1:-1])), label='equilibriun')
    ax.set_yscale("log", nonpositive='clip')
    ax.set_ylabel('Volume fraction solid')
    ax.set_xlabel('Radius (km)')
    ax.legend()

    ax = axs[1]
    ax.plot(analysis_radii[1:-1]/1000.0, 
        solid_vf[1:-1] / feot.volume_fraction_solid(xfunc(analysis_radii[1:-1]), 
                                                                pfunc(analysis_radii[1:-1]), 
                                                                tfunc(analysis_radii[1:-1])))
    ax.set_ylabel('Fraction of equilibrium solid volume')
    ax.set_xlabel('Radius (km)')

    plt.show()

    
