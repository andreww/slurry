#!/usr/bin/env python

import os
import pickle
import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt

import flayer
import particle_evolution
import feo_thermodynamics as feot
import multiprocessing
import bulk_case_runner

parameters = {'f_layer_thickness': 200.0E3,
              'delta_t_icb': 10.0,
              'xfe_outer_core': 0.95,
              'xfe_icb': 0.95,
              'growth_prefactor': 150.0,
              'i0': 1.0E-10,
              'surf_energy': 1.08E-2,
              'wetting_angle': 180.0,
              'hetrogeneous_radius': None,
              'number_of_analysis_points': 200,
              'r_icb': 1221.5E3, 
              'r_cmb': 3480.0E3,
              'gruneisen_parameter': 1.5,
              'chemical_diffusivity': 1.0E-9,
              'kinematic_viscosity': 1.0E-6,
              'verbose': True}

#bulk_case_runner.plot_case_setup(**parameters)

filename = f"tmp3.pickle"
if os.path.exists(filename):
    # We have this model run on disk. Just read...
    print("Reading result cache!")
    data = bulk_case_runner.load_case_data(filename)    
else:
    # Run this model case
    print("Running model!")
    data = bulk_case_runner.run_flayer_case(parameters, filename)

print(f"Chemical diffusivity: {data['chemical_diffusivity']:.3g}, kinematic viscosoity: {data['kinematic_viscosity']:.2e}")
print(f"Inner core growth rate is {data['growth_rate']:.3e} km/Myr")
print(f"Max vf_ratio is {np.nanmax(data['vf_ratio']):.3e}")
if data['growth_rate'] > 1.0E-16:
    max_extra_o = np.max(feot.mass_percent_o(data["opt_xl"]) -  feot.mass_percent_o(data["xfe_outer_core"]))
    print(f"Max extra O in liquid {max_extra_o:.2e} % O by mass")
    max_particle_radius = data["particle_radii"][data["particle_radii"] > 0.0].max()
    min_particle_radius = data["particle_radii"][data["particle_radii"] > 0.0].min()
    print(f"Particle radii between {max_particle_radius:.3g} and {min_particle_radius:.3g} m")

#max_vf_ratio.append(np.nanmax(data['vf_ratio']))
#icb_flux.append(data["growth_rate"])
#max_particle_size.append(max_particle_radius)
#max_excess_oxygen_mass.append(max_extra_o)

#bulk_case_runner.plot_case_csd_nuc(**data)
#bulk_case_runner.plot_case_solid_frac(**data)
print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n\n")
