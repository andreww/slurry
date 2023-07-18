#!/usr/bin/env python

import os
import pickle

import yaml
import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt

import flayer
import particle_evolution
import feo_thermodynamics as feot
import multiprocessing
import bulk_case_runner


def main(parameters, filename=None):

    print("Running model!")
    data = bulk_case_runner.run_flayer_case(parameters, filename)

    print(f"Chemical diffusivity: {data['chemical_diffusivity']:.3g}, kinematic viscosoity: {data['kinematic_viscosity']:.2e}")
    print(f"Inner core growth rate is {data['growth_rate']:.3e} km/Myr")
    print(f"Max vf_ratio is {np.nanmax(data['vf_ratio']):.3e}")
    if data['growth_rate'] > 1.0E-16:
        max_extra_o = np.max(feot.mass_percent_o(data["out_x_points"]) -  feot.mass_percent_o(data["xfe_outer_core"]))
        print(f"Max extra O in liquid {max_extra_o:.2e} % O by mass")
        max_particle_radius = data["particle_radii"][data["particle_radii"] > 0.0].max()
        min_particle_radius = data["particle_radii"][data["particle_radii"] > 0.0].min()
        print(f"Particle radii between {max_particle_radius:.3g} and {min_particle_radius:.3g} m")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--picklefile", type=str)
    parser.add_argument("yaml_input", type=str)
    args = parser.parse_args()

    with open(args.yaml_input, 'r') as f:
        input_params = yaml.load(f, yaml.CLoader)


    main(input_params, filename=args.picklefile)
