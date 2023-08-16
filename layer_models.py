#!/usr/bin/env python
# coding: utf-8

# Script to run a bunch of layer models with
# defined temperature and composition profiles
# (quadratic profile) and append results to a 
# pandas data frame saved on disk.
import collections
import pathlib
import pickle

import yaml
import numpy as np
import pandas as pd

import flayer
import feo_thermodynamics as feot
import earth_model
import layer_setup

prem = earth_model.Prem()



def main(input_params, cases_f, outfile, outdir=None):
    """
    Loop over the cases, running each case in turn
    """
        
    for line in cases_f:
        cases_dict = collections.defaultdict(list) # We'te just going to add one
        words = line.split()
        case_name = words[0]
        delta_t_icb = float(words[1])
        delta_x_icb = float(words[2])
        this_i0 = None
        if len(words) > 3:
            this_i0 = float(words[3])
         
        # Mass of outer core - for o prod unit conversion
        m_oc = prem.mass(input_params["r_cmb"]/1000.0, r_inner=input_params["r_icb"]/1000.0)
        
        # Setup case
        r_icb = input_params["r_icb"]
        r_flayer_top = input_params["r_icb"] + input_params["f_layer_thickness"]
        
        # Discretisation points
        nucleation_radii = np.linspace(input_params["r_icb"], r_flayer_top, 
                                       input_params["number_of_analysis_points"])
        analysis_radii = np.linspace(input_params["r_icb"], r_flayer_top,
                                     input_params["number_of_analysis_points"])
        knott_radii = np.linspace(input_params["r_icb"], r_flayer_top,
                                  input_params["number_of_knots"])

        # Reuse general layer setup code, but we don't need many of the functions
        # so we never need to rerun this (fairly expensive) function
        _, adiabatic_temperature_function, _, _, _, pressure_function, gravity_function, _ \
            = layer_setup.setup_flayer_functions(input_params["r_icb"], input_params["r_cmb"],
                                                 input_params["f_layer_thickness"], 
                                                 input_params["gruneisen_parameter"], 10, 
                                                 input_params["xfe_outer_core"],
                                                 input_params["xfe_outer_core"], knott_radii)


        temperature_function = layer_setup.fit_quad_func_boundaries(
            r_icb, r_flayer_top, adiabatic_temperature_function(r_icb)+delta_t_icb, 
            adiabatic_temperature_function(r_flayer_top))
        composition_function = layer_setup.fit_quad_func_boundaries(
            r_icb, r_flayer_top, input_params["xfe_outer_core"]+delta_x_icb, input_params["xfe_outer_core"])
        
        # Check the temperature is below the liquidus temperature at the ICB
        t_icb = temperature_function(r_icb)
        x_icb = composition_function(r_icb)
        tl = feot.find_liquidus(composition_function(r_icb), pressure_function(r_icb))
        dt_liq = tl - t_icb
        
        # Check temperature increases downwards
        t_flayer_top = temperature_function(r_flayer_top)
        dt_cond = t_icb - t_flayer_top
        
        # Check the layer is stratified
        Nbv, N2 = layer_setup.estimate_brunt_vaisala_frequency(
            r_flayer_top, r_icb, temperature_function, adiabatic_temperature_function,
            composition_function, gravity_function, pressure_function)
        
        cases_dict["case"].append(case_name)
        cases_dict["dt"].append(delta_t_icb)
        cases_dict["dx"].append(delta_x_icb)
        cases_dict["N2"].append(N2)
        cases_dict["dT_liq"].append(dt_liq)
        cases_dict["dT_cond"].append(dt_cond)
        print(f"Dt = {delta_t_icb}, delta_x = {delta_x_icb}, BV freq = {Nbv}, dT_liq = {dt_liq}, dT_cond = {dt_cond}")
        if (N2 >= 0.0) and (dt_liq > 0.0) and (dt_cond > 0.0):
            
            k0 = input_params["growth_prefactor"]
            dl = input_params["chemical_diffusivity"]
            k = input_params["thermal_conductivity"]
            mu = input_params["kinematic_viscosity"]
            if this_i0 is None:
                i0 = input_params["i0"]
            else:
                i0 = this_i0
                input_params["i0"] = i0 # So it gets stored right in the output
            surf_energy = input_params["surf_energy"]
            wetting_angle = input_params["wetting_angle"]
            hetrogeneous_radius = input_params["hetrogeneous_radius"]
            
            try:
                solutions, particle_densities, growth_rate, solid_vf, \
                particle_radii, partial_particle_densities, crit_nuc_radii, \
                nucleation_rates, _, _, total_latent_heat, total_o_rate = flayer.evaluate_flayer(
                    temperature_function, composition_function, pressure_function, gravity_function,
                    0.0, 1.0E20, k0, dl, k, mu, i0, surf_energy, wetting_angle, hetrogeneous_radius,
                    nucleation_radii, analysis_radii, r_icb, 
                    r_flayer_top, Nbv, verbose=False, silent=True, diffusion_problem=False)
            except (AssertionError, ValueError) as error:
                print("Something went wrong in this point:")
                print(error)
                cases_dict["total_latent_heat"].append(None)
                cases_dict["total_o_rate"].append(None)
                cases_dict["max_particle_radius"].append(None)  
                cases_dict["max_solid_volume_fraction"].append(None)
                cases_dict["max_nucleation_rate"].append(None)

            else:
            
                output_data = dict(input_params)
    
                output_data["solutions"] = solutions
                output_data["particle_densities"] = particle_densities
                output_data["growth_rate"] = growth_rate
                output_data["solid_vf"] = solid_vf
                output_data["particle_radii"] = particle_radii
                output_data["partial_particle_densities"] = partial_particle_densities
                output_data["crit_nuc_radii"] = crit_nuc_radii
                output_data["nucleation_rates"] = nucleation_rates
                output_data["total_latent_heat"] = total_latent_heat
                output_data["total_o_rate"] = total_o_rate
                output_data["Nbv"] = Nbv
                
                output_data["analysis_radii"] = analysis_radii
            
                if outdir is not None:
                    case_file = case_name + ".pkl"
                    with open(outdir/case_file, 'wb') as f:
                        pickle.dump(output_data, f)
            
                print(f"Heat from crystalisation = {total_latent_heat/1.0E12} TW")
                print(f"Oxygen production rate = {total_o_rate/1.0E9} Tg/s")
            
                oc_enrichment_rate = ((total_o_rate*60.0*60.0*24.0*365.0*1.0E9)/m_oc)
                print(f"Oxygen enrichment rate of outer core = {oc_enrichment_rate*100.0} wt. % Oxygen / Ga")
            
                max_particle_radius = particle_radii.max()
                print(f"maximum particle radius = {max_particle_radius} m")
                
                max_sold_volume_fraction = solid_vf.max()
                print(f"maximum solid fraction = {max_sold_volume_fraction}")
                
                max_nucleation_rate = np.nanmax(nucleation_rates) # nan in region of no nuc, which is OK
                print(f"max_nucleation_rate = {max_nucleation_rate}")
            
                cases_dict["total_latent_heat"].append(total_latent_heat)
                cases_dict["total_o_rate"].append(total_o_rate)
                cases_dict["max_particle_radius"].append(max_particle_radius)
                cases_dict["max_solid_volume_fraction"].append(max_sold_volume_fraction)
                cases_dict["max_nucleation_rate"].append(max_nucleation_rate)
            
        else:
            cases_dict["total_latent_heat"].append(None)
            cases_dict["total_o_rate"].append(None)
            cases_dict["max_particle_radius"].append(None)
            cases_dict["max_solid_volume_fraction"].append(None)
            cases_dict["max_nucleation_rate"].append(None)

        # Append the summary results
        new_df = pd.DataFrame(cases_dict)
        if outfile.exists():
            # Open old file and create data frame
            old_df = pd.read_csv(outfile)
            joint_df = old_df.append(new_df)
            # Mode is w - overwrites
            joint_df.to_csv(outfile, index=False)
        else:
            new_df.to_csv(outfile, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--picklefiledir", type=str, help="Directory for detailed output files")
    parser.add_argument("yaml_input", type=str, help="YAML file specifying all snow parameters")
    parser.add_argument("cases", type=str, help="Text file specifying each case (two columns, dT from adiabat and dX from well mixed)")
    parser.add_argument("output_f", type=str, help="CSV file for summary output data, will append if exists")
    args = parser.parse_args()

    with open(args.yaml_input, 'r') as f:
        input_params = yaml.load(f, yaml.CLoader)
        
    if args.picklefiledir:
        outdir = pathlib.Path(args.picklefiledir)
        if outdir.exists():
            assert outdir.is_dir(), "picklefiledir must be a directory"
        else:
            # like mkdir -p
            outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = None # We'll not save details
    
    outfile = pathlib.Path(args.output_f)
    
    with open(args.cases, 'r') as cases_f:
        main(input_params, cases_f, outfile, outdir=outdir)



