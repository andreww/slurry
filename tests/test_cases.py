# Test cases that confirm that we reproduce
# results from the 2025 paper

import yaml
import numpy as np
import pytest

import layer_models


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide") 
def test_case():

    # This is extracted from the code we used to manage the big parameter sweep,
    # so the input is a bit eccentric and split into things we (used to) sweep over
    # and things we kept fixed for a whole sweep...

    # Our 'sweep' parameters (highlighted light blue square in the overview figure 8):
    case_name = 'test'
    delta_t_icb = -10.0
    delta_x_icb = 0.008125 # rounded in table
    this_i0 = None # Take it from the yaml

    # Other model parameters as yaml:
    input_data = """
    f_layer_thickness : 200000
    xfe_outer_core : 0.83
    growth_prefactor : 150.0
    i0 : 1.000000E-11
    surf_energy : 1.08
    wetting_angle : 5.0
    number_of_analysis_points : 100
    number_of_knots : 5
    r_icb : 1221500
    r_cmb : 3480000
    gruneisen_parameter : 1.5
    chemical_diffusivity : 1.000000E-09
    kinematic_viscosity : 2.000000E-06
    thermal_conductivity : 100.0
    max_time : 1.0E20
    hetrogeneous_radius : 10.0e-10
    """

    input_params = yaml.load(input_data, yaml.CLoader)

    # Run the model
    # Note that if BV freq is imaginary, or the case fails to run, 
    # output_data will be None...
    cases_dict, output_data = layer_models.run_case(case_name,
        delta_t_icb, delta_x_icb, this_i0, input_params)

    # Key results ("light blue square" row of table 2 of the SI)
    qf = 6.28 * 1.0E12
    delta_rho = 26.7 # We don't (easily) have the total. Only the solid part
    max_particle_size = 0.0335 
    icb_growth_rate = 1.01
    max_i = 8.93e-12 
    max_phi = 1.37e-10

    # Fairly relaxed absolute tolerances - lots of rounding in the table
    np.testing.assert_allclose(cases_dict["total_latent_heat"], qf, atol=1.0E10, err_msg='Qf mismatch')
    np.testing.assert_allclose(cases_dict["max_particle_radius"], max_particle_size, atol=2.0E-2, err_msg='rp mismatch')
    np.testing.assert_allclose(cases_dict["max_solid_volume_fraction"], max_phi, atol=2.0E-12, err_msg='phi mismatch')
    np.testing.assert_allclose(cases_dict["max_nucleation_rate"], max_i, atol=0.02, err_msg='Imax mismatch')    