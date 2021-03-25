#!/usr/bin/env python

import numpy as np
import scipy.optimize as spo

import thermodynamic_model


# Thermodynamic parameters!
#
# Things marked 'old model' are from Komabayashi and Fei 2010 (doi: 10.1029/2009/JB006442)
# Fe parameters (hcp and liquid) match those in Komabayashi 2020 
# (https://link.springer.com/article/10.1007/s00269-020-01102-w)

fe_fcc_a = 16300.921
fe_fcc_b = 381.47162
fe_fcc_c = -52.2754
fe_fcc_d = 0.000177578
fe_fcc_e = -395355.43
fe_fcc_f = -2476.28
fe_fcc_v0 = 6.82
fe_fcc_k0 = 163.4
fe_fcc_kp = 5.38
fe_fcc_a0 = 7.0e-5
fe_fcc_ag0 = 5.5
fe_fcc_k = 1.4

fe_hcp_a = 12460.921
fe_hcp_b = 386.99162
#fe_hcp_a = 14405.9211 # old model
#fe_hcp_b = 384.8716162 # old model
fe_hcp_c = -52.2754
fe_hcp_d = 0.000177578
fe_hcp_e = -395355.43
fe_hcp_f = -2476.28
fe_hcp_v0 = 6.753
fe_hcp_k0 = 163.4
fe_hcp_kp = 5.38
fe_hcp_a0 = 5.8e-5
fe_hcp_ag0 = 5.1
fe_hcp_k = 1.4

fe_liquid_a = -9007.3402
fe_liquid_b = 290.29866
fe_liquid_c = -46.0
fe_liquid_d = 0.0
fe_liquid_e = 0.0
fe_liquid_f = 0.0
fe_liquid_v0 = 6.88
fe_liquid_k0 = 148.0
fe_liquid_kp = 5.8
fe_liquid_a0 = 9.0e-5
fe_liquid_ag0 = 5.1
# fe_liquid_a0 = 9.2e-5 # old model
# fe_liquid_ag0 = 5.22 # old model
fe_liquid_k = 0.56 # see footnote c in table 2!

feo_solid_a = -279318.0
feo_solid_b = 252.848
feo_solid_c = -46.12826
feo_solid_d = -0.0057402984
feo_solid_e = 0.0
feo_solid_f = 0.0
feo_solid_v0 = 12.256
feo_solid_k0 = 149.0
feo_solid_kp = 3.83
feo_solid_a0 = 4.5e-5
feo_solid_ag0 = 4.25
feo_solid_k = 1.4

feo_liquid_a = -245310.0
feo_liquid_b = 231.879
feo_liquid_c = -46.12826
feo_liquid_d = -0.0057402984
feo_liquid_e = 0.0
feo_liquid_f = 0.0
feo_liquid_v0 = 13.16
feo_liquid_k0 = 128.0
feo_liquid_kp = 3.85
feo_liquid_a0 = 4.7e-5
feo_liquid_ag0 = 4.5
feo_liquid_k = 1.4


def liquid_free_energy(x_fe, p, t):
    """
    Free energy of the Fe-FeO liquid
    
    x_fe is mol fraction Fe (1.0 for pure Fe, 0.0 for pure FeO)
    p is pressure (in GPa)
    t is temperature (in K)
    Does not need to be within liquid stability field (but extrapolation
    of data may be in error).
    
    Returns free energy in J/mol (not kJ/mol)
    """
    x_feo = 1.0 - x_fe
    
    if x_fe != 0.0:
        mu_fe = thermodynamic_model.chemical_potential(x_fe, p, t, fe_liquid_a, fe_liquid_b, fe_liquid_c, 
                                   fe_liquid_d, fe_liquid_e, fe_liquid_f,
                                   fe_liquid_v0, fe_liquid_k0, fe_liquid_kp,
                                   fe_liquid_a0, fe_liquid_ag0, fe_liquid_k)
    
    if x_feo != 0.0:
        mu_feo = thermodynamic_model.chemical_potential(x_feo, p, t, feo_liquid_a, feo_liquid_b, feo_liquid_c, 
                                    feo_liquid_d, feo_liquid_e, feo_liquid_f,
                                    feo_liquid_v0, feo_liquid_k0, feo_liquid_kp,
                                    feo_liquid_a0, feo_liquid_ag0, feo_liquid_k)
    
    if x_fe == 0.0:
        g = thermodynamic_model.end_member_free_energy(p, t, feo_liquid_a, feo_liquid_b, feo_liquid_c, 
                                  feo_liquid_d, feo_liquid_e, feo_liquid_f,
                                  feo_liquid_v0, feo_liquid_k0, feo_liquid_kp,
                                  feo_liquid_a0, feo_liquid_ag0, feo_liquid_k)
    elif x_feo == 0.0:
        g = thermodynamic_model.end_member_free_energy(p, t, fe_liquid_a, fe_liquid_b, fe_liquid_c, 
                                   fe_liquid_d, fe_liquid_e, fe_liquid_f,
                                   fe_liquid_v0, fe_liquid_k0, fe_liquid_kp,
                                   fe_liquid_a0, fe_liquid_ag0, fe_liquid_k)
    else:
        g = x_fe*mu_fe + x_feo*mu_feo

    return g


# Avoid hand coding loops. 
liquid_free_energies = np.vectorize(liquid_free_energy)


def solid_free_energies(x_fe, p, t):
    """
    Free energy of the HCP Fe-FeO solid mixture and end members
    
    x_fe is mol fraction Fe (1.0 for pure Fe, 0.0 for pure FeO)
    p is pressure (in GPa)
    t is temperature (in K)
    Does not need to be within solid stability field (but extrapolation
    of data may be in error).
    
    Returns free energy of mechanical mixture, free energy of HCP Fe,
    and free energy of FeO in J/mol (not kJ/mol)
    """
    g_feo = thermodynamic_model.end_member_free_energy(p, t, feo_solid_a, feo_solid_b, feo_solid_c, 
                                                       feo_solid_d, feo_solid_e, feo_solid_f,
                                                       feo_solid_v0, feo_solid_k0, feo_solid_kp,
                                                       feo_solid_a0, feo_solid_ag0, feo_solid_k)
    g_fe = thermodynamic_model.end_member_free_energy(p, t, fe_hcp_a, fe_hcp_b, fe_hcp_c, 
                                                      fe_hcp_d, fe_hcp_e, fe_hcp_f,
                                                      fe_hcp_v0, fe_hcp_k0, fe_hcp_kp,
                                                      fe_hcp_a0, fe_hcp_ag0, fe_hcp_k)
    g_mixture = x_fe * g_fe + (1.0 - x_fe) * g_feo
    return g_mixture, g_fe, g_feo


def fe_liquid_chemical_potential(x_fe, p, t):
    """
    Chemical potential of Fe in Fe-FeO liquid
    
    x_fe is mol fraction Fe (1.0 for pure Fe, 0.0 for pure FeO)
    p is pressure (in GPa)
    t is temperature (in K)
    Does not need to be within liquid stability field (but extrapolation
    of data may be in error).
    
    Returns chemical potential of Fe in Fe-FeO in J/mol (not kJ/mol). Will return NaN
    for x_fe = 0.0
    """
    return thermodynamic_model.chemical_potential(x_fe, p, t, fe_liquid_a, fe_liquid_b, fe_liquid_c, 
                                                  fe_liquid_d, fe_liquid_e, fe_liquid_f,
                                                  fe_liquid_v0, fe_liquid_k0, fe_liquid_kp,
                                                  fe_liquid_a0, fe_liquid_ag0, fe_liquid_k)


def feo_liquid_chemical_potential(x_fe, p, t):
    """
    Chemical potential of FeO in Fe-FeO liquid
    
    x_fe is mol fraction Fe (1.0 for pure Fe, 0.0 for pure FeO)
    p is pressure (in GPa)
    t is temperature (in K)
    Does not need to be within liquid stability field (but extrapolation
    of data may be in error).
    
    Returns chemical potential of FeO in Fe-FeO in J/mol (not kJ/mol). Will return NaN
    for x_fe = 1.0
    """
    return thermodynamic_model.chemical_potential(1.0-x_fe, p, t, feo_liquid_a, feo_liquid_b, feo_liquid_c, 
                                                  feo_liquid_d, feo_liquid_e, feo_liquid_f,
                                                  feo_liquid_v0, feo_liquid_k0, feo_liquid_kp,
                                                  feo_liquid_a0, feo_liquid_ag0, feo_liquid_k)


def _delta_mu_fe_liquid(x, p, t, gsolid):
    return fe_liquid_chemical_potential(x, p, t) - gsolid


def _delta_mu_feo_liquid(x, p, t, gsolid):
    return feo_liquid_chemical_potential(x, p, t) - gsolid


def find_liquidus_compositions(p, t):
    """
    At some Ps and Ts we should have two liquidus compositions 
    (one on each side of the eutectic). These are the compositions
    for the liquid where mu_liquid == mu_solid for Fe or FeO, 
    respectivly
    """
    _, g_fe_solid, g_feo_solid = solid_free_energies(0.5, p, t)

    if _delta_mu_fe_liquid(1.0, p, t, g_fe_solid) <= 0.0:
        x_fe_side = 1.0 # above pure phase melting T
    else:
        x_fe_side = spo.brentq(_delta_mu_fe_liquid, 0.000000001, 1.0, args=(p, t, g_fe_solid))

    if _delta_mu_feo_liquid(0.0, p, t, g_feo_solid) <= 0.0:
        x_feo_side = 0.0 # above pure phase melting T
    else:
        x_feo_side = spo.brentq(_delta_mu_feo_liquid, 0.0, 0.999999999, args=(p, t, g_feo_solid))
        
    if x_fe_side < x_feo_side:
        # below eutectic?
        x_fe_side = 0.0
        x_feo_side = 1.0
        
    return x_fe_side, x_feo_side

@np.vectorize
def phase_relations_molar(x, p, t):
    """
    Details of the equilbrium assemblage
    
    For composition x (mole fraction Fe), p (GPa) and t (K) 
    return the mole fraction liquid (phi_lq), the mole fraction Fe 
    (phi_fe), the mole fraction FeO (phi_feo), and the liquid
    composition (as a mole fraction, x_lq)
    """
    # first get the liquidus compositions
    x_fe_side, x_feo_side = find_liquidus_compositions(p, t)
    
    if (x <= x_fe_side) and (x >= x_feo_side):
        # pure liquid
        phi_lq = 1.0
        x_lq = x
        phi_fe = 0.0
        phi_feo = 0.0
        phi_solid = 0.0
    elif x_fe_side < x_feo_side:
        # below eutectic
        phi_lq = 0.0
        x_lq = np.nan
        phi_fe = x
        phi_feo = 1.0 - x
        phi_solid = 1.0
    elif x > x_fe_side:
        # Fe + liquid
        x_lq = x_fe_side
        phi_lq = (1.0 - x) / (1.0 - x_lq)
        phi_fe = 1.0 - phi_lq
        phi_feo = 0.0
        phi_solid = phi_fe
    elif x < x_feo_side:
        # FeO + liquid
        x_lq = x_feo_side
        phi_lq = x / x_lq
        phi_feo = 1.0 - phi_lq
        phi_fe = 0.0
        phi_solid = phi_feo
    else:
        print("Error case:", x, p, t, x_fe_side, x_feo_side)
        assert False, "something went wrong"
        
    return x_lq, phi_fe, phi_lq, phi_feo, phi_solid