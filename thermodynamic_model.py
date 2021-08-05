#!/usr/bin/env python

import numpy as np
import scipy.optimize as spo
import numba


# cannot jit due to brentq. Can we
# rearange EOS in terms of V to avoid?
# NB: this is easily the biggest cost from profiling! 
# maybe just build an interpolation and fit this? See issues...
def numerical_vinet_eos_volume(p, v0, k0, kp):
    """
    Return the volume at some pressure
    
    Given 1 bar reference volume (v0), bulk modulus (k0),
    and its pressure derivative (kp) return the volume
    at pressure p. These should normally all be values
    at 298 K (thermal expansion is added later). Units 
    of p and k0 are GPa, v0 and the returned volume
    are cm^3/mol, and kp is dimensionless. The solution
    is via a root finding method.
    """
    p = spo.brentq(_pressure_error, 2.0, 20.0,
                   args=(v0, k0, kp, p))

    return p
    

# Avoid hand coding loops. Cannot numba vectorize 
# function due to brentq...
# Also crap hack to avoid warning. See
# https://github.com/andreww/slurry/issues/8
numerical_vinet_eos_volumes_func = np.vectorize(numerical_vinet_eos_volume)
def numerical_vinet_eos_volumes(p, v0, k0, kp):
    oldsettings = np.seterr(all='ignore')
    results = numerical_vinet_eos_volumes_func(p, v0, k0, kp)
    np.seterr(**oldsettings)
    return results

@numba.jit
def _pressure_error(v, v0, k0, kp, p_target):
    return vinet_eos_pressure(v, v0, k0, kp) - p_target
 
    
@numba.vectorize
def vinet_eos_volumes(p, v0, k0, kp):
    """
    Return the volume at some pressure
    
    Given 1 bar reference volume (v0), bulk modulus (k0),
    and its pressure derivative (kp) return the volume
    at pressure p. These should normally all be values
    at 298 K (thermal expansion is added later). Units 
    of p and k0 are GPa, v0 and the returned volume
    are cm^3/mol, and kp is dimensionless. 
    
    The solution is the direct inverse of an approximaton
    to the Vinnet EOS (the exact expression cannot be 
    inverted) as described in Etter and Dinnebier (2014;
    J. Applied Crys. 47:384-390 
    http://dx.doi.org/10.1107/S1600576713032287). This
    solution (there are four) is valid for k0 > 5 GPa,
    kp > 1.5 GPa and 0 < P < 100 GPa (at least). We also
    provide a numerical implementation to check and other
    solutions are given in the SI to that paper. No checks
    on the valididty are perfomed. 
    
    The major advantage of this approach is that Numba can
    compile it. This isn't a big deal for one or two calls,
    but all of the rest of the thermodynamic model is built
    on this so compiling this allows everything else to be
    compiled too.
    """
    prss = p
    uuu = 3.0*k0
    vvv = (3.0/2.0)*(kp - 1.0)

    f = -(1/4)*(-4*uuu*vvv**3-3*uuu*vvv**2)/(uuu*vvv**3)+(1/2)*((1/4)*(-4*uuu*vvv**3-3*uuu*vvv**2)**2 / \
        (uuu**2* vvv**6)-(9*uuu*vvv**2-6*prss+6*uuu*vvv**3+6*uuu*vvv)/(uuu* vvv**3)+ \
        (36*uuu**2*vvv**3*prss+21*uuu**2*vvv**4*prss+8*uuu**3*vvv**3-8* prss**3+24*uuu*vvv**3*prss**2 \
         +24*uuu*vvv*prss**2-6*uuu**2*vvv**2* prss+36*uuu*vvv**2*prss**2+(-(-72*uuu**4*vvv**3-720 \
        *uuu**3*vvv**3*prss-756*uuu**2*vvv*prss**2-432*uuu**3*vvv**4*prss-3060*uuu**2*vvv**3* prss**2-3024 \
        *prss**3*uuu*vvv-2664*uuu**2*vvv**4*prss**2-1296*uuu**2* vvv**2*prss**2-7920*uuu*vvv**2*prss**3 \
        +288*uuu*prss**3-825*uuu**2* vvv**5*prss**2-9576*prss**3*uuu*vvv**3+1152*vvv*prss**4-6000*uuu* vvv**4 \
        *prss**3+576*vvv**2*prss**4+1152*prss**4-2304*uuu*vvv**5* prss**3+192*prss**4*vvv**3-512*uuu*vvv**6 \
        *prss**3)/vvv)**(1/2)*uuu* vvv**2)**(1/3)/(uuu*vvv**3)+2*(-(uuu**2*vvv**2)-6*uuu*vvv**2*prss+2* prss**2 \
        -4*uuu*vvv**3*prss-4*uuu*vvv*prss)/(uuu*vvv**3*(36*uuu**2* vvv**3*prss+21*uuu**2*vvv**4*prss+8*uuu**3 \
        *vvv**3-8*prss**3+24*uuu* vvv**3*prss**2+24*uuu*vvv*prss**2-6*uuu**2*vvv**2*prss+36*uuu*vvv**2* prss**2 \
        +(-(-72*uuu**4*vvv**3-720*uuu**3*vvv**3*prss -756*uuu**2*vvv* prss**2-432*uuu**3*vvv**4*prss \
        -3060*uuu**2*vvv**3*prss**2-3024*prss**3* uuu*vvv-2664*uuu**2*vvv**4*prss**2-1296*uuu**2*vvv**2*prss**2 \
        -7920*uuu* vvv**2*prss**3+288*uuu*prss**3-825*uuu**2*vvv**5*prss**2-9576*prss**3* uuu*vvv**3+1152*vvv*prss**4 \
        -6000*uuu*vvv**4*prss**3+576*vvv**2* prss**4+1152*prss**4-2304*uuu*vvv**5*prss**3+192*prss**4*vvv**3 \
        -512* uuu*vvv**6*prss**3)/vvv)**(1/2)*uuu*vvv**2)**(1/3))-(-3*uuu*vvv**2+2* prss-2*uuu*vvv**3-2*uuu*vvv)/( \
        uuu*vvv**3))**(1/2)-(1/2)*((1/2)*(-4* uuu*vvv**3-3*uuu*vvv**2)**2/(uuu**2*vvv**6)-(9*uuu*vvv**2-6*prss+6*uuu \
        * vvv**3+6*uuu*vvv)/(uuu*vvv**3)-(36*uuu**2*vvv**3*prss+21*uuu**2*vvv**4* prss+8*uuu**3*vvv**3-8*prss**3 \
        +24*uuu*vvv**3*prss**2+24*uuu*vvv* prss**2-6*uuu**2*vvv**2*prss+36*uuu*vvv**2*prss**2+(-(-72*uuu**4* vvv**3 \
        -720*uuu**3*vvv**3*prss -756*uuu**2*vvv*prss**2-432*uuu**3*vvv**4* prss -3060*uuu**2*vvv**3*prss**2 \
        -3024*prss**3*uuu*vvv-2664*uuu**2* vvv**4*prss**2-1296*uuu**2*vvv**2*prss**2-7920*uuu*vvv**2*prss**3+288 \
        * uuu*prss**3-825*uuu**2*vvv**5*prss**2-9576*prss**3*uuu*vvv**3+1152*vvv* prss**4-6000*uuu*vvv**4*prss**3 \
        +576*vvv**2*prss**4+1152*prss**4-2304* uuu*vvv**5*prss**3+192*prss**4*vvv**3-512*uuu*vvv**6 \
        * prss**3)/vvv)**(1/2)*uuu*vvv**2)**(1/3)/(uuu*vvv**3)-2*(-(uuu**2* vvv**2)-6*uuu*vvv**2*prss+2*prss**2 \
        -4*uuu*vvv**3*prss-4*uuu*vvv*prss ) \
        / (uuu*vvv**3*(36*uuu**2*vvv**3*prss+21*uuu**2*vvv**4*prss+8* uuu**3*vvv**3-8*prss**3+24*uuu*vvv**3 \
        *prss**2+24*uuu*vvv*prss**2-6* uuu**2*vvv-2*prss+36*uuu*vvv**2*prss**2+(-(-72*uuu**4*vvv**3-720* uuu**3 \
        *vvv**3*prss -756*uuu**2*vvv*prss**2-432*uuu**3*vvv**4*prss -3060* uuu**2*vvv**3*prss**2-3024*prss**3 \
        *uuu*vvv-2664*uuu**2*vvv**4* prss**2-1296*uuu**2*vvv**2*prss**2-7920*uuu*vvv**2*prss**3+288*uuu* prss**3 \
        -825*uuu**2*vvv**5*prss**2-9576*prss**3*uuu*vvv**3+1152*vvv* prss**4-6000*uuu*vvv**4*prss**3 \
        +576*vvv**2*prss**4+1152*prss**4-2304* uuu*vvv**5*prss**3+192*prss**4*vvv**3-512*uuu*vvv**6 \
        * prss**3)/vvv)**(1/2)*uuu*vvv**2)**(1/3))+(-3*uuu*vvv**2+2*prss-2*uuu* vvv**3-2*uuu*vvv)/(uuu*vvv**3) \
        +((9*uuu*vvv**2-6*prss+6*uuu*vvv**3+6* uuu*vvv)*(-4*uuu*vvv**3-3*uuu*vvv**2)/(uuu**2*vvv**6)-2 \
        *(-4*uuu* vvv**3-9*uuu*vvv**2-12*uuu*vvv-6*uuu)/(uuu*vvv**3)-(1/4)*(-4*uuu* vvv**3-3*uuu*vvv**2)**3/(uuu**3 \
        *vvv**9)) /((1/4)*(-4*uuu*vvv**3-3*uuu* vvv**2)**2/(uuu**2*vvv**6)-(9*uuu*vvv**2-6*prss+6*uuu*vvv**3 \
        +6*uuu* vvv)/(uuu*vvv**3)+(36*uuu**2*vvv**3*prss+21*uuu**2*vvv**4*prss+8* uuu**3*vvv**3-8*prss**3 \
        +24*uuu*vvv**3*prss**2+24*uuu*vvv*prss**2-6* uuu**2*vvv**2*prss+36*uuu*vvv**2*prss**2+(-(-72*uuu**4*vvv**3 \
        -720* uuu**3*vvv**3*prss -756*uuu**2*vvv*prss**2-432*uuu**3*vvv**4*prss -3060 \
        * uuu**2*vvv**3*prss**2-3024*prss**3*uuu*vvv-2664*uuu**2*vvv**4* prss**2-1296*uuu**2*vvv**2*prss**2 \
        -7920*uuu*vvv**2*prss**3+288*uuu* prss**3-825*uuu**2*vvv**5*prss**2-9576*prss**3*uuu*vvv**3+1152*vvv \
        * prss**4-6000*uuu*vvv**4*prss**3+576*vvv**2*prss**4+1152*prss**4-2304* uuu*vvv**5*prss**3+192*prss**4 \
        *vvv**3-512*uuu*vvv**6* prss**3)/vvv)**(1/2)*uuu*vvv**2)**(1/3)/(uuu*vvv**3)+2*(-(uuu**2* vvv**2) \
        -6*uuu*vvv**2*prss+2*prss**2-4*uuu*vvv**3*prss-4*uuu*vvv* \
        prss ) /(uuu*vvv**3*(36*uuu**2*vvv**3*prss+21*uuu**2*vvv**4 \
        *prss+8* uuu**3*vvv**3-8*prss**3+24*uuu*vvv**3*prss**2+24*uuu*vvv*prss**2-6* uuu**2*vvv**2*prss \
        +36*uuu*vvv**2*prss**2+(-(-72*uuu**4*vvv**3-720* uuu**3*vvv**3*prss -756*uuu**2*vvv*prss**2 \
        -432*uuu**3*vvv**4*prss -3060* uuu**2*vvv**3*prss**2-3024*prss**3*uuu*vvv-2664*uuu**2*vvv**4 \
        * prss**2-1296*uuu**2*vvv**2*prss**2-7920*uuu*vvv**2*prss**3+288*uuu* prss**3-825*uuu**2*vvv**5 \
        *prss**2-9576*prss**3*uuu*vvv**3+1152*vvv* prss**4-6000*uuu*vvv**4*prss**3+576*vvv**2*prss**4 \
        +1152*prss**4-2304* uuu*vvv**5*prss**3+192*prss**4*vvv**3-512*uuu*vvv**6* prss**3)/vvv)**(1/2) \
        *uuu*vvv**2)**(1/3))-(-3*uuu*vvv**2+2*prss-2*uuu* vvv**3-2*uuu*vvv)/(uuu*vvv**3))**(1/2))**(1/2)
    
    v = f**3 * v0
    return v 


@numba.jit
def vinet_eos_pressure(v, v0, k0, kp):
    """
    Return the pressure at some volume according to K14 eqn (3)
    
    Given 1 bar reference volume (v0), bulk modulus (k0),
    and its pressure derivative (kp) return the pressure
    at volume v. These should normally all be values
    at 298 K (thermal expansion is added later). Units 
    of the returned pressure and k0 are GPa, w and v0 
    are cm^3/mol, and kp is dimensionless. 
    """
    x = (v/v0)**(1.0/3.0)
    p = 3.0 * k0 * x**-2 * (1.0 - x) * np.exp(
           1.5 * (kp - 1.0) * (1.0 - x))
    return p


@numba.jit
def thermal_expansion(v, v0, a0, ag0, k):
    """
    Parameterisation of thermal expansion with pressure
    
    Uses Anderson-Gruneisen parameter. v is the volume
    of interest (i.e. at some pressure), v0 is the reference
    volume, a0 is the reference thermal expansivity, ag0
    is the 1 bar Anderson-Gruneisen parameter, k is a 
    dimensionless parameter.
    EQ5 of K14
    """
    a = a0 * np.exp(-1.0 * (ag0/k) * (1.0 - (v / v0)**k))
    return a


@numba.jit
def expand_volume(v, t, v0, a0, ag0, k):
    """
    Calculate thermal expansion and apply to 'cold' volume by integration
    
    v should be volume at 298 K. Integrate alpha = 1/V dV/dT to 
    find v at temperature T.
    """
    
    # Thermal expansion is 1/V dV/dT so integrate to find V
    # assume thermal expanion is not temperature dependent
    # but note that volume expansion is large so we need to 
    # solve differential equation above (analytical solution now)
    a = thermal_expansion(v, v0, a0, ag0, k)    
    v = v + v * (np.exp(a*t - a*298.0) - 1.0)
    
    return v


@numba.jit
def end_member_free_energy(p, t, a, b, c, d, e, f, v0, k0, kp, a0, ag0, k):
    dp = 0.5
    ps = np.arange(p+dp, step=dp) # in GPa
    vs = vinet_eos_volumes(ps, v0, k0, kp)
    vs = expand_volume(vs, t, v0, a0, ag0, k) # in cm^3/mol
    # Worry about units here -> GPa and cm^3 converted to Pa and m^3
    vdp = np.trapz(vs * 1.0E-6, ps * 1.0E9) 
    g_onebar = free_energy_onebar(t, a, b, c, d, e, f)
    g_pt = g_onebar + vdp                     # Result in J
    return g_pt


@numba.jit
def free_energy_onebar(t, a, b, c, d, e, f):
    g_onebar = a + b*t + c*t*np.log(t) + \
               d*(t**2.0) + e*(t**-1.0) + f*(t**0.5)
    return g_onebar


@numba.jit
def end_member_delta_g(t, p, liquid_a, liquid_b, liquid_c, liquid_d, liquid_e, liquid_f,
                       liquid_v0, liquid_k0, liquid_kp, liquid_a0, liquid_ag0, liquid_k,
                       solid_a, solid_b, solid_c, solid_d, solid_e, solid_f,
                       solid_v0, solid_k0, solid_kp, solid_a0, solid_ag0, solid_k):
    """
    Difference in free energy (in J/mol) between solid and liquid at temperature t (in K)
    and pressure p (in GPa)
    
    NB: order of t and p reversed for use in end_member_melting_temperature
    """
    solid_g = end_member_free_energy(p, t, solid_a, solid_b, solid_c, solid_d, solid_e, solid_f,
                       solid_v0, solid_k0, solid_kp, solid_a0, solid_ag0, solid_k)
    liquid_g = end_member_free_energy(p, t, liquid_a, liquid_b, liquid_c, liquid_d, liquid_e, liquid_f,
                       liquid_v0, liquid_k0, liquid_kp, liquid_a0, liquid_ag0, liquid_k)
    return solid_g - liquid_g

    
# Cannot jit due to brentq
def end_member_melting_temperature(p, liquid_a, liquid_b, liquid_c, liquid_d, liquid_e, liquid_f,
                       liquid_v0, liquid_k0, liquid_kp, liquid_a0, liquid_ag0, liquid_k,
                       solid_a, solid_b, solid_c, solid_d, solid_e, solid_f,
                       solid_v0, solid_k0, solid_kp, solid_a0, solid_ag0, solid_k):
    """
    Find the melting temperature of end members (in K) at pressure p (in GPa)
    
    Works by finding the point where the free energies are equal.
    """
    tm = spo.brentq(end_member_delta_g, 298.0, 15000.0,
                    args=(p, liquid_a, liquid_b, liquid_c, liquid_d, liquid_e, liquid_f,
                          liquid_v0, liquid_k0, liquid_kp, liquid_a0, liquid_ag0, liquid_k,
                          solid_a, solid_b, solid_c, solid_d, solid_e, solid_f,
                          solid_v0, solid_k0, solid_kp, solid_a0, solid_ag0, solid_k))
    return tm

end_member_melting_temperatures = np.vectorize(end_member_melting_temperature)


@numba.jit
def chemical_potential(x, p, t, a, b, c, d, e, f, v0, k0, kp, a0, ag0, k):
    mu_0 = end_member_free_energy(p, t, a, b, c, d, e, f, v0, k0, kp, a0, ag0, k)
    activity = x # ideal solution - could implement non-ideal here 
                 # but not good at ICB pressure
    r = 8.31446261815324 # gas constant J/K/mol
    mu = mu_0 + r * t * np.log(activity)
    return mu