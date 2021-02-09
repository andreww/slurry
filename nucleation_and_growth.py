import numpy as np
import phase_equilibrium
import params


def sun_velocity_fit(dt, tm, dhm, k0):
    """
    dT is the undercooling
    Tm is the melting temperature
    dmu is the difference in chemical potentials
    k0 is a rate constant
    r is the particle radius
    """
    t = tm - dt
    d_mu = (tm - t) * (dhm / tm) # K13 / first para of Sun SI
    kB = 1.380649e-23 #J/K
    v = k0 * (1.0 - np.exp(-d_mu/(params.kb * t)))
    return v


def tauv(dT, Tm):
    """
    hf    = enthalpy of fusion
    gamma = surface tension
    hc    = enthalpy correction - see DPA19, below eqn 2
    """
    hf    = 0.976e10
    gamma = 1.05
    prefac= 5.366215575895298e-46   # Other way up to Huguet
    
    hc   = 1.0 - 7.046e-5*dT
    T_Fe = dT + Tm
    num  = 16 * np.pi * Tm**2 * gamma**3
    den  = 3  * params.kb * hf**2 * hc**2 * T_Fe * dT**2
    fac  = num/den
    # For large fac the next line overflows. Need to fix!
    tau_v = prefac * np.exp(fac)
    return tau_v 


def tau_phase_TO(rad, T_abs, O_abs):
    
    """Calculate the phase equilibrium timescale tau_P = N*tau_N + tau_G
    as a function of absolute temperature T_abs and O concentration O_conc. 
    Assume a radius rad to get number of particles N from the equilibrium 
    volume fraction v_f. The total volume of solid V_f = v_f * Vsl, the 
    slurry volume. """

    tN      = np.zeros([len(T_abs), len(O_abs)])
    tG      = np.zeros([len(T_abs), len(O_abs)])
    ttot_TO = np.zeros([len(T_abs), len(O_abs)])

    time = 0 
    for t in T_abs:
        oo = 0
        for o in O_abs: 
            x_fe, x_lq, x_feo, phi_fe, phi_lq, phi_feo, x_liquidus, t_liquidus = \
            phase_equilibrium.phase_relations_komabayashi2004(t, o)
            v_f = phase_equilibrium.volume_fraction_solid(phi_fe, phi_lq, x_lq) # eq sol vol @ this T & c
        
            dT  = t - t_liquidus 
        
            if dT > 0: continue
                
            # Why do we need params.Vsl???
        
            V_f = v_f * params.Vsl                                   # total solid volume
            N   = 3 * V_f / (4 * np.pi * rad**3) 
        
            tN[time,oo] = N * tauv(dT, t_liquidus) / params.Vsl / params.secinyr
            G           = sun_velocity_fit(-dT, t_liquidus, params.cu_dhm, params.cu_k0) # CHK dT DEF POSITIVE!
            tG[time,oo] = rad/G/params.secinyr
        
            ttot_TO[time,oo] = tN[time,oo] + tG[time,oo]
                
            oo = oo + 1
        time = time + 1
    return ttot_TO, tN, tG


# Really should take the loops out of this function, and the function above and merge
# in a vectorised way over the input arguments.
def tau_phase_r(r, Temp, Oconc): 
    
    """Calculate the phase equilibrium timescale tau_P = N*tau_N + tau_G
    as a function of radius r. 
    Assume a radius rad to get number of particles N from the equilibrium 
    volume fraction v_f. The total volume of solid V_f = v_f * Vsl, the 
    slurry volume. """
    
    ttot_r = np.zeros(len(r))
    rr     = 0
    for rad in r:
        x_fe, x_lq, x_feo, phi_fe, phi_lq, phi_feo, x_liquidus, t_liquidus = \
        phase_equilibrium.phase_relations_komabayashi2004(Temp, Oconc)
        v_f = phase_equilibrium.volume_fraction_solid(phi_fe, phi_lq, x_lq) # eq sol vol @ this T & c
        
        dT  = Temp - t_liquidus 
        
        if dT > 0: continue
        
        V_f = v_f * params.Vsl                                   # total solid volume
        N   = 3 * V_f / (4 * np.pi * rad**3) 
        
        tN_r = N * tauv(dT, t_liquidus) / params.Vsl / params.secinyr
        G_r  = sun_velocity_fit(-dT, t_liquidus, params.cu_dhm, params.cu_k0) # CHK dT DEF POSITIVE!
        tG_r = rad/G_r/params.secinyr
    
        ttot_r[rr] = tN_r + tG_r
                
        rr = rr + 1
        
    return ttot_r