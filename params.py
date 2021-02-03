#!/usr/bin/env python

import numpy as np

# Module to define some parameters. These were global (yuck yuck yuck)
# and really should be somewhere else. At least this way we know what we use...

# Define parameters
secinyr = 60.0*60.0*24.0*365.0
k     = 100.0                    # thermal conductivity of solid iron 
rhos  = 12700                    # solid density
rhol  = 12100                    # liquid density
cp    = 750                      # specific heat
kappa = k/(rhol*cp)              # thermal diffusivity of liquid
Dliq  = 1e-9                     # self-diffusion of O in liquid
Dsol  = 1e-12                    # self-diffusion of O in solid
mu    = 1e-6                     # kinematic viscosity
drho  = rhos-rhol                # density difference
g     = 3.7                      # gravity at ICB from PREM
Vic      = 4 * np.pi * 1221000.0**3 / 3.0  # IC volume
kb       = 1.380648e-23          # Boltzmann constant
secingyr = 3.1536e+16      
ri       = 1221e3
rf       = ri + 200e3

Tm = 5500.0                      # Default melting temperature at the ICB

Vic = 4.0*np.pi*ri**3 / 3.0
Vsl = 4.0*np.pi*(rf**3 - ri**3) / 3.0

Pr = mu / kappa
Sc = mu / Dliq
print("Pr = ", Pr, ' Sc = ', Sc)

r = np.logspace(-8,3,num=100)     # Array of particle sizes in metres