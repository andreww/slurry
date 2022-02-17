# Supplementary material: modelling the F-layer in Earth’s core as a non-equilibrium slurry

These files are the supplementary material for the manuscript
"Modelling the F-layer in Earth’s core as a non-equilibrium slurry"
by A. M. Walker, C. J. Davies, A. J. Wilson and M. I. Bergman
to be submitted to *Proceedings of the Royal Society A*. A preprint
of this manuscript can be found [here when done](https://www.example.com).

The files are source code files in python which implement the
model described in the manuscript along with Jupyter notebook
files that demonstrate aspects of the model, provide additional
information on its derivation, and generate the figures shown in
the paper. All software is made available under an MIT license
which permits reuse subject to conditions found in the
[LICENSE](./LICENSE) file. 

## Table of contents

This repository contains the following files implementing or
describing the model:

* **[thermodynamic_model.ipynb](./thermodynamic_model.ipynb)**: details of the thermodynamic model reproducing key figures from Komabayashi 2014, and Figure 1 from the manuscript.
* **thermodynamic_model.py** and **feo_thermodynamics.py**: python modules implementing the model of Komabayashi 2014.
* **[falling.ipynb](./falling.ipynb)**: details of the calculation of the falling velocity of the particle and boundary layer analysis. Code to generate Figure 2.
* **falling.py**: python module implementing the self-consistent calculation of falling velocity and Re, and the calculation of boundary layer thickness.
* **new_growth.ipynb** and **growth_with_boundary_layer.ipynb**: Growth rate calculation. Needs significant cleanup. Figure 3
* **particle_evolution.ipynb**: single particle calcuation. Needs significant cleanup. Figure 4.
* **new_nucleation.ipynb**: needs cleanup. Needs to make new figure for paper. remove restricted volume stuff.
* **PREM**
* **F-layer solution**
* **cases**

* **[intro.ipynb](./intro.ipynb)**: a jupyter notebook containing the same information as this file