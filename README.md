# Supplementary material: modelling the F-layer in Earth’s core as a non-equilibrium slurry

These files are part of the supplementary material for the manuscript
"Modelling the F-layer in Earth’s core as a non-equilibrium slurry"
by A. M. Walker, C. J. Davies, A. J. Wilson and M. I. Bergman
to be submitted to *Proceedings of the Royal Society A*. A preprint
of this manuscript can be found [EarthArXiv](https://eartharxiv.org/).

The files are source code files in python which implement the
model described in the manuscript along with Jupyter notebook
files that demonstrate aspects of the model, provide additional
information on its derivation, and generate the figures shown in
the paper. All software is made available under an MIT license
which permits reuse subject to conditions found in the
[LICENSE](./LICENSE) file. 

## Jupyter notebooks

This repository contains the following notebooks implementing or
describing the model:

* **[1_thermodynamic_model.ipynb](./1_thermodynamic_model.ipynb)**: details of the thermodynamic model reproducing key figures from Komabayashi 2014, and Figure 2 from the manuscript. The notebook uses **thermodynamic_model.py** and **feo_thermodynamics.py**, two python modules implementing the model of Komabayashi 2014.
* **[2_falling.ipynb](./2_falling.ipynb)**: details of the calculation of the falling velocity of the particle and boundary layer analysis. Code to generate Figure 3. The notebook uses **falling.py**, a python module implementing the self-consistent calculation of falling velocity and Re, and the calculation of boundary layer thickness.
* **[3_growth.ipynb](./3_growth.ipynb)**: Growth rate calculation with and without a boundary layer. Code to generate Figure 4. The notebook makes use of **growth.py**, a python module implementing the calculation of crystal growth rate as well as feo_thermodynamics.py.
* **[4_particle_evolution.ipynb](./4_particle_evolution.ipynb)**: Calculation of the coupled growth rate and falling velocity for a single particle. Generation of Figure 5. Makes use of **particle_evolution.py**, feo_thermodynamics.py, growth.py and falling modules. 
* **[5_nucleation.ipynb](./5_nucleation.ipynb)**: Calculation of nucleation rate and generation of Figure 6. Makes use of **nucleation.py** and feo_thermodynamics.py modules.
* **[6_prem.ipynb](./6_prem.ipynb)**: Demonstration of an implementation of the provisional reference Earth model used to calculate pressure and gravitational acceleration in the F-layer. The implementation is in the **earth_model.py** and **peice_poly.py** modules.
* **[7_f_layer_solution.ipynb](./7_f_layer_solution.ipynb)**: This notebook shows the calculation of a single case of our F-layer model including nucleation, falling and growth of particles, and evaluation of output number density of crystals. The notebook is not used to generate any figures in the paper, but may be of use for exploring model behaviour.
* **[8_plot_flayer_figures.ipynb](./8_plot_flayer_figures.ipynb)**: Plot summary figures (Figures 1(c), 7 and 8) showing the range of "Earth like" models of the F-layer. This notebook relies of a collection of model runs to be pre-computed (see notes below). Table 2 of the SI file is also created by this notebook. The notebook makes use of the **plotting.py** and **bulk_case_runner.py** modules.

## F-layer cases

For the cases reported in the manuscript (in Table 1 of the SI, and in Figures 7 amd 8) we ran the models in "batch" mode,
stored the results, and use the final notebook to plot these. The command line programme used to run a single set of cases 
(fixed parameters, but a grid search over DT and DX looking for "Earth like" models) is `layer_models.py` which is intended
to run in a directory and populate this with output for different temperature and composition profiles. These directories
(and the input parameter sets) are set up by `cases/run_cases.py` which copies and modifies the files in `cases/template`
before launching the shell script `cases/template/run.sh`. It takes a week or two (on a M1 Mac Min) to run all cases
and this process generates approximatly 4 GB of data once compressed. We do not include this data alongside the code.

## Other files and further development

Some of the code (and notebooks inside `development_work`) are not used to create the models presented in this manuscript.
These represent several alternitive lines of development and possible future work. Notably, much of the code in the repository
is in need of refactoring ahead of further development of the work presented here. 
