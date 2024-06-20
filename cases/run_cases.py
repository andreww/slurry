#!/usr/bin/env python

import subprocess
import pathlib
import shutil

# A script to generate input files and run F-layer cases
# NB: this runs in serial and takes about a week to finish.

# Lists of parameters - by default, I suppose we could take these from command line
i0 = [1.0E-7, 1.0E-8, 1.0E-9, 1.0E-10, 1.0E-11]
dl_mu_pairs = [(1.0E-11, 2.0E-6), (1.0E-11, 3.0E-5), (1.0E-11, 6.0E-8), (1.0E-12, 2.0E-6), (1.0E-9, 2.0E-6)]

# Template
template_path = pathlib.Path("template")

def dir_name(myi0, dl, mu):
    name = f"i0_{myi0:1.0E}_dl_{dl:1.0E}_mu_{mu:1.0E}"
    return name

def replace_params(infile, outfile, i0, dl, mu):
    """
    NB: will overwrite outfile
    """
    with open(infile, 'r') as infh:
        dat = infh.read()

    dat = dat.replace('# This is the "Light blue square" from the paper\n', '')
    dat = dat.replace("i0 : 1.0e-11", f"i0 : {i0}")
    dat = dat.replace("chemical_diffusivity : 1.0e-9", f"chemical_diffusivity : {dl}")
    dat = dat.replace("kinematic_viscosity : 2.0e-6", f"kinematic_viscosity : {mu}")

    with open(outfile, 'w+') as outfh:
        outfh.write(dat)


def main(i0=i0, dl_mu_pairs=dl_mu_pairs):
    for myi0 in i0:
        for mydl_mu in dl_mu_pairs:
            dl = mydl_mu[0]
            mu = mydl_mu[1]
            target_path = pathlib.Path(dir_name(myi0, dl, mu))
            print(f"Working on {target_path}")
            if not target_path.exists():
                target_path = shutil.copytree(template_path, target_path)
                replace_params(target_path/"parameters.yaml", target_path/"parameters.yaml", myi0, dl, mu)
                print(" -> files ready, about to run")
                subprocess.run(["./run.sh"], cwd=target_path.resolve())
                print(" -> done")
            else:
                print(" -> skipped (exists)")
            

if __name__ == "__main__":
    main()