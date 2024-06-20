#!/bin/bash

SLURRY_HOME='../..'

python $SLURRY_HOME/layer_models.py -d detailed_output parameters.yaml grid.in grid.csv > grid.out 2> grid.err
