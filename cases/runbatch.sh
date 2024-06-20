#!/usr/bin/bash

# This runs everything (in a directory called i0 something)
# at once in the background. Probably not what you want to do
# unless you have plenty of cores (and memory)

for d in i0* ; do cd $d; echo $d; ./run.sh& ; cd .. ; done

