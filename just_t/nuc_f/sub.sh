# Running in current working directory (-cwd), exporting variables (-V) 
#$ -cwd -V

# Name
#$ -N slurry-T-F

# Time
#$ -l h_rt=48:00:00

# Memory
#$ -l h_vmem=1G

#s -q tartarus


module load anaconda
source activate slurry

export PYTHONPATH=/home/home02/earcd/slurry_stuff/slurry:$PYTHONPATH

/home/home02/earcd/slurry_stuff/slurry/run_flayer.py -f slurry-T-A.pickle params.yaml > slurry-T-A.out 2>&1

