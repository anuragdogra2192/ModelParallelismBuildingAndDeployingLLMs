#!/bin/bash

#SBATCH -N 2
#SBATCH --job-name=dli_secondSlurmJob
#SBATCH -o /dli/nemo/logs/%j.out
#SBATCH -e /dli/nemo/logs/%j.err

srun -l /dli/code/test.sh   
