#!/bin/bash

# Partition for the job:
#SBATCH --partition=physical
#SBATCH --constraint=physg5

# The name of the job:
#SBATCH --job-name="mpi"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-01:00:00

# The project ID which this job should run under:
#SBATCH --account="punim0520"

# Use this email address:
#SBATCH --mail-user=lingxiao1@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# ends successfully
#SBATCH --mail-type=END

#output file
#SBATCH --output=output/output30mpi_8.out

# The modules to load:
module load gcc/11.2.0 
module load openmpi/4.1.1

# compile
mpicc -fopenmp solution.c -o solution

for i in {1..5}  
do
srun -n 8 ./solution 2 2 input/input30.txt
done
my-job-stats -a -n -s
