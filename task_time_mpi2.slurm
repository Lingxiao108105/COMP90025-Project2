#!/bin/bash

# Partition for the job:
#SBATCH --partition=physical

# The name of the job:
#SBATCH --job-name="mpi time 2"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8

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
#SBATCH --output=output/output30_eva_2.out

# The modules to load:
module load gcc/11.2.0 
module load openmpi/4.1.1

# compile
mpicc -fopenmp time_evaluate.c -o time_evaluate

for i in {1..5}  
do
srun -n 2 ./time_evaluate 2 8 input/input30.txt
done

for i in {1..5}  
do
srun -n 2 ./time_evaluate 3 8 input/input30.txt
done

my-job-stats -a -n -s
