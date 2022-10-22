#!/bin/bash

# Partition for the job:
#SBATCH --partition=physical

# The name of the job:
#SBATCH --job-name="parallel"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-10:00:00

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
#SBATCH --output=output/output30par_all.out

# The modules to load:
module load gcc/11.2.0 
module load openmpi/4.1.1

# compile
mpicc -fopenmp solution.c -o solution

#sequential
for thread_number in {2..16}
do
    for i in {1..5}  
    do
    srun -n 1 ./solution 1 $thread_number input/input30.txt
    done
done
my-job-stats -a -n -s
