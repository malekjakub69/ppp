#!/usr/bin/env bash
#SBATCH --account=DD-23-135
#SBATCH --job-name=PPP-lab8
#SBATCH -p qcpu_exp
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=36

ml Score-P

rm farm *.o
rm -rf prof_*

scorep-mpiicpc -O3 -std=c++17 farm.cpp -o farm

# Export filter


# Enter execution commands


