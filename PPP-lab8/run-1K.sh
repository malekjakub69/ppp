#!/usr/bin/env bash
#SBATCH --account=DD-23-135
#SBATCH --job-name=PPP-lab8
#SBATCH -p qcpu_exp
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=128

ml Scalasca Vampir Cube

rm farm *.o
rm -rf prof_*

scorep-mpicxx -O3 -std=c++17 farm.cpp -o farm

# Export filter


# Enter execution commands


