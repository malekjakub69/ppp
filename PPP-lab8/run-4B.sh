#!/usr/bin/env bash
#SBATCH --account=DD-23-135
#SBATCH --job-name=PPP-lab8
#SBATCH -p qcpu_exp
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=36

ml Scalasca Cube

rm farm *.o
rm -rf scan_*

scorep --user mpic++ -O3 -std=c++17 farm.cpp -o farm

#Enter your commands
export SCOREP_FILTERING_FILE=filter.filt
export SCOREP_METRIC_PAPI="PAPI_TOT_CYC,PAPI_TOT_INS"

export SCOREP_MEMORY_RECORDING=true
export SCOREP_MPI_MEMORY_RECORDING=true

# Enter your commands

