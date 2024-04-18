#!/usr/bin/env bash
#SBATCH --account=DD-23-135
#SBATCH --job-name=PPP-lab8
#SBATCH -p qcpu_exp
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=128

ml Scalasca Vampir Cube

rm farm *.o
rm -rf scan_*

scorep --user mpic++ -O3 -std=c++17 farm.cpp -o farm

# Export your filter
export SCOREP_FILTERING_FILE=filter.filt

# Export PAPI counters
export SCOREP_METRIC_PAPI="PAPI_TOT_CYC,PAPI_TOT_INS"

# Enable memory profiling
export SCOREP_MEMORY_RECORDING=true
export SCOREP_MPI_MEMORY_RECORDING=true

# Enter your command
