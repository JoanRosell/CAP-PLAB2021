#!/bin/bash
#SBATCH --exclusive
# This script compiles and executes the project
# Arguments (listed in read order):
#   (int) epochs: number of training iterations
#   (int) numIn: number of input neurons
#   (int) numHid: number of hidden neurons
#   (int) numOut: number of output neurons
#

# Load required modules
module unload gcc
module load gcc/10.2.0
module load cmake/3.13.4
module load openmpi/3.0.0
module load tau/2.29

# TAU exports
export TAU_MAKEFILE=/soft/tau-2.29/x86_64/lib/Makefile.tau-mpi
export TAU_OPTIONS=-optCompInst
#export TAU_TRACE=1

# Select the compilation variable for CMake
export CC=$(which tau_cc.sh)

# Remove the old build system
rm -rf build/* 

# Compile using CMake
./compile.sh

# Parse parameters
filename=$1
epochs=$2
numIn=$3
numHid=$4
numOut=$5

# Execute
mpirun ./build/CAP-PLAB2021.exe $epochs $numIn $numHid $numOut

# Process the trace files
#tau_treemerge.pl
#tau2slog2 tau.trc tau.edf -o tau.slog2
pprof

