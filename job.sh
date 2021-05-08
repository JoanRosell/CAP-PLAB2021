#!/bin/bash
#SBATCH --exclusive
# This script compiles and executes the project in two steps:
#   1. Compile and run the project using TAU to trace the execution
#   2. Compile and run the project without TAU
#
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

# Parse parameters
filename=$1
epochs=$2
numIn=$3
numHid=$4
numOut=$5

# Tracing constants
TRACE_EPOCHS=5

# TAU exports
export TAU_MAKEFILE=/soft/tau-2.29/x86_64/lib/Makefile.tau-mpi
export TAU_OPTIONS=-optCompInst
export TAU_TRACE=1

# 1. Traced version:
# Select the compilation variable for CMake, in this case use the TAU wrappers
export CC=$(which tau_cc.sh)

# Remove the old build system
rm -rf build/* 

# Compile, run and process the output files
./compile.sh
mpirun ./build/CAP-PLAB2021.exe $TRACE_EPOCHS $numIn $numHid $numOut
tau_treemerge.pl
tau2slog2 tau.trc tau.edf -o $filename.slog2


# 2. Non-traced version
# Select the compilation variable for CMake, in this case use the standard MPI wrappers
export CC=$(which mpicc)

# Remove the old build system
rm -rf build/* 

# Compile and run
./compile.sh
mpirun ./build/CAP-PLAB2021.exe $epochs $numIn $numHid $numOut

