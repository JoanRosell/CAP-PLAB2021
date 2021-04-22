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
module load gcc/8.2.0
module load cmake/3.13.4
module load openmpi/3.0.0
module load tau/2.29

# Compile the source code into an executable
export TAU_MAKEFILE=/soft/tau-2.29/x86_64/lib/Makefile.tau-mpi
export TAU_OPTIONS=-optCompInst
export CC=/soft/tau-2.29/x86_64/bin/tau_cc.sh

./compile.sh

# Parse parameters
filename=$1
epochs=$2
numIn=$3
numHid=$4
numOut=$5

# Execute
mpirun -n 1 ./build/CAP-PLAB2021.exe $epochs $numIn $numHid $numOut
pprof

