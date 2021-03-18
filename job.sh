#!/bin/bash
#SBATCH --exclusive
# This script compiles and executes the project
# Arguments (listed in read order):
#   (int) epochs: number of training iterations
#   (int) numIn: number of input neurons
#   (int) numHid: number of hidden neurons
#   (int) numOut: number of output neurons
#
# Identify assigned node
hostname
echo
lscpu
echo

# Load required modules
module add cmake/3.13.4

# Compile the source code into an executable
export CC=/soft/gcc-10.2.0/bin/gcc
./compile.sh

# Parse parameters
filename=$1
epochs=$2
numIn=$3
numHid=$4
numOut=$5

# Execute
perf stat -d ./build/CAP-PLAB2021.exe $epochs $numIn $numHid $numOut 2>&1
perf record -o $filename.data ./build/CAP-PLAB2021.exe $epochs $numIn $numHid $numOut 2>&1
