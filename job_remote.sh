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
module add gcc/10.2.0 # Using gcc 10.2.0 results in a cmake warning. The project compiles and runs, but more research needs to be made.
module add cmake/3.13.4

# Compile the source code into an executable
gcc -Ofast -fopenmp -lm common.c nn-main.c -o build/CAP-PLAB2021.exe

# Parse parameters
filename=$1
epochs=$2
numIn=$3
numHid=$4
numOut=$5

# Execute
perf stat -d ./build/CAP-PLAB2021.exe $epochs $numIn $numHid $numOut 2>&1
perf record -o $filename.data ./build/CAP-PLAB2021.exe $epochs $numIn $numHid $numOut 2>&1
