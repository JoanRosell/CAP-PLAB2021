#!/bin/bash
#SBATCH --exclusive
#SBATCH --partition=cuda.q
#SBATCH --gres=gpu:GeForceRTX3080:1 #aolin24
#SBATCH -w aolin24
#
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
#module load cmake/3.13.4
module add cuda/11.2
module load nvidia-hpc-sdk/21.2

# Parse parameters
filename=$1
epochs=$2
numIn=$3
numHid=$4
numOut=$5

# Compile the source code into an executable
export CC=$(which nvcc)

./compile.sh

#nsys nvprof --print-gpu-trace build/CAP-PLAB2021.exe $epochs > $filename.prof
build/CAP-PLAB2021.exe $epochs

