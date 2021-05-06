#!/bin/bash
# This script launches an sbatch job to a SLURM cluster
# Arguments (listed in read order):
#   (str) filename: name of the output and error files passed to the sbatch command
#   (int) epochs: number of training iterations
#   (int) numIn: number of input neurons
#   (int) numHid: number of hidden neurons
#   (int) numOut: number of output neurons
#
# Parse parameters
filename=$1
epochs=$2
numIn=$3
numHid=$4
numOut=$5

# Launch sbatch
sbatch -N 4 -n 38 -o $filename.out -e $filename.err job.sh $filename $epochs $numIn $numHid $numOut
