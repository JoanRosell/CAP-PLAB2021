#!/bin/bash

# Identify assigned node
hostname
echo

# Load required modules
module add gcc/8.2.0 # Using gcc 10.2.0 results in a cmake warning. The project compiles and runs, but more research needs to be made.
module add cmake/3.13.4

# Compile the source code into an executable
./compile.sh

# Execute
./build/CAP-PLAB2021.exe
