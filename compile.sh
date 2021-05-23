#!/bin/bash

rm -rf build/*
# THIS IS A QUICK FIX UNTIL CMAKE IS AVAIABLE AT THE AOLIN CLUSTER
#cmake -E make_directory build
#cmake -E chdir build cmake ..
#cmake -E chdir build cmake --build .
nvcc -O3 common.c nn-main.cu -o build/CAP-PLAB2021.exe

