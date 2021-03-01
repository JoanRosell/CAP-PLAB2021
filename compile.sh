#!/bin/bash


hostname
echo

cmake -E make_directory build
cmake -E chdir build cmake ..
cmake -E chdir build cmake --build .

