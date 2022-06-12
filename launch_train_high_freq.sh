#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/results_high_freq
mkdir -p "$JOB_RESULTS_DIR"

JOB_NAME=extented_high_freq_train_prime_55000_grad_steps

sbatch --job-name "$JOB_NAME" "$SRC_DIR"/train_high_freq.sbatch