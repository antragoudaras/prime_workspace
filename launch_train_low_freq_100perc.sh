#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/results_low_freq_100perc
mkdir -p "$JOB_RESULTS_DIR"

JOB_NAME=extented_low_freq_train_prime_60000_grad_steps_batch_size_1000_100percent_split

sbatch --job-name "$JOB_NAME" "$SRC_DIR"/train_low_freq_100perc.sbatch