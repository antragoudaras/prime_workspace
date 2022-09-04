#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/results_photonics_positive_worst_80_split
EXCEL_RESULTS_DIR="$PROJECT_DIR"/photonics_optimized_positive_results_worst_80_split

mkdir -p "$JOB_RESULTS_DIR"
mkdir -p "$EXCEL_RESULTS_DIR"

JOB_NAME=train_photonics_60000_grad_steps_batch_size_375_worst_80_split_positive

sbatch --job-name "$JOB_NAME" "$SRC_DIR"/train_photonics_positive.sbatch