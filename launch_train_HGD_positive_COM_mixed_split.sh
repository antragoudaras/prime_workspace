#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/HGD_positive_COM_mixed_split_logs
OPTIMIZED_DATASET_DIR="$PROJECT_DIR"/HGD_positive_COM_optimized_params

mkdir -p "$JOB_RESULTS_DIR"
mkdir -p "$OPTIMIZED_DATASET_DIR"

JOB_NAME=train_HGD_60000_grad_steps_votes_one_positive_COM_mixed_split

sbatch --job-name "$JOB_NAME" "$SRC_DIR"/train_HGD_positive_COM_mixed_split.sbatch