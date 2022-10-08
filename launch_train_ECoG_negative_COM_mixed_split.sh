#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/ECoG_negative_COM_mixed_split_logs
OPTIMIZED_DATASET_DIR="$PROJECT_DIR"/ECoG_negative_COM_optimized_params

mkdir -p "$JOB_RESULTS_DIR"
mkdir -p "$OPTIMIZED_DATASET_DIR"

JOB_NAME=train_ECoG_60000_grad_steps_batch_size_1000_negative_COM_mixed_split

sbatch --job-name "$JOB_NAME" "$SRC_DIR"/train_ECoG_negative_COM_mixed_split.sbatch