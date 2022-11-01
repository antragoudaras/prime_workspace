#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
JOB_RESULTS_DIR="$PROJECT_DIR"/photonics_positive_mixed_80_split_logs
EXCEL_RESULTS_DIR="$PROJECT_DIR"/november_photonics_optimized_positive_results_mixed_80_split

mkdir -p "$JOB_RESULTS_DIR"
mkdir -p "$EXCEL_RESULTS_DIR"

CQL_ALPHA=5.0
INFEASIBLE_ALPHA=5.0
NUM_VOTES=1
TRAIN_STEPS=120001

JOB_NAME=PRIME_photonics_${TRAIN_STEPS}_grad_steps_mixed_split_${NUM_VOTES}_votes_${CQL_ALPHA}_cql_alpha_${INFEASIBLE_ALPHA}_infeasible_alpha

sbatch --job-name "$JOB_NAME" "$SRC_DIR"/train_photonics_positive_mixed_split.sbatch --cql_alpha ${CQL_ALPHA} --infeasible_alpha ${INFEASIBLE_ALPHA} --num_votes ${NUM_VOTES} --train_steps ${TRAIN_STEPS}